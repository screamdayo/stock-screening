"""
download.py
J-Quants APIからプライム銘柄リストと日足株価データを取得する。
このファイルは「データを取ってくること」だけに専念する。

取得方式は2種類:
1. 日付ベース（_get_bars_for_date）: 1日分・全銘柄をまとめて取る。
   日次スクリーニング（直近数十日分）向け。
2. 銘柄ベース（_get_bars_for_code）: 1銘柄・指定期間をまとめて取る。
   J-Quantsの仕様上、codeを指定するとfrom/toで期間をまとめて取得できるため、
   バックテストのような「長期間・全銘柄」を取りたい場合はこちらの方が
   リクエスト数・ページング回数の面で有利になりやすい。

関数一覧:
- get_price_history()            : 日次スクリーニング用（直近数十日、日付ベース）
- get_price_history_range()      : バックテスト用・初回一括取得（銘柄ベース、CSVキャッシュ対応）
- get_price_history_incremental(): バックテスト用・差分取得
                                    （既存CSVがあれば「その続きの日」だけ取得して追記する）
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

import config
from logger import get_logger

logger = get_logger(__name__)

HEADERS = {"x-api-key": config.JQUANTS_API_KEY}
BASE_URL = "https://api.jquants.com/v2"


def _request_with_retry(url, params=None):
    """
    requests.get() の共通ラッパー。
    以下の一時的な障害に対して自動リトライする:
    - タイムアウト（サーバーが応答しない）
    - 接続エラー（ネットワークが一瞬切れた等）
    - 502 / 503 / 504（サーバー側の一時的な障害でよく返るステータス）

    リトライしても解決しない障害（401認証エラー、400不正リクエストなど）は
    リトライせずそのまま例外を送出する。

    タイムアウト秒数・リトライ回数・待機時間は config.py で調整できる。
    """
    last_exception = None

    for attempt in range(1, config.API_MAX_RETRIES + 1):
        try:
            res = requests.get(
                url,
                headers=HEADERS,
                params=params,
                timeout=config.API_REQUEST_TIMEOUT_SECONDS,
            )

            if res.status_code in config.API_RETRYABLE_STATUS_CODES:
                logger.warning(f"リトライ対象のステータス {res.status_code} "
                      f"({attempt}/{config.API_MAX_RETRIES}回目): {url}")
                last_exception = requests.exceptions.HTTPError(
                    f"{res.status_code} Server Error: {res.text}"
                )
                _wait_before_retry(attempt)
                continue

            # リトライ対象外のステータス（400, 401, 404など）はそのまま返す。
            # 呼び出し側で res.raise_for_status() 等の判定を行う想定。
            return res

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            logger.warning(f"タイムアウト/接続エラー "
                  f"({attempt}/{config.API_MAX_RETRIES}回目): {e}")
            last_exception = e
            _wait_before_retry(attempt)
            continue

    # 全リトライを使い切っても失敗した場合はここに来る
    logger.warning(f"{config.API_MAX_RETRIES}回リトライしましたが失敗しました: {url}")
    raise last_exception


def _wait_before_retry(attempt):
    """リトライ前に少し待つ。回数を重ねるごとに待機時間を伸ばす（指数バックオフ）。"""
    wait_seconds = config.API_RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1))
    time.sleep(wait_seconds)


def _fetch_master_df():
    """
    J-Quantsのマスターデータ（銘柄一覧）を取得し、対象市場のみに絞ったDataFrameを返す。
    get_target_codes() と get_code_to_name_map() で同じデータを2回取得しないよう、
    共通処理としてここに切り出している。

    列名 "CoName" に会社名が入っている想定
    （J-QuantsのAPI仕様変更で列名が変わった場合はここだけ直せばよい）。
    """
    res = _request_with_retry(f"{BASE_URL}/equities/master")
    res.raise_for_status()
    data = res.json()

    df = pd.DataFrame(data["data"])
    df = df[df["MktNm"] == config.TARGET_MARKET].copy()
    df["Code"] = df["Code"].astype(str)

    return df


def get_target_codes():
    """
    対象市場（デフォルト：プライム）の銘柄コード一覧を取得する。
    優先株式など、末尾が特定の数字の銘柄は除外する。

    J-Quantsのマスターデータの銘柄コードは5桁（普通株は末尾に0を付加した形）で
    返ってくる。優先株式などの判定はこの5桁のままの末尾で行う必要があるため、
    「除外判定 → 4桁への変換」の順序で処理する
    （先に4桁化してしまうと末尾の判定基準が変わってしまうため注意）。
    """
    df = _fetch_master_df()
    target = df["Code"].tolist()

    # 除外判定は5桁のままの末尾で行う（優先株式は末尾が0以外になる）
    target = [
        code for code in target
        if not code.endswith(config.EXCLUDED_CODE_SUFFIXES)
    ]

    # 除外後に、価格データ側と一致するよう4桁表記へ変換する
    target = [_normalize_code(code) for code in target]

    logger.info(f"{config.TARGET_MARKET}銘柄数: {len(target)}件")
    return set(target)


def get_code_to_name_map():
    """
    対象市場の {4桁銘柄コード: 会社名} の辞書を取得する。
    Discord通知などで、コードだけでなく会社名も表示したい場合に使う。

    get_target_codes() と同じマスターデータを使うため、
    このためだけに追加のAPIリクエストは発生しない
    （_fetch_master_df のキャッシュは行っていないので、両方呼ぶ場合は
    2回リクエストが飛ぶ。両方必要な場合は get_target_codes_and_names() を使うこと）。
    """
    df = _fetch_master_df()

    if "CoName" not in df.columns:
        logger.warning("マスターデータにCoName列が見つかりません。"
                        "会社名なしで処理を続行します（J-Quantsの仕様変更の可能性）。")
        return {}

    # 優先株式などは除外し、4桁表記に変換してからマッピングを作る
    df = df[~df["Code"].str.endswith(config.EXCLUDED_CODE_SUFFIXES)].copy()
    df["Code"] = df["Code"].apply(_normalize_code)

    return dict(zip(df["Code"], df["CoName"]))


def get_target_codes_and_names():
    """
    get_target_codes() と get_code_to_name_map() の両方が必要な場合に、
    マスターデータの取得を1回で済ませるための関数。

    戻り値: (target_codes: set, code_to_name: dict) のタプル
    """
    df = _fetch_master_df()
    has_name_column = "CoName" in df.columns

    if not has_name_column:
        logger.warning("マスターデータにCoName列が見つかりません。"
                        "会社名なしで処理を続行します（J-Quantsの仕様変更の可能性）。")

    filtered = df[~df["Code"].str.endswith(config.EXCLUDED_CODE_SUFFIXES)].copy()
    filtered["Code"] = filtered["Code"].apply(_normalize_code)

    target_codes = set(filtered["Code"].tolist())
    code_to_name = (
        dict(zip(filtered["Code"], filtered["CoName"]))
        if has_name_column else {}
    )

    logger.info(f"{config.TARGET_MARKET}銘柄数: {len(target_codes)}件")
    return target_codes, code_to_name


# =====================================================================
# 日付ベースの取得（1日分・全銘柄。日次スクリーニング向け）
# =====================================================================

def _get_bars_for_date(date_str):
    """指定日（YYYYMMDD形式）の全銘柄分の株価データを取得する（ページング対応）"""
    all_rows = []
    pagination_key = None

    while True:
        params = {"date": date_str}
        if pagination_key:
            params["pagination_key"] = pagination_key

        res = _request_with_retry(
            f"{BASE_URL}/equities/bars/daily",
            params=params
        )
        if res.status_code != 200:
            logger.warning(f"APIエラー ({date_str}): {res.status_code} {res.text}")
        res.raise_for_status()
        body = res.json()

        rows = body.get("data", [])
        all_rows.extend(rows)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break

    return all_rows


def get_price_history():
    """
    【日次スクリーニング用】
    直近の営業日を遡りながら、必要な日数分の全銘柄株価データをまとめて取得する。
    戻り値: pandas.DataFrame（列: Code, Date, O, C, H, L ... など）
    """
    all_rows = []
    collected_days = 0
    day = datetime.now()
    lookback = 0

    while (collected_days < config.TARGET_BUSINESS_DAYS
           and lookback < config.MAX_LOOKBACK_DAYS):

        date_str = day.strftime("%Y%m%d")
        rows = _get_bars_for_date(date_str)

        if rows:
            all_rows.extend(rows)
            collected_days += 1
            logger.info(f" {date_str}: {len(rows)}件取得")
        else:
            logger.info(f" {date_str}: データなし（休日等）")

        day -= timedelta(days=1)
        lookback += 1

    logger.info(f"取得した株価レコード数（合計）: {len(all_rows)}件")
    return _finalize_df(all_rows)


# =====================================================================
# 銘柄ベースの取得（1銘柄・期間指定。バックテスト向け）
# =====================================================================

def _get_bars_for_code(code, from_date=None, to_date=None, _is_fallback_retry=False):
    """
    指定銘柄コードの株価データを、期間指定（from/to）でまとめて取得する。
    from_date, to_date: "YYYYMMDD"形式の文字列。省略時はプランで取得可能な最大範囲になる。

    J-Quantsの仕様: codeを指定した場合、from/toで期間をまとめて指定できる
    （dateのように1日ずつループする必要がない）。

    契約プランがカバーしていない古い日付をfromに指定すると400エラーになる
    （例: 「Your subscription covers the following dates: 2021-07-10 ~」）。
    この場合は致命的なエラーとして処理全体を止めるのではなく、fromを外して
    「プランがカバーする最大範囲」で再取得を試みる
    （＝取れる範囲だけでバックテストを継続できるようにする）。
    """
    all_rows = []
    pagination_key = None

    params_base = {"code": code}
    if from_date:
        params_base["from"] = from_date
    if to_date:
        params_base["to"] = to_date

    while True:
        params = dict(params_base)
        if pagination_key:
            params["pagination_key"] = pagination_key

        res = _request_with_retry(
            f"{BASE_URL}/equities/bars/daily",
            params=params
        )

        if res.status_code == 400 and from_date and not _is_fallback_retry:
            # 契約プランの対象期間外である可能性が高いので、fromを外して
            # 「取得可能な最大範囲」で一度だけ再取得を試みる
            logger.warning(
                f"code={code}: from={from_date}が契約プランの対象期間外の可能性があります。"
                f"fromを外して取得可能な範囲で再取得します。"
            )
            return _get_bars_for_code(code, from_date=None, to_date=to_date, _is_fallback_retry=True)

        if res.status_code != 200:
            logger.warning(f"APIエラー (code={code}): {res.status_code} {res.text}")
        res.raise_for_status()
        body = res.json()

        rows = body.get("data", [])
        all_rows.extend(rows)

        pagination_key = body.get("pagination_key")
        if not pagination_key:
            break

    return all_rows


def get_price_history_range(years=None, cache_filename=None):
    """
    【バックテスト用・初回一括取得】
    過去N年分の全銘柄株価データを、銘柄ごとにfrom/to指定でまとめて取得する。
    件数が多いため、一度取得したらCSVにキャッシュして再利用できるようにする。

    years: 遡る年数（省略時はconfig.BACKTEST_YEARSを使用）
    cache_filename: data/以下に保存するキャッシュファイル名
                    （Noneならキャッシュを使わず毎回取得）

    既にキャッシュが存在する場合は、そのキャッシュをそのまま返す。
    データを最新化したい場合は get_price_history_incremental() を使うこと。
    """
    if years is None:
        years = config.BACKTEST_YEARS

    if cache_filename:
        cache_path = os.path.join(config.DATA_DIR, cache_filename)
        if os.path.exists(cache_path):
            logger.info(f"キャッシュを読み込みます: {cache_path}")
            df = pd.read_csv(cache_path, parse_dates=["Date"])
            return df

    target_codes = get_target_codes()

    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * years + 30)  # 少し余裕を持たせる

    from_str = start_date.strftime("%Y%m%d")
    to_str = end_date.strftime("%Y%m%d")

    all_rows = []
    total = len(target_codes)

    for i, code in enumerate(sorted(target_codes), start=1):
        rows = _get_bars_for_code(code, from_date=from_str, to_date=to_str)
        all_rows.extend(rows)

        if i % 50 == 0 or i == total:
            logger.info(f"進捗: {i}/{total}銘柄 取得済み（累計{len(all_rows)}件）")

    logger.info(f"取得した株価レコード数（合計）: {len(all_rows)}件")
    df = _finalize_df(all_rows)

    if cache_filename:
        _save_cache(df, cache_filename)

    return df


def get_price_history_incremental(cache_filename, years=None):
    """
    【バックテスト用・差分取得】
    既存のCSVキャッシュがあれば、「キャッシュの最新日の翌日」から
    「今日」までの分だけを追加取得してCSVに追記する。
    キャッシュが存在しない場合は、get_price_history_range() と同じ
    初回一括取得を行う。

    これにより、毎回5年分を取り直す必要がなくなり、
    2回目以降の実行は「直近の数日分」だけ取得すればよくなる。

    cache_filename: data/以下のキャッシュファイル名（必須）
    years: キャッシュが存在しない場合の初回取得年数（省略時はconfig.BACKTEST_YEARSを使用）

    戻り値: pandas.DataFrame（更新後の全期間データ）
    """
    if years is None:
        years = config.BACKTEST_YEARS

    cache_path = os.path.join(config.DATA_DIR, cache_filename)

    if not os.path.exists(cache_path):
        logger.info(f"キャッシュが存在しないため、初回一括取得を行います: {cache_path}")
        return get_price_history_range(years=years, cache_filename=cache_filename)

    logger.info(f"既存キャッシュを読み込みます: {cache_path}")
    existing_df = pd.read_csv(cache_path, parse_dates=["Date"])

    if existing_df.empty:
        logger.info("キャッシュが空のため、初回一括取得を行います。")
        return get_price_history_range(years=years, cache_filename=cache_filename)

    latest_cached_date = existing_df["Date"].max()
    from_date = latest_cached_date + timedelta(days=1)
    to_date = datetime.now()

    if from_date.date() > to_date.date():
        logger.info(f"キャッシュは既に最新（{latest_cached_date.date()}）です。"
              f"追加取得は不要です。")
        return existing_df

    from_str = from_date.strftime("%Y%m%d")
    to_str = to_date.strftime("%Y%m%d")

    logger.info(f"差分取得します: {from_str} 〜 {to_str}")

    target_codes = get_target_codes()

    new_rows = []
    total = len(target_codes)

    for i, code in enumerate(sorted(target_codes), start=1):
        rows = _get_bars_for_code(code, from_date=from_str, to_date=to_str)
        new_rows.extend(rows)

        if i % 50 == 0 or i == total:
            logger.info(f"差分取得 進捗: {i}/{total}銘柄（累計{len(new_rows)}件）")

    if not new_rows:
        logger.info("新しいデータはありませんでした（休日のみ等）。")
        return existing_df

    new_df = _finalize_df(new_rows)

    # 既存データと結合し、重複（同一Code・同一Date）があれば新しい方を優先して排除
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["Code", "Date"], keep="last")
    combined_df = combined_df.sort_values(["Code", "Date"]).reset_index(drop=True)

    logger.info(f"差分取得件数: {len(new_df)}件 / "
          f"更新後の合計件数: {len(combined_df)}件")

    _save_cache(combined_df, cache_filename)

    return combined_df


def _save_cache(df, cache_filename):
    os.makedirs(config.DATA_DIR, exist_ok=True)
    cache_path = os.path.join(config.DATA_DIR, cache_filename)
    df.to_csv(cache_path, index=False)
    logger.info(f"キャッシュを保存しました: {cache_path}")


def _finalize_df(all_rows):
    df = pd.DataFrame(all_rows)
    if not df.empty:
        df["Code"] = df["Code"].astype(str).apply(_normalize_code)
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values(["Code", "Date"])
    return df


def _normalize_code(code):
    """
    J-Quantsの銘柄コードは仕様上5桁（普通株は末尾に0を付加した形）で返ってくる。
    例: トヨタ自動車 "72030" -> 一般的な4桁表記 "7203"

    末尾が"0"の5桁コードのみ4桁に変換する。
    優先株式などは末尾が0以外の5桁になることがあり、その場合はそのまま5桁で扱う
    （config.EXCLUDED_CODE_SUFFIXESで別途除外される想定）。
    """
    code = str(code)
    if len(code) == 5 and code.endswith("0"):
        return code[:-1]
    return code
