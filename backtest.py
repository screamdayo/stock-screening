"""
backtest.py

「戦略が出したシグナル」を受け取ってトレードをシミュレートし、
成績（勝率・平均利益・平均損失・PF）を集計する。

重要: このファイルは特定の戦略（くいっと等）を一切知らない。
      run_backtest.py 側で

          strategy = registry.get_strategy("kuitto")
          signals = strategy(price_df, target_codes)
          trades = run_backtest(signals)

      のように、どの戦略を使うかは呼び出し側が決める。
      新しい戦略を追加してもこのファイルは変更不要。

ルール（config.pyで調整可能）:
- エントリー: シグナル発生日の翌営業日の始値
- 保有期間: 最大 BACKTEST_HOLD_DAYS 営業日
- 利確: 保有中の高値が エントリー価格 * (1 + TAKE_PROFIT_PCT/100) 以上になったら決済
- 損切り: 保有中の安値が エントリー価格 * (1 - STOP_LOSS_PCT/100) 以下になったら決済
- どちらにも該当しないまま保有日数が尽きたら、最終日の終値で強制決済
- 同日に利確・損切り両方の条件を満たした場合は、保守的に損切りを優先する
"""

import pandas as pd

import config
from logger import get_logger

logger = get_logger(__name__)


def _simulate_one_trade(g, signal_idx):
    """
    1つのシグナルに対して、エントリーから決済までをシミュレートする。

    g: 銘柄の時系列DataFrame（Date, O, H, L, C列を持つ）
    signal_idx: シグナルが発生した行のインデックス

    戻り値: dict（トレード結果）または None（エントリーできない場合）
    """
    entry_idx = signal_idx + 1

    if entry_idx >= len(g):
        return None

    entry_row = g.iloc[entry_idx]
    entry_price = entry_row["O"]

    if pd.isna(entry_price) or entry_price <= 0:
        return None

    take_profit_price = entry_price * (1 + config.BACKTEST_TAKE_PROFIT_PCT / 100)
    stop_loss_price = entry_price * (1 - config.BACKTEST_STOP_LOSS_PCT / 100)

    hold_end_idx = min(entry_idx + config.BACKTEST_HOLD_DAYS - 1, len(g) - 1)

    exit_price = None
    exit_reason = None
    exit_date = None
    holding_days = 0

    for idx in range(entry_idx, hold_end_idx + 1):
        row = g.iloc[idx]
        holding_days = idx - entry_idx + 1

        high = row.get("H")
        low = row.get("L")

        hit_take_profit = pd.notna(high) and high >= take_profit_price
        hit_stop_loss = pd.notna(low) and low <= stop_loss_price

        if hit_take_profit and hit_stop_loss:
            exit_price = stop_loss_price
            exit_reason = "stop_loss"
            exit_date = row["Date"]
            break
        elif hit_stop_loss:
            exit_price = stop_loss_price
            exit_reason = "stop_loss"
            exit_date = row["Date"]
            break
        elif hit_take_profit:
            exit_price = take_profit_price
            exit_reason = "take_profit"
            exit_date = row["Date"]
            break

    if exit_price is None:
        final_row = g.iloc[hold_end_idx]
        exit_price = final_row["C"]
        exit_reason = "time_exit"
        exit_date = final_row["Date"]
        holding_days = hold_end_idx - entry_idx + 1

    profit_pct = (exit_price - entry_price) / entry_price * 100

    return {
        "entry_date": entry_row["Date"],
        "entry_price": round(entry_price, 2),
        "exit_date": exit_date,
        "exit_price": round(exit_price, 2),
        "exit_reason": exit_reason,
        "holding_days": holding_days,
        "profit_pct": round(profit_pct, 3),
    }


def run_backtest(signals, price_data_by_code):
    """
    シグナルのリストを受け取り、それぞれについてトレードをシミュレートする。

    signals: 戦略関数（例: strategies/kuitto.py の find_signals）が返す
             シグナルのリスト。各要素は code, signal_date, signal_idx を持つ軽量なdict。
    price_data_by_code: {銘柄コード: 指標計算済みDataFrame} の辞書。
             同じく戦略関数が返す2番目の戻り値。

    戻り値: トレード結果のリスト（dictのリスト）。各dictにはcode, signal_dateも含む。
    """
    trades = []
    skipped_missing_code = 0

    for sig in signals:
        code = sig["code"]
        g = price_data_by_code.get(code)

        if g is None:
            # 通常は起きないはずだが、念のためのガード
            skipped_missing_code += 1
            continue

        result = _simulate_one_trade(g, sig["signal_idx"])
        if result is None:
            continue

        result["code"] = code
        result["signal_date"] = sig["signal_date"]
        trades.append(result)

    if skipped_missing_code > 0:
        logger.warning(f"price_data_by_codeに存在しない銘柄コードのシグナルを "
                        f"{skipped_missing_code}件スキップしました。")

    logger.info(f"シミュレートしたトレード数: {len(trades)}件")
    return trades


def summarize_trades(trades):
    """
    トレード結果のリストから、勝率・平均利益・平均損失・PFなどを集計する。
    戻り値: dict（サマリー統計）
    """
    if not trades:
        return {
            "total_trades": 0,
            "win_rate": None,
            "avg_profit_pct": None,
            "avg_loss_pct": None,
            "profit_factor": None,
            "avg_holding_days": None,
        }

    df = pd.DataFrame(trades)

    wins = df[df["profit_pct"] > 0]
    losses = df[df["profit_pct"] <= 0]

    total_trades = len(df)
    win_rate = len(wins) / total_trades * 100

    avg_profit_pct = wins["profit_pct"].mean() if len(wins) > 0 else 0.0
    avg_loss_pct = losses["profit_pct"].mean() if len(losses) > 0 else 0.0

    gross_profit = wins["profit_pct"].sum() if len(wins) > 0 else 0.0
    gross_loss = -losses["profit_pct"].sum() if len(losses) > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else None

    summary = {
        "total_trades": total_trades,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(win_rate, 2),
        "avg_profit_pct": round(avg_profit_pct, 3),
        "avg_loss_pct": round(avg_loss_pct, 3),
        "profit_factor": round(profit_factor, 3) if profit_factor not in (None, float("inf")) else profit_factor,
        "avg_holding_days": round(df["holding_days"].mean(), 2),
        "take_profit_count": len(df[df["exit_reason"] == "take_profit"]),
        "stop_loss_count": len(df[df["exit_reason"] == "stop_loss"]),
        "time_exit_count": len(df[df["exit_reason"] == "time_exit"]),
    }

    return summary


def print_summary(summary):
    if summary["total_trades"] == 0:
        logger.info("バックテスト結果サマリー: トレードがありませんでした。")
        return

    report_lines = [
        "",
        "=" * 40,
        "バックテスト結果サマリー",
        "=" * 40,
        f"総トレード数       : {summary['total_trades']}件",
        f"勝ちトレード数      : {summary['win_count']}件",
        f"負けトレード数      : {summary['loss_count']}件",
        f"勝率               : {summary['win_rate']}%",
        f"平均利益（勝ち時）   : +{summary['avg_profit_pct']}%",
        f"平均損失（負け時）   : {summary['avg_loss_pct']}%",
        f"プロフィットファクター: {summary['profit_factor']}",
        f"平均保有日数        : {summary['avg_holding_days']}日",
        "-" * 40,
        f"利確決済    : {summary['take_profit_count']}件",
        f"損切り決済  : {summary['stop_loss_count']}件",
        f"期日決済    : {summary['time_exit_count']}件",
        "=" * 40,
    ]
    logger.info("\n".join(report_lines))


def _summarize_group(group_df):
    """1つのグループ（月 or 年）分のトレードから主要指標を計算する共通処理。"""
    wins = group_df[group_df["profit_pct"] > 0]
    losses = group_df[group_df["profit_pct"] <= 0]

    total_trades = len(group_df)
    win_rate = len(wins) / total_trades * 100 if total_trades > 0 else None

    gross_profit = wins["profit_pct"].sum() if len(wins) > 0 else 0.0
    gross_loss = -losses["profit_pct"].sum() if len(losses) > 0 else 0.0

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    else:
        profit_factor = float("inf") if gross_profit > 0 else None

    return pd.Series({
        "total_trades": total_trades,
        "win_count": len(wins),
        "loss_count": len(losses),
        "win_rate": round(win_rate, 2) if win_rate is not None else None,
        "total_profit_pct": round(group_df["profit_pct"].sum(), 3),
        "avg_profit_pct": round(group_df["profit_pct"].mean(), 3),
        "profit_factor": round(profit_factor, 3) if profit_factor not in (None, float("inf")) else profit_factor,
    })


def build_monthly_summary(trades):
    """
    トレード結果を「決済月（exit_date基準）」ごとに集計する。
    戻り値: pandas.DataFrame（月ごとの成績）
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["month"] = df["exit_date"].dt.strftime("%Y-%m")

    monthly = df.groupby("month").apply(_summarize_group).reset_index()
    monthly = monthly.sort_values("month")
    return monthly


def build_yearly_summary(trades):
    """
    トレード結果を「決済年（exit_date基準）」ごとに集計する。
    戻り値: pandas.DataFrame（年ごとの成績）
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df["year"] = df["exit_date"].dt.year

    yearly = df.groupby("year").apply(_summarize_group).reset_index()
    yearly = yearly.sort_values("year")
    return yearly


def build_equity_curve(trades, initial_capital=1_000_000, position_size_pct=100):
    """
    資産推移（エクイティカーブ）を計算する。

    [注意] この関数は「1トレードずつ順番に全資金を投入する」という
    簡略化されたモデルであり、同時に大量のシグナルが発生する戦略
    （例: 東証プライム全銘柄をスクリーニングする戦略）では、
    決済順に何百件ものトレードが束になって複利計算されてしまい、
    資産が非現実的に爆発する（例: 10^100円規模になる）バグを引き起こす。

    同時に複数銘柄を保有する前提のバックテストでは、代わりに
    build_equity_curve_limited() を使うこと。

    各トレードの決済日（exit_date）順に並べ、複利で資産が
    どう推移したかをシミュレートする。

    initial_capital: 初期資金（円）。デフォルト100万円。
    position_size_pct: 1トレードにつき資産の何%を投じるか。
                        デフォルト100%（全力で1銘柄ずつ順番に取引する単純化した想定）。
                        実際は複数銘柄を同時保有できるが、この関数では
                        「決済順に資産が積み上がっていく」という簡略化したモデルを使う。

    戻り値: pandas.DataFrame（列: exit_date, code, profit_pct, capital）
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades)
    df["exit_date"] = pd.to_datetime(df["exit_date"])
    df = df.sort_values("exit_date").reset_index(drop=True)

    capital = initial_capital
    capital_history = []

    for _, row in df.iterrows():
        position_size = capital * (position_size_pct / 100)
        profit_amount = position_size * (row["profit_pct"] / 100)
        capital += profit_amount
        capital_history.append(capital)

    df["capital"] = capital_history
    return df[["exit_date", "code", "profit_pct", "capital"]]


def build_equity_curve_limited(trades, initial_capital=1_000_000, max_positions=5):
    """
    同時保有数を max_positions に制限したエクイティカーブを計算する。

    build_equity_curve() との違い:
        旧関数は「1トレードずつ全資金を投入して複利」という前提のため、
        同時に大量のシグナルが出る戦略では資産が非現実的に爆発してしまう。
        この関数では、常に最大 max_positions 銘柄までしか同時保有できない
        という制約を入れ、資金をポジション数で分割する。上限に達している
        タイミングで来た新規シグナルはエントリーされずスキップされる。

    ルール:
        - 保有中ポジション数が max_positions に達していれば、新規シグナルは見送り
        - 空きがあれば、エントリー日が早い順にエントリー
        - 1ポジションあたりの投入額 = エントリー時点の総資産 / max_positions
        - 同日にエントリーとエグジットが重なる場合は、エグジットを先に処理する
          （decision: 決済で空いた枠に同日の新規シグナルが入れるようにするため）

    trades: run_backtest() が返すトレード結果のリスト。
            各要素の dict に entry_date, exit_date, code, profit_pct が必要。
    initial_capital: 初期資金（円）
    max_positions: 同時に保有できる最大ポジション数

    戻り値: pandas.DataFrame
        列: date, event, code, profit_pct, capital, open_positions
        capital は「その時点で確定している総資産」（保有中ポジションの含み損益は含まない）
        以下の統計は result.attrs に格納される:
            skipped_trades  : 上限超過でエントリーできなかったトレード数
            entered_trades  : 実際にエントリーが成立したトレード数
            total_trades    : 対象トレード総数
            final_capital   : 最終資金
            total_return_pct: トータルリターン(%)
    """
    if not trades:
        return pd.DataFrame()

    df = pd.DataFrame(trades).copy()
    df["entry_date"] = pd.to_datetime(df["entry_date"])
    df["exit_date"] = pd.to_datetime(df["exit_date"])

    # entry_date順に並べる。これが「シグナルが来た順にエントリーを試みる」順序になる。
    df = df.sort_values(["entry_date", "exit_date"]).reset_index(drop=True)

    capital = initial_capital
    open_slots = {}  # {trade_idx: 投入額}
    history_rows = []
    skipped_count = 0

    # 全トレードをエントリーイベントとエグジットイベントに分解し、日付順に処理する
    events = []
    for idx, row in df.iterrows():
        events.append((row["entry_date"], "entry", idx))
        events.append((row["exit_date"], "exit", idx))

    # 同日であれば exit を entry より先に処理する
    # （decision: ポジションが空いてから新規エントリー可能にする）
    events.sort(key=lambda e: (e[0], 0 if e[1] == "exit" else 1))

    entered = set()  # 実際にエントリーが成立したトレードidx

    for date, kind, idx in events:
        row = df.loc[idx]

        if kind == "entry":
            if len(open_slots) >= max_positions:
                # 上限に達しているのでこのシグナルは見送り
                skipped_count += 1
                continue
            position_size = capital / max_positions if max_positions > 0 else 0
            open_slots[idx] = position_size
            entered.add(idx)
            history_rows.append({
                "date": date, "event": "entry", "code": row["code"],
                "profit_pct": None, "capital": capital,
                "open_positions": len(open_slots),
            })

        elif kind == "exit":
            if idx not in open_slots:
                # エントリーが成立していなかった（上限で見送られた）トレードの決済は無視
                continue
            position_size = open_slots.pop(idx)
            profit_amount = position_size * (row["profit_pct"] / 100)
            capital += profit_amount
            history_rows.append({
                "date": date, "event": "exit", "code": row["code"],
                "profit_pct": row["profit_pct"], "capital": capital,
                "open_positions": len(open_slots),
            })

    result = pd.DataFrame(history_rows)
    result.attrs["skipped_trades"] = skipped_count
    result.attrs["entered_trades"] = len(entered)
    result.attrs["total_trades"] = len(df)
    result.attrs["final_capital"] = capital
    result.attrs["total_return_pct"] = (capital - initial_capital) / initial_capital * 100

    return result


def print_equity_summary(result, initial_capital=1_000_000):
    """build_equity_curve_limited() の結果を人間向けに整形して表示する。"""
    if result.empty:
        logger.info("トレードがありませんでした。")
        return

    final_capital = result.attrs["final_capital"]
    total_return_pct = result.attrs["total_return_pct"]

    report_lines = [
        "",
        "=" * 40,
        "エクイティカーブ サマリー（同時保有数制限あり）",
        "=" * 40,
        f"対象トレード数     : {result.attrs['total_trades']}件",
        f"実際にエントリー   : {result.attrs['entered_trades']}件",
        f"上限超過でスキップ : {result.attrs['skipped_trades']}件",
        f"初期資金           : {initial_capital:,.0f}円",
        f"最終資金           : {final_capital:,.0f}円",
        f"トータルリターン   : {total_return_pct:,.2f}%",
        "=" * 40,
    ]
    logger.info("\n".join(report_lines))
