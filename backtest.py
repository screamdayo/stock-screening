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
        print(f"[backtest] 警告: price_data_by_codeに存在しない銘柄コードのシグナルを "
              f"{skipped_missing_code}件スキップしました。")

    print(f"[backtest] シミュレートしたトレード数: {len(trades)}件")
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
    print("=" * 40)
    print("バックテスト結果サマリー")
    print("=" * 40)
    print(f"総トレード数       : {summary['total_trades']}件")

    if summary["total_trades"] == 0:
        print("トレードがありませんでした。")
        return

    print(f"勝ちトレード数      : {summary['win_count']}件")
    print(f"負けトレード数      : {summary['loss_count']}件")
    print(f"勝率               : {summary['win_rate']}%")
    print(f"平均利益（勝ち時）   : +{summary['avg_profit_pct']}%")
    print(f"平均損失（負け時）   : {summary['avg_loss_pct']}%")
    print(f"プロフィットファクター: {summary['profit_factor']}")
    print(f"平均保有日数        : {summary['avg_holding_days']}日")
    print("-" * 40)
    print(f"利確決済    : {summary['take_profit_count']}件")
    print(f"損切り決済  : {summary['stop_loss_count']}件")
    print(f"期日決済    : {summary['time_exit_count']}件")
    print("=" * 40)


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
