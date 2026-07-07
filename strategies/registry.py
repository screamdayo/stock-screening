"""
strategies/registry.py

利用可能な戦略を一覧管理する。
新しい戦略を追加したら、ここに1行追加するだけで
run_backtest.py や main.py から名前で呼び出せるようになる。
"""

from strategies import kuitto

# 戦略名 -> シグナル検出関数（バックテスト用: find_signals）
STRATEGIES = {
    "kuitto": kuitto.find_signals,
    # 新しい戦略を追加する場合はここに追記する。例:
    # "golden_cross": golden_cross.find_signals,
}

# 戦略名 -> 日次スクリーニング用関数（最新日だけ判定するもの）
LATEST_SCREENERS = {
    "kuitto": kuitto.find_latest_signals,
}


def get_strategy(name):
    """バックテスト用の戦略関数を名前から取得する。"""
    if name not in STRATEGIES:
        available = ", ".join(STRATEGIES.keys())
        raise ValueError(f"未登録の戦略です: '{name}'（利用可能: {available}）")
    return STRATEGIES[name]


def get_latest_screener(name):
    """日次スクリーニング用の関数を名前から取得する。"""
    if name not in LATEST_SCREENERS:
        available = ", ".join(LATEST_SCREENERS.keys())
        raise ValueError(f"未登録の戦略です: '{name}'（利用可能: {available}）")
    return LATEST_SCREENERS[name]
