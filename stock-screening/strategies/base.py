"""
strategies/base.py

すべての戦略が守るべき「型」を定義する。
新しい戦略を追加するときは、この形式に合わせて関数を書けば
backtest.py 側は一切変更せずに使い回せる。

戦略関数のシグネチャ:

    def find_signals(price_df, target_codes) -> tuple[list[dict], dict]

    price_df: download.get_price_history_range() などで取得した
              全銘柄・全期間の株価DataFrame（列: Code, Date, O, H, L, C ...）
    target_codes: 対象とする銘柄コードのset

    戻り値: (signals, price_data_by_code) のタプル

        signals: シグナルのリスト。各要素は以下の3キーのみを持つ軽量な dict。
            code: 銘柄コード
            signal_date: シグナル発生日（Timestamp）
            signal_idx: その銘柄の時系列DataFrame内でのインデックス（0始まり）

        price_data_by_code: {銘柄コード: 指標計算済みDataFrame} の辞書。
            1銘柄につきDataFrameを1つだけ保持する
            （シグナルの数だけDataFrameを複製しないための設計）。

    backtest.py はシグナルの code をキーに price_data_by_code を参照して
    該当銘柄のDataFrameを取得し、シミュレーションを行う。
    「シグナルをどう判定するか」の中身は各戦略ファイルが自由に決めてよい。
"""

from typing import Protocol, List, Dict, Tuple
import pandas as pd


class Strategy(Protocol):
    """型ヒント用のプロトコル。厳密な継承は不要で、
    同じ形の関数を書けば自動的にこのインターフェースに適合する。"""

    def __call__(
        self, price_df: pd.DataFrame, target_codes: set
    ) -> Tuple[List[Dict], Dict[str, pd.DataFrame]]:
        ...
