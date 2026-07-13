# stock-screening

東証プライム市場を対象に、株価パターンを検出して自動でDiscord通知する。
過去データを使ったバックテスト（勝率・PFなどの検証）機能も含む。
複数の戦略（スクリーニング条件）を切り替えて使える設計になっている。

## フォルダ構成

```
stock-screening/
├── .github/
│   └── workflows/
│       ├── stock_screening.yml   # 日次自動実行の設定（GitHub Actions）
│       └── export_prices.yml     # GitHub Pages用の株価データ自動更新
│
├── docs/                # GitHub Pages公開ルート（保有ポジション利確チャート）
│   ├── index.html       # SBI証券CSVをブラウザ内だけで解析・チャート表示する静的サイト
│   └── prices/           # 銘柄コード→株価のJSON（GitHub Actionsが自動生成、公開情報のみ）
│
├── data/                # ダウンロードした株価データのCSVキャッシュ（.gitignore対象）
├── output/              # スクリーニング・バックテストの結果（.gitignore対象）
├── logs/                # 実行ログ（.gitignore対象）
│
├── strategies/          # 戦略（シグナル検出ロジック）を集めたパッケージ
│   ├── base.py          # 戦略が守るべき共通インターフェースの定義
│   ├── registry.py       # 戦略名→関数の対応表。新戦略はここに登録する
│   └── kuitto.py        # 「くいっと」戦略（5日MA上向き＆陽線）の実装
│
├── config.py            # 条件・APIキーなど、調整はここだけでOK
├── download.py          # J-Quants APIからデータ取得
├── indicator.py         # 移動平均・RSIなどの指標計算
├── backtest.py          # バックテストのシミュレーションと集計（戦略の中身を知らない）
├── notifier.py          # Discord通知
├── main.py              # 日次スクリーニングの実行エントリーポイント
├── run_backtest.py       # バックテストの実行エントリーポイント
├── export_docs_prices.py # docs/prices/への株価データエクスポート（GitHub Pages用）
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 設計方針: 戦略プラグイン方式

`backtest.py` と `main.py` は「どの戦略を使うか」を知らない。
戦略は `strategies/` 以下に独立したファイルとして実装し、
`strategies/registry.py` に登録することで、コードの他の部分を
一切変更せずに追加・切り替えができる。

### 新しい戦略を追加する手順

1. `strategies/kuitto.py` をコピーして `strategies/新戦略名.py` を作る
2. `find_signals(price_df, target_codes)` の中身を、新しい条件判定ロジックに書き換える
   （戻り値の形式は `strategies/base.py` のコメントを参照）
3. 日次スクリーニングにも対応させたい場合は `find_latest_signals(price_df, target_codes)` も実装する
4. `strategies/registry.py` の `STRATEGIES` と `LATEST_SCREENERS` に1行ずつ追記する
5. `config.py` の `ACTIVE_STRATEGY` を新しい戦略名に変更するか、
   バックテストのみなら `python run_backtest.py --strategy 新戦略名` で実行する

## セットアップ

1. J-Quants APIキーとDiscord Webhook URLを取得する
2. GitHubリポジトリの Settings → Secrets and variables → Actions に以下を登録する
   - `JQUANTS_API_KEY`
   - `DISCORD_WEBHOOK_URL`
3. ローカルで試す場合は環境変数として同様に設定してから実行する

```bash
pip install -r requirements.txt
export JQUANTS_API_KEY="xxxx"
export DISCORD_WEBHOOK_URL="xxxx"
```

## 日次スクリーニングの実行

```bash
python main.py
```

`config.py` の `ACTIVE_STRATEGY` で指定した戦略が使われる。
GitHub Actionsでは平日16:00（JST）に自動実行される。

## バックテストの実行

```bash
python run_backtest.py                      # config.pyのACTIVE_STRATEGYを使用
python run_backtest.py --strategy kuitto   # 戦略を明示的に指定
```

過去5年分のデータで、シグナル発生の翌営業日の始値でエントリーし、
最大10営業日保有した場合の成績を検証する（ルールは`config.py`で調整可能）。

### バックテストのルール

- エントリー: シグナル発生日の翌営業日の**始値**
- 保有期間: 最大 `BACKTEST_HOLD_DAYS` 営業日（デフォルト: 10日）
- 利確: 保有中の高値が `+BACKTEST_TAKE_PROFIT_PCT` %（デフォルト: +5%）に到達したら決済
- 損切り: 保有中の安値が `-BACKTEST_STOP_LOSS_PCT` %（デフォルト: -3%）に到達したら決済
- どちらにも到達しなければ、保有期間終了時の終値で強制決済
- 同日に利確・損切り両方の条件を満たした場合は、保守的に損切りを優先する

### 出力ファイル（`output/`フォルダ）

- `trades.csv` : 全トレードの明細（最新実行分、常に上書き）
- `losing_trades.csv` : 負けトレードだけを抜き出したもの（敗因分析用）
- `trades_<戦略名>_<日時>.csv` : 実行ごとの履歴（比較用に残る）
- `summary_<戦略名>_<日時>.json` : 勝率・PFなどのサマリー統計

`trades.csv` の主な列:

| 列名 | 内容 |
|---|---|
| code | 銘柄コード |
| signal_date | シグナルが発生した日 |
| entry_date | エントリー日（シグナル発生日の翌営業日） |
| entry_price | エントリー価格（始値） |
| exit_date | 決済日 |
| exit_price | 決済価格 |
| exit_reason | 決済理由（take_profit / stop_loss / time_exit） |
| holding_days | 実際の保有日数 |
| profit_pct | 損益（%） |
| result | win / loss |

`exit_reason`ごとに集計すれば、「利確できずに期日決済で負けたケースが多い」
「損切りにすぐ引っかかるケースが多い」など、負けパターンの傾向を分析できる。

### データキャッシュと差分取得について

バックテストは5年分の大量データを扱うため、`data/` フォルダにCSVとしてキャッシュされる。

**初回実行時**：全銘柄・全期間を取得してキャッシュを新規作成する。
**2回目以降**：キャッシュの最新日の翌日から今日までの「差分」だけを取得してキャッシュに追記する
（`get_price_history_incremental()`）。そのため、条件（`config.py`）だけを変えて
何度も再検証したい場合や、日を跨いで定期的に検証したい場合でも、
毎回5年分を丸ごと取り直す必要がなく、実行時間を大幅に短縮できる。

データ取得は銘柄コードごとに期間（from/to）を指定してまとめて取る方式を採用しており、
1日ずつ全銘柄を取得する方式と比べてページング回数が少なく済み、より高速に動作する
（`download.py` の `_get_bars_for_code` を参照）。

データを完全に作り直したい場合（欠損の疑いがある場合など）は、
`data/` 内の該当CSVを削除してから再実行すると、初回一括取得からやり直される。

## 保有ポジションの利確チャート表示（GitHub Pages）

現在保有中のポジションを、エントリー価格・利確ライン・損切りラインを重ねた
インタラクティブなローソク足チャートで確認できる。**保有ポジションの情報（SBI証券の
約定履歴CSV）は一切リポジトリに保存されず、ブラウザ内の処理だけで完結する**設計。

### 仕組み

```
GitHub Actions（サーバー側）
  └ export_docs_prices.py が毎日J-Quantsから株価を取得し、
    docs/prices/<銘柄コード>.json として自動commit
    （銘柄コード→株価という公開情報のみ。個人の保有情報は含まない）

GitHub Pages（docs/index.html、静的サイト）
  └ SBI証券からダウンロードした約定履歴CSVを、ファイル選択ボタンで読み込む
  └ FileReaderでブラウザ内だけ解析（Shift_JISデコード→CSVパース→
    保有ポジション計算）。サーバーへの送信は一切なし
  └ 算出した保有銘柄コードをキーに docs/prices/<コード>.json をfetch
  └ Plotly（CDN）でローソク足＋エントリー/利確/損切りラインを描画
```

### 使い方

1. **GitHub Pagesを有効化**: リポジトリの Settings → Pages → Source を
   「Deploy from a branch」、フォルダを `/docs` に設定する
2. スマホのブラウザで `https://<ユーザー名>.github.io/<リポジトリ名>/` を開く
3. SBI証券に **PCブラウザ or スマホブラウザ** でログインし、
   「口座管理」→「取引履歴」から**約定履歴CSV**をダウンロードする
   （投資信託の取引も含まれるが自動的に無視される。信用取引にも対応）
4. 上記ページの「CSVファイルを選択」からダウンロードしたCSVを選ぶ
   → その場で保有ポジション一覧とチャートが表示される
5. 利確/損切りラインの%は、ページ内で銘柄ごとに調整可能
   （初期値は共通のデフォルト%。設定はその端末のlocalStorageにのみ保存される）

### ポジション計算のロジック

約定履歴（買い・売りの記録）から、移動平均法で現在の保有数量・平均取得単価を算出する。

- 現物取引（株式現物買/売）、信用取引（信用新規買/返済売＝買い建て、
  信用新規売/返済買＝売り建て）をそれぞれ別のポジションとして扱う
- 保有数量が一度0になった時点で「連続保有期間」がリセットされ、
  そこから先の買いだけで平均取得単価とエントリー日を再計算する
- 投資信託（銘柄コードを持たない取引）は自動的に除外される

### 株価データの更新

`.github/workflows/export_prices.yml` が平日16:30（JST）ごろに自動実行され、
対象市場全銘柄の直近`config.CHART_BUSINESS_DAYS`営業日分の株価を
`docs/prices/`以下にJSONとして出力・commitする。ローカルで手動実行する場合:

```bash
python export_docs_prices.py
```

## 免責事項

本ツールの出力は投資判断の参考情報であり、投資助言や利益を保証するものではない。
実際の売買は自己責任で行うこと。
