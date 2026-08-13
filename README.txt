スクリーニング結果ビュー導入ファイル

反映する4ファイル:
1. docs/screening.html  ← 新規追加
2. export_docs_prices.py  ← 置き換え
3. .github/workflows/export_prices.yml  ← 置き換え
4. notifier.py  ← 置き換え

反映後:
GitHub Actions の「株価データ更新（GitHub Pages用）」を手動実行してください。
成功すると docs/screening.json が生成されます。

結果ビュー:
https://screamdayo.github.io/stock-screening/screening.html

A / B / 見送り判定はブラウザのlocalStorageに保存され、GitHubには保存されません。
