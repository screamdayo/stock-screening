import requests
import pandas as pd
import os

JQUANTS_API_KEY = os.environ["JQUANTS_API_KEY"]

res = requests.get(
    "https://api.jquants.com/v2/equities/master",
    headers={"x-api-key": JQUANTS_API_KEY}
)
data = res.json()
df = pd.DataFrame(data["data"])
prime = df[df["MktNm"] == "プライム"]

# コード 94345, 94346 を探す
print(prime[prime["Code"].isin(["94345", "94346"])])
print("\n==== プライム銘柄サンプル ====")
print(prime.head(10))
