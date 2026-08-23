import polars as pl
df = pl.DataFrame({"impressions": ["1-0 2-1", "3-1"]})

def parse_mind_impressions(imp_str):
    if not imp_str: return {"clicked": [], "unclicked": []}
    c, u = [], []
    for item in str(imp_str).split():
        if '-' in item:
            aid, click = item.split('-')
            if click == '1': c.append(aid)
            else: u.append(aid)
    return {"clicked": c, "unclicked": u}

parsed = df.select(pl.col("impressions").map_elements(parse_mind_impressions, return_dtype=pl.Struct([pl.Field("clicked", pl.List(pl.Utf8)), pl.Field("unclicked", pl.List(pl.Utf8))])).alias("parsed"))

df2 = df.with_columns([
    parsed["parsed"].struct.field("clicked").alias("clicked_articles"),
    parsed["parsed"].struct.field("unclicked").alias("unclicked_articles")
])
print(df2)
