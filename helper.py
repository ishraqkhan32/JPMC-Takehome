import pandas as pd

def column_summary(df):
    rows = []

    for col in df.columns:
        s = df[col]
        s_str = s.astype(str)

        # Counts
        nan_count = s.isna().sum()
        qmark_count = (s_str == "?").sum()
        not_in_universe_count = s_str.str.contains(
            "not in universe", case=False, na=False
        ).sum()

        # Build compact sentinel summary
        sentinel_lines = []
        if qmark_count > 0:
            sentinel_lines.append(f"?: {qmark_count}")
        if nan_count > 0:
            sentinel_lines.append(f"NaN: {nan_count}")
        if not_in_universe_count > 0:
            sentinel_lines.append(f"not in universe: {not_in_universe_count}")

        sentinel_summary = "\n".join(sentinel_lines) if sentinel_lines else ""

        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "n_unique": s.nunique(dropna=True),
            "special_values": sentinel_summary,
            "unique_values": sorted(s.dropna().unique().tolist())
        })

    return pd.DataFrame(rows)