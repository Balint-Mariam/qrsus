import pandas as pd

in_path = r"C:\Users\user\Desktop\simpozion\date simp.xlsx"
out_path = r"C:\Users\user\Desktop\simpozion\date_simp_concat.xlsx"

# Read the Excel file (first sheet by default)
df = pd.read_excel(in_path)

cols = list(df.columns)
if len(cols) % 2 != 0:
    raise ValueError("Number of columns must be even: pairs of Date + Variable.")

pairs = [(cols[i], cols[i + 1]) for i in range(0, len(cols), 2)]

series_list = []
for date_col, val_col in pairs:
    tmp = df[[date_col, val_col]].copy()

    # Try parsing as calendar dates (dd/mm/yyyy). If that fails, treat as Excel serial.
    parsed = pd.to_datetime(tmp[date_col], errors="coerce", dayfirst=True)
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(tmp[date_col], errors="coerce", unit="D", origin="1899-12-30")

    tmp[date_col] = parsed
    tmp = tmp.rename(columns={date_col: "Date", val_col: val_col})
    # Remove rows without valid dates and ensure unique Date index
    tmp = tmp.dropna(subset=["Date"]).groupby("Date", as_index=True).agg("first")
    series_list.append(tmp)

# Align on same dates and drop any date with missing values
result = pd.concat(series_list, axis=1, join="inner").dropna()

result.to_excel(out_path, index=True)
print("Saved:", out_path)
