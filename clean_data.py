import argparse
from pathlib import Path
import pandas as pd
import numpy as np

EXPECTED_DTYPES = {
    "CustomerID": "string",
    "CustomerName": "string",
    "Gender": "string",
    "Age": "Int64",
    "SignupDate": "datetime64[ns]",
    "City": "string",
    "Email": "string",
    "AnnualIncome": "float64",
    "PurchaseAmount": "float64",
    "SatisfactionScore": "Int64",
}

def parse_numeric(series):
    """Parse numeric strings that may contain currency symbols/commas."""
    cleaned = (series.astype("string")
               .str.replace(r"[^0-9.\\-]", "", regex=True)
               .replace("", pd.NA))
    return pd.to_numeric(cleaned, errors="coerce")

def dtype_accuracy(df):
    matches = 0
    for col, expected in EXPECTED_DTYPES.items():
        if col in df.columns and str(df[col].dtype) == expected:
            matches += 1
    return round(matches / len(EXPECTED_DTYPES) * 100, 1)

def quality_snapshot(df, stage):
    return {
        "Stage": stage,
        "Rows": len(df),
        "Columns": len(df.columns),
        "TotalNullCells": int(df.isna().sum().sum()),
        "DuplicateRows": int(df.duplicated().sum()),
        "DtypeAccuracyPercent": dtype_accuracy(df),
    }

def iqr_cap(series):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    count = int(((series < lower) | (series > upper)).sum())
    return series.clip(lower, upper), lower, upper, count

def clean_dataset(input_path, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path, dtype={"CustomerID": "string"})
    original_columns = list(df.columns)

    # Initial quality report before transformations.
    before = quality_snapshot(df, "Before cleaning")
    null_report = df.isna().sum().rename("NullCount").to_frame()
    null_report.to_csv(output_dir / "nulls_by_column_before.csv")

    # Remove exact duplicate records first.
    df = df.drop_duplicates().copy()

    # Standardise identifiers and text categories.
    df["CustomerID"] = df["CustomerID"].astype("string").str.strip().str.upper()
    df["CustomerName"] = df["CustomerName"].astype("string").str.strip()
    df["Email"] = df["Email"].astype("string").str.strip().str.lower()
    gender_map = {
        "m": "Male", "male": "Male",
        "f": "Female", "female": "Female",
        "other": "Other"
    }
    df["Gender"] = df["Gender"].astype("string").str.strip().str.lower().map(gender_map)
    city_map = {
        "hyderabad": "Hyderabad",
        "bengaluru": "Bengaluru", "bangalore": "Bengaluru",
        "mumbai": "Mumbai", "chennai": "Chennai"
    }
    df["City"] = (df["City"].astype("string").str.strip().str.lower()
                  .map(city_map))

    # Convert mixed-format dates. Invalid date strings become NaT.
    df["SignupDate"] = pd.to_datetime(df["SignupDate"], format="mixed", errors="coerce", dayfirst=True)

    # Parse numeric values even when entered as text/currency.
    for col in ["Age", "AnnualIncome", "PurchaseAmount", "SatisfactionScore"]:
        df[col] = parse_numeric(df[col])

    # Essential fields: remove records without a usable customer ID or signup date.
    df = df.dropna(subset=["CustomerID", "SignupDate"]).copy()

    # Remove impossible values before calculating robust medians.
    df.loc[~df["Age"].between(0, 110), "Age"] = np.nan
    df.loc[df["AnnualIncome"] < 0, "AnnualIncome"] = np.nan
    df.loc[df["PurchaseAmount"] < 0, "PurchaseAmount"] = np.nan
    df.loc[~df["SatisfactionScore"].between(1, 5), "SatisfactionScore"] = np.nan

    # Missing-data strategy: median for numeric; mode for categorical.
    for col in ["Age", "AnnualIncome", "PurchaseAmount", "SatisfactionScore"]:
        df[col] = df[col].fillna(df[col].median())
    for col in ["Gender", "City"]:
        mode = df[col].mode(dropna=True)
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    df["CustomerName"] = df["CustomerName"].fillna("Unknown").replace("", "Unknown")
    df["Email"] = df["Email"].fillna("Not Provided").replace("", "Not Provided")

    # IQR outlier detection and capping. Keep rows to avoid discarding valid customers.
    outlier_rows = []
    for col in ["Age", "AnnualIncome", "PurchaseAmount", "SatisfactionScore"]:
        capped, lower, upper, count = iqr_cap(df[col].astype(float))
        outlier_rows.append({
            "Column": col, "OutlierCount": count,
            "LowerIQRFence": round(float(lower), 3),
            "UpperIQRFence": round(float(upper), 3),
            "Decision": "Cap to IQR fences; retain rows"
        })
        df[col] = capped

    # Enforce final dtypes and domain constraints.
    df["CustomerID"] = df["CustomerID"].astype("string")
    for col in ["CustomerName", "Gender", "City", "Email"]:
        df[col] = df[col].astype("string")
    df["SignupDate"] = pd.to_datetime(df["SignupDate"])
    df["Age"] = df["Age"].round().astype("Int64")
    df["SatisfactionScore"] = df["SatisfactionScore"].round().clip(1, 5).astype("Int64")
    df["AnnualIncome"] = df["AnnualIncome"].astype("float64")
    df["PurchaseAmount"] = df["PurchaseAmount"].astype("float64")

    # Stable ordering improves readability.
    df = df.sort_values("CustomerID").reset_index(drop=True)

    after = quality_snapshot(df, "After cleaning")
    pd.DataFrame([before, after]).to_csv(output_dir / "data_quality_before_after.csv", index=False)
    pd.DataFrame(outlier_rows).to_csv(output_dir / "outlier_report.csv", index=False)
    df.to_csv(output_dir / "cleaned_customer_data.csv", index=False, date_format="%Y-%m-%d")

    print("Before vs after quality report:")
    print(pd.DataFrame([before, after]).to_string(index=False))
    print(f"Removed exact duplicates: {before['DuplicateRows']}")
    print("Saved cleaned data and reports to:", output_dir.resolve())
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean a deliberately messy customer dataset.")
    parser.add_argument("--input", default="data/messy_customer_data.csv")
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    clean_dataset(Path(args.input), Path(args.output))
