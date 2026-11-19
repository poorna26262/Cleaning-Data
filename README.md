# Data Cleaning Project — Messy Customer Dataset

A beginner-friendly but professional-style data cleaning workflow using Python, pandas and NumPy.

## Project structure
- `data/messy_customer_data.csv` — deliberately messy **synthetic** input data.
- `notebooks/data_cleaning.ipynb` — guided notebook with markdown explanations and executable steps.
- `src/clean_data.py` — reusable end-to-end cleaning script.
- `outputs/` — generated cleaned CSV and quality reports after running the script.

> The sample data is synthetic for practice. It is not real customer information.

## Setup
Python 3.10+ recommended.

```bash
python -m venv .venv
```

Windows PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Run the notebook
From the project root:
```bash
jupyter notebook
```
Open `notebooks/data_cleaning.ipynb` and run all cells in order.

## Run the script
From the project root:
```bash
python src/clean_data.py --input data/messy_customer_data.csv --output outputs
```

## Cleaning decisions documented in the project
- Remove exact duplicate rows.
- Trim and standardise CustomerID, gender and city labels.
- Parse mixed date strings to pandas datetime; invalid dates are treated as missing and removed because signup date is essential for time-based analysis.
- Keep CustomerID as string to preserve identifier semantics.
- Convert age, income, purchase amount and satisfaction to numeric values; remove currency symbols and thousands separators.
- Impute Age, AnnualIncome, PurchaseAmount and SatisfactionScore with their medians (robust to skew/outliers).
- Impute Gender and City with their modes (categorical fields).
- Fill missing CustomerName and Email with explicit placeholders rather than inventing personal data.
- Drop rows missing CustomerID or a valid SignupDate because those fields are essential identifiers/time fields.
- Detect numeric outliers with the 1.5 × IQR rule and cap them at IQR fences rather than deleting potentially valid high-value customers.
- Enforce final dtypes and validate ranges.

## Outputs
- `cleaned_customer_data.csv`
- `data_quality_before_after.csv`
- `nulls_by_column_before.csv`
- `outlier_report.csv`

The before/after report compares row count, total missing cells, duplicate rows, and schema dtype accuracy.
