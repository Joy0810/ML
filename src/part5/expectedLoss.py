#!/usr/bin/env python3
import pandas as pd
from pathlib import Path
import sys

def find_salary_column(df: pd.DataFrame, keyword: str):
    """
    Returns the first column name in df that contains the given keyword.
    Raises a KeyError if none found.
    """
    matches = [c for c in df.columns if keyword.lower() in c.lower()]
    if not matches:
        raise KeyError(f"Could not find any column containing '{keyword}' in {df.columns.tolist()}")
    return matches[0]

def compute_expected_loss(df_prob: pd.DataFrame, df_pred: pd.DataFrame, salary_col: str):
    # align keys
    df_pred['EmployeeIndex'] = df_prob['EmployeeIndex']
    df = pd.merge(df_prob, df_pred, on='EmployeeIndex')
    df['P_leave'] = 1.0 - df['P_stay']
    df['ExpectedLoss'] = df['P_leave'] * df[salary_col]
    total = df['ExpectedLoss'].sum()
    return df[['EmployeeIndex', 'P_stay', 'P_leave', salary_col, 'ExpectedLoss']], total

def main():
    base = Path(__file__).parent

    # 1) Load stay‐probabilities
    prob_csv = base.parent / 'part4' / 'likely_to_stay_salaries.csv'
    df_prob = pd.read_csv(prob_csv)

    # 2) Load both prediction DataFrames
    fixed_csv = base.parent / 'part3' / 'RandomForest' / 'Predicted_FutureSalary_FromIncrement.csv'
    perf_csv  = base.parent / 'part3' / 'SVR' / 'Linear' / 'performancebased.csv'
    df_fixed  = pd.read_csv(fixed_csv)
    df_perf   = pd.read_csv(perf_csv)

    # 3) Detect the actual salary‐column names
    try:
        fixed_col = find_salary_column(df_fixed, 'increment')  # still searching for 'increment' column name
        perf_col  = find_salary_column(df_perf, 'perf')
    except KeyError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # 4) Compute expected losses
    df_fixed_loss, total_fixed = compute_expected_loss(df_prob, df_fixed, fixed_col)
    df_perf_loss, total_perf = compute_expected_loss(df_prob, df_perf, perf_col)

    # 5) Output and write CSVs
    print("=== Fixed-Based Predictions ===")
    print(f"Detected salary column: {fixed_col}")
    print(f"Total expected salary loss: ₹{total_fixed:,.2f}\n")

    print("=== Performance-Based Predictions ===")
    print(f"Detected salary column: {perf_col}")
    print(f"Total expected salary loss: ₹{total_perf:,.2f}\n")

    out_fixed = base / 'expected_loss_fixed.csv'
    out_perf  = base / 'expected_loss_performance.csv'
    df_fixed_loss.to_csv(out_fixed, index=False)
    df_perf_loss.to_csv(out_perf, index=False)

    print(f"Wrote per-employee fixed-based losses to: {out_fixed}")
    print(f"Wrote per-employee performance-based losses to: {out_perf}")

if __name__ == '__main__':
    main()
