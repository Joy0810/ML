import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def histogram(losses: pd.Series, model_name: str, bins: int = 30):
    """
    Plots a histogram of the losses.
    """
    plt.figure()
    plt.hist(losses, bins=bins, alpha=0.75)
    plt.xlabel('Expected Salary Loss')
    plt.ylabel('Number of Employees')
    plt.title(f'{model_name}: Distribution of Expected Salary Loss')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # assume script lives alongside the two CSVs
    p = Path(__file__).parent

    # Load
    df_fixed  = pd.read_csv(p / 'expected_loss_fixed.csv')  # now considered "Fixed-Based"
    df_perf = pd.read_csv(p / 'expected_loss_performance.csv')

    fixed_losses  = df_fixed['ExpectedLoss']
    perf_losses   = df_perf['ExpectedLoss']

    # Fixed-Based: histogram only
    histogram(fixed_losses, 'Fixed-Based Model')

    # Performance-Based: histogram only
    histogram(perf_losses, 'Performance-Based Model')

if __name__ == '__main__':
    main()
