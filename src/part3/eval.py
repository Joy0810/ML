import matplotlib.pyplot as plt
import numpy as np

# Model names
models = [' Random Forest', 'Lasso', 'Ridge', 'SVR rbf', 'SVR Linear', 'SVR Polynomial']

# Metrics (using fixed growth rate as actual)
r2_scores = [0.9978, 0.9975, 0.9976, 0.9953, 0.8812, 0.9953]
rmse = [257.00, 263.41, 258.01, 341.08, 1707.39, 341.08]
mape = [2.20, 2.26, 2.31, 3.08, 18.95, 3.08]

# Find best model indices
best_r2_index = np.argmax(r2_scores)
best_rmse_index = np.argmin(rmse)
best_mape_index = np.argmin(mape)

# Save best model info
best_r2_model = models[best_r2_index]
best_r2_value = r2_scores[best_r2_index]

best_rmse_model = models[best_rmse_index]
best_rmse_value = rmse[best_rmse_index]

best_mape_model = models[best_mape_index]
best_mape_value = mape[best_mape_index]

# Plotting function (with save feature)
def plot_metric(values, title, ylabel, best_index, filename, color='skyblue', highlight_color='green'):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(models))
    bars = ax.bar(x, values, width=0.6, color=color)
    bars[best_index].set_color(highlight_color)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Annotate values
    for rect in bars:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10)

    # Highlight best
    ax.annotate('Best Model',
                xy=(x[best_index], values[best_index]),
                xytext=(0, 10), textcoords='offset points',
                ha='center', va='bottom', fontsize=12, color='black',
                arrowprops=dict(arrowstyle='->', color='black'))

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Saved graph: {filename}")

# Generate and save plots
plot_metric(r2_scores, 'Fixed Growth: R² Score Comparison', 'R² Score', best_r2_index, 'fixed_r2_score_comparison.png', color='lightblue', highlight_color='darkblue')
plot_metric(rmse, 'Fixed Growth: RMSE Comparison', 'RMSE', best_rmse_index, 'fixed_rmse_comparison.png', color='lightcoral', highlight_color='darkred')
plot_metric(mape, 'Fixed Growth: MAPE Comparison', 'MAPE (%)', best_mape_index, 'fixed_mape_comparison.png', color='lightgreen', highlight_color='darkgreen')

# Print best scores
print("\nBest Scores Summary (Fixed Growth Rate):")
print(f"✅ Best R² Score Model     : {best_r2_model} ({best_r2_value:.4f})")
print(f"✅ Best RMSE Model         : {best_rmse_model} ({best_rmse_value:.2f})")
print(f"✅ Best MAPE Model         : {best_mape_model} ({best_mape_value:.2f}%)")