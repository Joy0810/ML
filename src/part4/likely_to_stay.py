import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve,
    precision_score, recall_score, f1_score, accuracy_score
)

# 1. Locate the Part 1 output file (Attrition probabilities)
base_dir   = Path(__file__).parent            # ML/src/part4
part1_file = base_dir.parent / 'part1' / 'part1lg_output.txt'

# 2. Read in the attrition probabilities
df = pd.read_csv(
    part1_file,
    sep='\t',
    index_col='EmployeeIndex'
)

# 3. Compute “stay” probability
df['P_stay'] = 1.0 - df['Attrition_Probability']

# 4. Prepare true labels and scores for threshold selection
labels_csv = base_dir.parent.parent / 'dataset' / 'WA_Fn-UseC_-HR-Employee-Attrition.csv'
labels_df  = pd.read_csv(labels_csv, index_col='EmployeeNumber')

# y_true: 1 if Attrition == 'No', else 0
# Use positional indexing because df.index is 0..N-1
y_true   = (labels_df['Attrition'] == 'No').iloc[df.index].astype(int).values
y_scores = df['P_stay'].values

# 5. **Threshold sweep**: compute metrics at each candidate
thresholds = np.linspace(0, 1, 101)
metrics = []
for t in thresholds:
    y_pred = (y_scores >= t).astype(int)
    metrics.append({
        'threshold':  t,
        'accuracy':   accuracy_score(y_true, y_pred),
        'precision':  precision_score(y_true, y_pred, zero_division=0),
        'recall':     recall_score(y_true, y_pred, zero_division=0),
        'f1':         f1_score(y_true, y_pred, zero_division=0)
    })

metrics_df = pd.DataFrame(metrics)
# Print top 5 thresholds by F1
print("\nTop 5 thresholds by F1 score:")
print(metrics_df.sort_values('f1', ascending=False).head(5))

# Choose the best F1 threshold
best_f1_row      = metrics_df.loc[metrics_df['f1'].idxmax()]
best_threshold   = best_f1_row['threshold']
best_f1, best_prec, best_rec = best_f1_row['f1'], best_f1_row['precision'], best_f1_row['recall']
print(f"\nBest F1 threshold: {best_threshold:.2f} (F1={best_f1:.3f}, Precision={best_prec:.3f}, Recall={best_rec:.3f})")

# 6. Now flag “likely to stay” using this best threshold
df['likely_to_stay'] = df['P_stay'] >= best_threshold

# --- Pie Chart: Stay vs. Leave ---
counts = df['likely_to_stay'].value_counts()
plt.figure()
plt.pie(
    counts.values,
    labels=['Stay', 'Leave'],
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'edgecolor': 'white'}
)
plt.title(f'Pie Chart of Stay vs. Leave (Threshold = {best_threshold:.2f})')
plt.axis('equal')
plt.show()
