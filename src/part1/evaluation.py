import matplotlib.pyplot as plt
import numpy as np

# Example of metrics for each model (replace with actual results from your scripts)
models = ['Decision Tree', 'SVM', 'Random Forest', 'Random Forest + SMOTE', 'XGBoost']

# Accuracy, Precision, Recall, F1-Score, and ROC AUC
accuracy = [0.78, 0.47, 0.84, 0.81, 0.84]
precision = [0.88, 0.90, 0.85, 0.86, 0.89]
recall = [0.85, 0.42, 0.99, 0.92, 0.92]
f1_score = [0.87, 0.57, 0.91, 0.89, 0.91]
roc_auc = [0.63, 0.64, 0.76, 0.74, 0.74]


# Plotting the comparison
x = np.arange(len(models))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(12, 8))

# Plot each metric with distinct colors
rects1 = ax.bar(x - 2*width, accuracy, width, label='Accuracy', color='lightblue')
rects2 = ax.bar(x - width, precision, width, label='Precision', color='lightgreen')
rects3 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
rects4 = ax.bar(x + width, f1_score, width, label='F1-Score', color='lightskyblue')
rects5 = ax.bar(x + 2*width, roc_auc, width, label='ROC AUC', color='orange')

# Highlight the best model
best_model_index = 4 

# Change the color of the best model's bars
rects1[best_model_index].set_color('darkblue')
rects2[best_model_index].set_color('darkgreen')
rects3[best_model_index].set_color('darkred')
rects4[best_model_index].set_color('darkcyan')
rects5[best_model_index].set_color('darkorange')

# Add some text for labels, title, and custom x-axis tick labels
ax.set_xlabel('Models', fontsize=14)
ax.set_ylabel('Scores', fontsize=14)
ax.set_title('Model Comparison (Accuracy, Precision, Recall, F1-Score, ROC AUC)', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(fontsize=12)

# Add gridlines for better readability
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Autolabel function to display the value on top of the bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%.2f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

# Call the autolabel function
autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)

# Add annotation to highlight the best model
ax.annotate('Best Model', 
            xy=(x[best_model_index], max(f1_score[best_model_index], precision[best_model_index], recall[best_model_index], accuracy[best_model_index])), 
            xytext=(0, 10), 
            textcoords='offset points', 
            ha='center', fontsize=12, color='black', weight='bold', arrowprops=dict(arrowstyle='->', color='black'))

# Show the plot
plt.tight_layout()
plt.show()
