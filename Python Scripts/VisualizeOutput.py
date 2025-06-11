import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc

# Step 1: Load WEKA CSV output
df = pd.read_csv("Model Output/M6_output.csv")  # make sure to save your output as CSV

# Step 2: Extract true and predicted labels
# Format in WEKA is "1:NEG", "2:POS", so we take what's after ":"
df['actual_label'] = df['actual'].str.split(':').str[1]
df['predicted_label'] = df['predicted'].str.split(':').str[1]
df['confidence'] = df['prediction']  # this is the model's confidence in predicted class

# Optional: binarize for PR curve
df['actual_binary'] = df['actual_label'].apply(lambda x: 1 if x == 'POS' else 0)
df['predicted_binary'] = df['predicted_label'].apply(lambda x: 1 if x == 'POS' else 0)

# Step 3: Confusion Matrix
cm = confusion_matrix(df['actual_label'], df['predicted_label'], labels=['NEG', 'POS'])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['NEG', 'POS'], yticklabels=['NEG', 'POS'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# Step 4: Precision-Recall Curve (using probability from prediction column)
precision, recall, _ = precision_recall_curve(df['actual_binary'], df['confidence'])
pr_auc = auc(recall, precision)

plt.figure(figsize=(6, 5))
plt.plot(recall, precision, label=f'PR Curve (AUC={pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
