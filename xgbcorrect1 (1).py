import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC



# Load the CSV file
file = '/home/group_shyam01/countperchr_unfiltered_all.csv'
df = pd.read_csv(file)

# Remove unnecessary columns
del df['Sample']
#df = df.fillna(0)
#del df['missense_chrY']
print(df)

# Extract column names
column_names = df.columns

# Print column names
print(column_names)

# Sample size of the smaller group (true samples)
sample_size = 44284

# Separate true and false samples
true_samples = df[df['Target'] == True]
false_samples = df[df['Target'] == False]

# Randomly sample from the false samples to match the sample size of the true samples
false_samples_balanced = false_samples.sample(n=sample_size, replace=False)

# Concatenate the balanced samples
balanced_df = pd.concat([true_samples, false_samples_balanced])

# Shuffle the DataFrame to randomize the order of samples
balanced_df = balanced_df.sample(frac=1).reset_index(drop=True)

# Convert target variable to binary integer
balanced_df['Target'] = balanced_df['Target'].astype(int)


# Separate features (X) and target (y)
X = balanced_df.drop('Target', axis=1).values
y = balanced_df['Target'].values

# Initialize K-fold cross-validation
kfold = KFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store evaluation metrics for each fold
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
conf_matrices = []

# Initialize lists to store ROC curve data
mean_fpr = np.linspace(0, 1, 100)
tprs = []

for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Create a StandardScaler object and fit on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost classifier
    classifier = XGBClassifier()
    classifier.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test_scaled)
    y_prob = classifier.predict_proba(X_test_scaled)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_value = roc_auc_score(y_test, y_prob)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Append scores to lists
    accuracy_scores.append(accuracy)
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    auc_scores.append(auc_value)
    conf_matrices.append(conf_matrix)

    # Compute ROC curve and area under the curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0

# Calculate and print the average scores across all folds
avg_accuracy = np.mean(accuracy_scores)
avg_precision = np.mean(precision_scores)
avg_recall = np.mean(recall_scores)
avg_f1 = np.mean(f1_scores)
avg_auc = np.mean(auc_scores)

std_accuracy = np.std(accuracy_scores)
std_precision = np.std(precision_scores)
std_recall = np.std(recall_scores)
std_f1 = np.std(f1_scores)
std_auc = np.std(auc_scores)

print("Average Scores Across All Folds:")
print(f"Average Accuracy: {avg_accuracy:.2f} ± {std_accuracy:.2f}")
print(f"Average Precision: {avg_precision:.2f} ± {std_precision:.2f}")
print(f"Average Recall: {avg_recall:.2f} ± {std_recall:.2f}")
print(f"Average F1 Score: {avg_f1:.2f} ± {std_f1:.2f}")
print(f"Average AUC Score: {avg_auc:.2f} ± {std_auc:.2f}")


# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), accuracy_scores, label='Accuracy', marker='o')
plt.plot(range(1, 11), precision_scores, label='Precision', marker='s')
plt.plot(range(1, 11), recall_scores, label='Recall', marker='^')
plt.plot(range(1, 11), f1_scores, label='F1 Score', marker='d')
plt.plot(range(1, 11), auc_scores, label='AUC Score', marker='p')
plt.xlabel('Fold')
plt.ylabel('Score')
# Customize the y-axis ticks
plt.yticks([0.65, 0.70, 0.75, 0.80])
plt.title('Cross-Validation Scores for XGBoost')
plt.grid(True)

plt.legend()
plt.show()

# Print and plot confusion matrices for each fold
for i, conf_matrix in enumerate(conf_matrices):
    print(f"\nConfusion Matrix for Fold {i+1}:")
    print(conf_matrix)

    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    plt.imshow(conf_matrix, cmap=plt.cm.Blues, interpolation='nearest')
    plt.title(f'Confusion Matrix - Fold {i+1}')
    plt.colorbar()
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.xticks([0, 1], ['False', 'True'])
    plt.yticks([0, 1], ['False', 'True'])
    plt.show()

# Calculate mean and standard deviation of scores across folds
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)

mean_precision = np.mean(precision_scores)
std_precision = np.std(precision_scores)

mean_recall = np.mean(recall_scores)
std_recall = np.std(recall_scores)

mean_f1 = np.mean(f1_scores)
std_f1 = np.std(f1_scores)

mean_auc = np.mean(auc_scores)
std_auc = np.std(auc_scores)

# Plot the results with error bars
plt.figure(figsize=(10, 6))
plt.errorbar(range(1, 11), accuracy_scores, yerr=std_accuracy, label='Accuracy', fmt='o')
plt.errorbar(range(1, 11), precision_scores, yerr=std_precision, label='Precision', fmt='s')
plt.errorbar(range(1, 11), recall_scores, yerr=std_recall, label='Recall', fmt='^')
plt.errorbar(range(1, 11), f1_scores, yerr=std_f1, label='F1 Score', fmt='d')
plt.errorbar(range(1, 11), auc_scores, yerr=std_auc, label='AUC Score', fmt='p')
plt.xlabel('Fold')
plt.ylabel('Score')
plt.title('Cross-Validation Scores for XGBoost with Error Bars')
plt.legend()
plt.show()


# Plot ROC curve for each fold
plt.figure(figsize=(8, 6))
for i, tpr in enumerate(tprs):
    plt.plot(mean_fpr, tpr, lw=1, alpha=0.3, label=f'ROC Fold {i+1}')

# Plot mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(auc_scores)
plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})', lw=2, alpha=0.8)

# Plot random guess line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Random Guess', alpha=0.8)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - XGBoost Classifier')
plt.legend(loc='lower right')
plt.show()

