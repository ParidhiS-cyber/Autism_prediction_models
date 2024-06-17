import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from scipy.stats import chi2_contingency, fisher_exact
import seaborn as sns
import matplotlib.pyplot as plt

# Load the CSV file
file = 'C:\\Users\\PARIDHI\\Downloads\\merged_types.csv'
df = pd.read_csv(file)

# Remove unnecessary columns
del df['Sample']
df = df.fillna(0)
#del df['missense_chrY']
print(df)

# Extract column names
column_names = df.columns

# Print column names
print(column_names)

# Sample size of the smaller group (true samples)
sample_size = 44224

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

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=10000)

# Stratified K-Fold cross-validation
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Lists to store confusion matrices for each fold
conf_matrices = []

# Perform 10-fold cross-validation
for train_index, test_index in kf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    conf_matrices.append(cm)

# Get the confusion matrix for the 10th fold
cm_10th_fold = conf_matrices[9]

# Print the confusion matrix
print("Confusion Matrix for the 10th fold:")
print(cm_10th_fold)

# Perform Chi-square test
chi2, p, dof, ex = chi2_contingency(cm_10th_fold)
print("\nChi-square test:")
print(f"Chi2: {chi2}, p-value: {p}, degrees of freedom: {dof}")

# Perform Fisher's exact test (only valid for 2x2 matrix)
oddsratio, p_fisher = fisher_exact(cm_10th_fold)
print("\nFisher's exact test:")
print(f"Odds ratio: {oddsratio}, p-value: {p_fisher}")