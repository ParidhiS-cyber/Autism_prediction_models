# Import the necessary libraries
import shap
import pandas as pd
import numpy as np
shap.initjs()
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Load data
file = '/home/group_shyam01/merged_types.csv'
df = pd.read_csv(file)
#df = df.fillna(0)
del df['Sample']
#del df['ChromosomeY']

# Sample balancing
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


balanced_df['Target'] = balanced_df['Target'].apply(int)
print(balanced_df)
# Separate features (X) and target (y)
X = balanced_df.drop('Target', axis=1)
y = balanced_df['Target']
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Create a StandardScaler object and fit on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a machine learning model
import xgboost as xgb
clf = xgb.XGBClassifier()
clf.fit(X_train_scaled, y_train)

# Make prediction on the testing data
y_pred = clf.predict(X_test_scaled)

# Classification Report
print(classification_report(y_pred, y_test))

# Compute SHAP values
explainer = shap.Explainer(clf, X_train_scaled, feature_names=X.columns)
shap_values = explainer(X_test_scaled)

# Plot SHAP summary plot
plt.figure(figsize=(15, 10))
shap.plots.beeswarm(shap_values, max_display=20)
#shap.summary_plot(shap_values, X_test_scaled, feature_names=X.columns)

# Automatically adjust subplot parameters to give some padding
plt.tight_layout()

plt.show()