import pandas as pd
import numpy as np
import shap
import xgboost
import matplotlib.pyplot as plt

# Load data
file = '/home/group_shyam01/merged_types.csv'
df = pd.read_csv(file)
#df = df.fillna(0)
del df['Sample']

# Sample balancing
sample_size = 44224
true_samples = df[df['Target'] == True]
false_samples = df[df['Target'] == False]
false_samples_balanced = false_samples.sample(n=sample_size, replace=False)
balanced_df = pd.concat([true_samples, false_samples_balanced]).sample(frac=1).reset_index(drop=True)

# Convert target to int
balanced_df['Target'] = balanced_df['Target'].apply(int)

# Separate features and target
X = balanced_df.drop('Target', axis=1)
y = balanced_df['Target']

# Train model
model = xgboost.XGBClassifier().fit(X, y)

# Compute SHAP values
explainer = shap.Explainer(model, X)
shap_values = explainer(X)

# Adjust plot size and layout
plt.figure(figsize=(15, 10))
shap.plots.beeswarm(shap_values, max_display=20)
plt.show()