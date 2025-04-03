import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from collections import Counter

# Load the dataset
df = pd.read_csv("breast-cancer.csv")

# Separate features and target
X = df.drop(columns=["id", "diagnosis"])
y = df["diagnosis"]

# Check the class distribution before upsampling
print("Original class distribution:", Counter(y))

# Calculate the number of samples needed for each class to make the total 5000
class_counts = y.value_counts()
total_samples = 5000

# Calculate the ratio for each class
class_ratio = class_counts / class_counts.sum()

# Calculate the number of samples for each class to reach a total of 5000
target_samples = (class_ratio * total_samples).round().astype(int)

# Print target samples for each class
print(f"Target samples for each class: {target_samples}")

# Set up RandomOverSampler with a dictionary specifying the target number of samples per class
sampling_strategy = {
    'M': target_samples['M'],
    'B': target_samples['B']
}

# Ensure sampling_strategy is correct (type and value check)
print(f"Sampling strategy: {sampling_strategy}")

# Initialize RandomOverSampler
ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=42)

# Perform oversampling
X_resampled, y_resampled = ros.fit_resample(X, y)

# Check the class distribution after upsampling
print("Resampled class distribution:", Counter(y_resampled))

# Recombine the features and target into a single DataFrame
df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
df_resampled["diagnosis"] = y_resampled

# Optionally add the "id" column back if necessary
df_resampled["id"] = range(1, len(df_resampled) + 1)

# Ensure the total rows are 5000
print(f"Total rows in the resampled dataset: {df_resampled.shape[0]}")

# Optionally, save the resampled dataset to a new CSV
df_resampled.to_csv("breast_cancer_resampled.csv", index=False)
