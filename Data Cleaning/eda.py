# import pandas as pd

# # Enable the recommended behavior for future pandas versions
# pd.set_option('future.no_silent_downcasting', True)

# # Load the dataset
# file_path = 'combined_features.csv'
# df = pd.read_csv(file_path)

# # Drop the 'folder_name' column
# df.drop('folder_name', axis=1, inplace=True)

# # Convert 'True'/'False' to numerical values (1/0)
# df.replace({True: 1, False: 0}, inplace=True)

# # Save the updated file
# df.to_csv('combined_features_cleaned.csv', index=False)

# print("Data cleaned successfully and saved as 'combined_features_cleaned.csv'")

# import pandas as pd

# # Load the cleaned dataset
# file_path = 'combined_features_cleaned.csv'
# df = pd.read_csv(file_path)

# # Check for missing values
# missing_values = df.isnull().sum()

# # Display columns with missing values
# missing_values = missing_values[missing_values > 0]

# if missing_values.empty:
#     print("No missing values found in the dataset.")
# else:
#     print("Missing values found in the following columns:\n")
#     print(missing_values)


# import pandas as pd

# # Load the cleaned dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Count the number of unique TLDs
# num_tlds = df['tld'].nunique()

# print(f"Number of unique TLDs: {num_tlds}")
# print("\nTop 10 most common TLDs:")
# print(df['tld'].value_counts().head(10))


# import pandas as pd
# import pickle  # For saving the mapping

# # Load the cleaned dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Frequency Encoding for the 'tld' column
# tld_counts = df['tld'].value_counts().to_dict()
# df['tld'] = df['tld'].map(tld_counts)

# # Save the updated dataset
# df.to_csv('combined_features_encoded.csv', index=False)

# # Save the TLD mapping for reverse lookup
# with open('tld_mapping.pkl', 'wb') as f:
#     pickle.dump(tld_counts, f)

# print("Frequency encoding applied successfully. Saved as 'combined_features_encoded.csv'.")
# print("TLD mapping saved as 'tld_mapping.pkl'.")


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load encoded dataset
# df = pd.read_csv('combined_features_encoded.csv')

# # Correlation matrix
# plt.figure(figsize=(20, 20))
# sns.heatmap(df.corr(), cmap='coolwarm', annot=False, linewidths=0.5)
# plt.title('Feature Correlation Heatmap')
# plt.show()


# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load dataset
# df = pd.read_csv('combined_features_encoded.csv')

# # Compute correlation matrix
# correlation_matrix = df.corr().abs()

# # Identify highly correlated features (threshold = 0.9)
# high_corr_pairs = []
# for i in range(len(correlation_matrix.columns)):
#     for j in range(i):
#         if correlation_matrix.iloc[i, j] > 0.9:
#             high_corr_pairs.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

# # Convert to DataFrame for better visualization
# high_corr_df = pd.DataFrame(high_corr_pairs, columns=['Feature_1', 'Feature_2', 'Correlation'])
# print("Highly Correlated Features:\n", high_corr_df)

# # Dropping one feature from each highly correlated pair
# drop_features = set()
# for feature1, feature2, _ in high_corr_pairs:
#     if feature1 not in drop_features and feature2 not in drop_features:
#         drop_features.add(feature2)  # Arbitrarily keep feature1 and drop feature2

# print("\nFeatures dropped:", list(drop_features))

# # Drop features from the dataset
# df_reduced = df.drop(columns=list(drop_features))

# # Save reduced dataset
# df_reduced.to_csv('combined_features_reduced.csv', index=False)
# print("\nReduced dataset saved as 'combined_features_reduced.csv'")

# import pandas as pd

# # Load the reduced dataset
# df = pd.read_csv('combined_features_reduced.csv')

# # Count the label occurrences
# label_counts = df['label'].value_counts()

# print("Label 0 (Legitimate):", label_counts[0])
# print("Label 1 (Phishing):", label_counts[1])


# import pandas as pd
# from imblearn.under_sampling import RandomUnderSampler

# # Load the dataset
# df = pd.read_csv('combined_features_reduced.csv')

# # Features and labels
# X = df.drop('label', axis=1)
# y = df['label']

# # Initialize undersampler
# undersampler = RandomUnderSampler(random_state=42)

# # Perform undersampling
# X_resampled, y_resampled = undersampler.fit_resample(X, y)

# # Combine the resampled data
# df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

# # Save the resampled dataset
# df_resampled.to_csv('combined_features_undersampled.csv', index=False)

# print(f"Resampled dataset saved as 'combined_features_undersampled.csv'")
# print("New class distribution:")
# print(df_resampled['label'].value_counts())



# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# # Load the dataset
# df = pd.read_csv('combined_features_undersampled.csv')

# # Create 'boxplot' folder if not exists
# output_folder = 'boxplot'
# os.makedirs(output_folder, exist_ok=True)

# # Select numeric features
# numeric_features = df.select_dtypes(include=['int64', 'float64']).columns

# # Generate and save individual boxplots
# for feature in numeric_features:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(x=df[feature])
#     plt.title(f'Boxplot for {feature}')
#     plt.savefig(os.path.join(output_folder, f'{feature}_boxplot.png'))
#     plt.close()  # Close plot to avoid display issues

# print(f"Boxplots saved successfully in '{output_folder}' folder.")

# Check unique values and their counts

# value_counts = df['abnormal_url'].value_counts()

# # Display the results
# print("Value Counts for 'abnormal_url':")
# print(value_counts)

# # Percentage distribution for better clarity
# print("\nPercentage Distribution:")
# print((value_counts / len(df)) * 100)

# import pandas as pd
# import os

# # Load the dataset
# df = pd.read_csv('combined_features.csv')

# # Strip whitespace from string columns to avoid hidden character issues
# for col in df.columns:
#     if df[col].dtype == 'object':
#         df[col] = df[col].str.strip()

# # Identify and drop constant or nearly constant features
# threshold = 0.99  # 99% of the data having the same value
# drop_cols = [col for col in df.columns if df[col].value_counts(normalize=True).max() >= threshold]
# df.drop(columns=drop_cols, inplace=True)

# print(f"Dropped constant/nearly constant features: {drop_cols}")

# # Save the cleaned dataset
# df.to_csv('combined_features_cleaned.csv', index=False)
# print("Cleaned dataset saved as 'combined_features_cleaned.csv'")



# import pandas as pd

# # Load the dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Check value counts for 'alarm_window'
# value_counts = df['has_meta_refresh'].value_counts()
# percentage_distribution = df['has_meta_refresh'].value_counts(normalize=True) * 100

# print("Value Counts for 'domain_is_ip':")
# print(value_counts)

# print("\nPercentage Distribution:")
# print(percentage_distribution)


# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Feature to analyze
# feature = 'use_of_display_none'

# # Display value counts
# print(f"Value Counts for '{feature}':")
# print(df[feature].value_counts().head(10))

# # Distribution Plot
# plt.figure(figsize=(8, 5))
# sns.histplot(df[feature], bins=50, kde=True)
# plt.title(f'Distribution of {feature}')
# plt.xlabel(feature)
# plt.ylabel('Count')
# plt.show()

# # Statistics for deeper insights
# print("\nStatistics:")
# print(df[feature].describe())

# # Outlier Analysis (IQR method)
# Q1 = df[feature].quantile(0.25)
# Q3 = df[feature].quantile(0.75)
# IQR = Q3 - Q1

# outliers = df[(df[feature] < (Q1 - 1.5 * IQR)) | (df[feature] > (Q3 + 1.5 * IQR))]

# print(f"\nNumber of Outliers in '{feature}': {len(outliers)}")
# print(outliers[feature].describe())


# import pandas as pd
# import numpy as np

# # Load the dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Function to detect outliers using IQR
# def detect_outliers_iqr(data, feature):
#     Q1 = data[feature].quantile(0.25)
#     Q3 = data[feature].quantile(0.75)
#     IQR = Q3 - Q1

#     # Outlier bounds
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR

#     # Identify outliers
#     outliers = data[(data[feature] < lower_bound) | (data[feature] > upper_bound)]

#     print(f"Feature: {feature}")
#     print(f"Number of Outliers: {outliers.shape[0]}")
#     print(f"Outlier Summary:\n{outliers[feature].describe()}")
#     print("-" * 50)

#     return outliers

# # List of features to check for outliers
# features_to_check = ['use_of_display_none', 'alarm_window', 'brand_freq_domain']

# # Detect outliers for each feature
# for feature in features_to_check:
#     detect_outliers_iqr(df, feature)


# import pandas as pd
# import numpy as np
# from scipy.stats.mstats import winsorize

# # Load your dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Winsorization for 'use_of_display_none'
# # Limits: 0.01 caps lowest 1%, 0.01 caps highest 1% (adjust as needed)
# df['use_of_display_none_winsorized'] = winsorize(df['use_of_display_none'], limits=[0.01, 0.01])

# # Display before and after summary
# print("Before Winsorization:")
# print(df['use_of_display_none'].describe())
# print("\nAfter Winsorization:")
# print(df['use_of_display_none_winsorized'].describe())

# # Optional: Visualize the distribution before and after
# import matplotlib.pyplot as plt
# import seaborn as sns

# plt.figure(figsize=(12, 6))

# plt.subplot(1, 2, 1)
# sns.boxplot(df['use_of_display_none'])
# plt.title('Before Winsorization')

# plt.subplot(1, 2, 2)
# sns.boxplot(df['use_of_display_none_winsorized'])
# plt.title('After Winsorization')

# plt.tight_layout()
# plt.show()


# import pandas as pd

# # Load the dataset
# df = pd.read_csv('combined_features_cleaned.csv')

# # Drop constant features
# df.drop(['alarm_window', 'brand_freq_domain'], axis=1, inplace=True)

# # Save the cleaned dataset
# df.to_csv('dataset.csv', index=False)

# print("Dropped 'alarm_window' and 'brand_freq_domain'. Cleaned dataset saved as 'cleaned_dataset.csv'.")


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('dataset.csv')

# Separate features and labels
X = data.drop('label', axis=1)   # Features
y = data['label']                # Labels

# Split data into train (70%), validation (15%), and test (15%)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Feature Scaling (Recommended for Deep Learning models)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Display dataset shapes
print(f"Training Set: {X_train.shape}")
print(f"Validation Set: {X_val.shape}")
print(f"Test Set: {X_test.shape}")

# Optional: Save the processed data for reuse
pd.DataFrame(X_train).to_csv('X_train.csv', index=False)
pd.DataFrame(X_val).to_csv('X_val.csv', index=False)
pd.DataFrame(X_test).to_csv('X_test.csv', index=False)

pd.DataFrame(y_train).to_csv('y_train.csv', index=False)
pd.DataFrame(y_val).to_csv('y_val.csv', index=False)
pd.DataFrame(y_test).to_csv('y_test.csv', index=False)
