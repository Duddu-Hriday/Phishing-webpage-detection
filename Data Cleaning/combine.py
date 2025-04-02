import pandas as pd

# Load the phishing and legitimate data
phishing_df = pd.read_csv('phishing_features_parallel.csv')
legitimate_df = pd.read_csv('legitimate_features.csv')

# Add a label column: 1 for phishing, 0 for legitimate
phishing_df['label'] = 1
legitimate_df['label'] = 0

# Combine both datasets
combined_df = pd.concat([phishing_df, legitimate_df], ignore_index=True)

# Save the combined dataframe to a new CSV file
combined_df.to_csv('combined_features.csv', index=False)

print("Combined CSV file saved as 'combined_features.csv'")
