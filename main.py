import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Allow printing more columns
pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

# Do not show pandas warnings
pd.set_option('mode.chained_assignment', None)

# Load the dataset https://ourworldindata.org/grapher/birth-rate-vs-death-rate
df = pd.read_csv('./data/zadanie1_dataset.csv')

# Remove outliers (0,5b) -----------------------------------------------------------------------------------------------

# Print min and max values of columns before removing outliers
print("*" * 100, "Before removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]  # Some values were higher than 1
df = df[(df['energy'] >= 0) & (df['energy'] <= 1)]  # No outliners found (just in case)
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 5)]  # Range should be <-60, 0>, one value is slightly above 0
df = df[(df['speechiness'] >= 0) & (df['speechiness'] <= 1)]  # No outliners found (just in case)
df = df[(df['acousticness'] >= 0) & (df['acousticness'] <= 1)]  # No outliners found (just in case)
df = df[(df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1)]  # No outliners found (just in case)
df = df[(df['liveness'] >= 0) & (df['liveness'] <= 1)]  # No outliners found (just in case)
df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]  # No outliners found (just in case)
df = df[(df['tempo'] > 0)]  # Some outliners found with 0 tempo
df = df[(df['duration_ms'] > 0) & (df['duration_ms'] <= 1967400)]  # Some outliners found (last one is long but valid)
df = df[(df['popularity'] >= 0) & (df['popularity'] <= 100)]  # No outliners found (just in case)
df = df[(df['number_of_artists'] >= 1) & (df['number_of_artists'] <= 30)]  # No outliners found (just in case)

# Print min and max values of columns
print("*" * 100, "After removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

# Handle missing values (0,5b) -----------------------------------------------------------------------------------------

# Count missing values in columns
print("*" * 100, "Missing values", "*" * 100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

# Deal with missing values and columns with no use
df = df.dropna(subset=['top_genre'])  # Drop 167 rows with missing values in top_genre
df = df.drop(['name', 'url', 'genres', 'filtered_genres'], axis=1)  # Drop columns with no use

print("*" * 100, "Missing values after removing them", "*" * 100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

# Column types and encoding (0,5b) ------------------------------------------------------------------------------------

# Print column types
print("*" * 100, "Column types", "*" * 100)
print(df.dtypes)

df = pd.get_dummies(df, columns=['top_genre'], prefix='top_genre_', prefix_sep='')  # Encode top_genre column
df = pd.get_dummies(df, columns=['emotion'], prefix='emotion_', prefix_sep='')  # Encode emotion column

print("*" * 100, "Column types", "*" * 100)
print(df.dtypes)

df.to_csv('./data/zadanie1_dataset_clean.csv', index=False)

# Split dataset into X and y
X = df.drop(columns=['emotion_calm', 'emotion_energetic', 'emotion_happy', 'emotion_sad'])
y = df[['emotion_calm', 'emotion_energetic', 'emotion_happy', 'emotion_sad']]

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                    random_state=42)

# Print dataset shapes
print("*"*100, "Dataset shapes", "*"*100)
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# Plot histograms before scaling
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*100, "Before scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))

# Scale data
scaler = MinMaxScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Standardize data
scaler = StandardScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Plot histograms after scaling/standardizing
X_train.hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()

# Print min and max values of columns
print("*"*100, "After scaling/standardizing", "*"*100)
print("-"*10, "Min", "-"*10)
print(X_train.min(numeric_only=True))
print("-"*10, "Max", "-"*10)
print(X_train.max(numeric_only=True))
