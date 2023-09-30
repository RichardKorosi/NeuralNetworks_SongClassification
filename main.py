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

df = pd.read_csv('./data/zadanie1_dataset.csv')

# Remove outliers (0,5b) -----------------------------------------------------------------------------------------------

# Print min and max values of columns before removing outliers
print("*" * 100, "Before removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]  # Some values were higher than 1
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 5)]  # Range should be <-60, 0>, one value is slightly above 0
df = df[(df['tempo'] > 0)]  # Some outliners found with 0 tempo
df = df[(df['duration_ms'] > 0) & (df['duration_ms'] <= 1967400)]  # Some outliners found (last one is long but valid)

# df = df[(df['energy'] >= 0) & (df['energy'] <= 1)]  # No outliners found (just in case)
# df = df[(df['speechiness'] >= 0) & (df['speechiness'] <= 1)]  # No outliners found (just in case)
# df = df[(df['acousticness'] >= 0) & (df['acousticness'] <= 1)]  # No outliners found (just in case)
# df = df[(df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1)]  # No outliners found (just in case)
# df = df[(df['liveness'] >= 0) & (df['liveness'] <= 1)]  # No outliners found (just in case)
# df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]  # No outliners found (just in case)
# df = df[(df['popularity'] >= 0) & (df['popularity'] <= 100)]  # No outliners found (just in case)
# df = df[(df['number_of_artists'] >= 1) & (df['number_of_artists'] <= 30)]  # No outliners found (just in case)

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
df = df.dropna(
    subset=['top_genre', 'popularity', 'number_of_artists'])  # Drop 167 rows with missing values in top_genre
df = df.drop(['name', 'url', 'genres', 'filtered_genres'], axis=1)  # Drop columns with no use

print("*" * 100, "Missing values after removing them", "*" * 100)
print(f"Lenght of dataset: {len(df)}")
print(df.isnull().sum())

# Column types and encoding (0,5b) ------------------------------------------------------------------------------------

# Print column types
print("*" * 100, "Column types", "*" * 100)
print(df.dtypes)

df = pd.get_dummies(df, columns=['top_genre'], prefix='', prefix_sep='')  # Encode top_genre column

# Use Label encoding for country
le = LabelEncoder()
df['emotion'] = le.fit_transform(df['emotion'])

print("*" * 100, "Column types", "*" * 100)
print(df.dtypes)

df.to_csv('./data/zadanie1_dataset_clean.csv', index=False)

# Split dataset into X and y (input and output) (0,5b) -----------------------------------------------------------------

# Split dataset into X and y
X = df.drop(columns=['emotion'])
y = df['emotion']

# Split dataset into train, valid and test
X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                    random_state=42)

# Scale and standardize data (0,5b) ------------------------------------------------------------------------------------

# Print dataset shapes
print("*" * 100, "Dataset shapes", "*" * 100)
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
print("*" * 100, "Before scaling/standardizing", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(X_train.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(X_train.max(numeric_only=True))

# Scale data
scaler = MinMaxScaler()
# !!!!!
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# Convert numpy arrays to pandas DataFrames
X_train = pd.DataFrame(X_train, columns=X.columns)
X_valid = pd.DataFrame(X_valid, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Plot histograms after scaling/standardizing
X_train[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists']].hist(bins=50, figsize=(20, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()


# Generate pie chart
sizes = (X_train[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall', 'disco', 'edm',
         'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial', 'j-pop', 'j-rock', 'jazz',
         'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']]
         .sum().sort_values(ascending=False).head(32))

plt.figure(figsize=(13, 13))
plt.title('Žánre')
plt.pie(sizes, autopct='', labels=None)

# Generate legend with labels and percentages
genres = sizes.index
percentages = [f'{genre}: {size/sum(sizes)*100:.1f}%' for genre, size in zip(genres, sizes)]
plt.legend(labels=percentages, loc='center', bbox_to_anchor=(1, 0.5), fontsize='large')

plt.show()

# Print min and max values of columns
print("*" * 100, "After scaling/standardizing", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(X_train.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(X_train.max(numeric_only=True))

# Train MLP model to predict country
print("*" * 100, "MLP", "*" * 100)
print(f"Random accuracy: {1 / len(y_train.unique())}")

clf = MLPClassifier(
    hidden_layer_sizes=(100, 100, 25, 15, 10),
    random_state=1,
    max_iter=100,
    validation_fraction=0.2,
    early_stopping=True,
    learning_rate='adaptive',
    learning_rate_init=0.001,
).fit(X_train, y_train)

# Predict on train set
y_pred = clf.predict(X_train)
print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
cm_train = confusion_matrix(y_train, y_pred)

# Predict on test set
y_pred = clf.predict(X_test)
print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
cm_test = confusion_matrix(y_test, y_pred)

# Create class names for confusion matrix
class_names = list(le.inverse_transform(clf.classes_))

disp = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on train set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

disp = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
disp.plot(ax=ax)
disp.ax_.set_title("Confusion matrix on test set")
disp.ax_.set(xlabel='Predicted', ylabel='True')
plt.show()

# Poznamky na opytanie
# 1. Pri filtrovani outlinerov sa mi rovno podarilo odstranovat aj null hodnoty, ci to je ok?
# 2. Vymazanie genres, filtered_genres okrem name a url, je to ok?
# 3. Po zakodovani neciselnych hodnot mam plno grafov. Je to ok (myslim ze ano)?
# 4. Emotion mozeme cez label encoding?
