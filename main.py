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

# Handle outliers (0,5b) -----------------------------------------------------------------------------------------------

# Print min and max values of columns before removing outliers
print("*" * 100, "Before removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]  # Some values were higher than 1
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 5)]  # Range should be <-60, 0>, one value is slightly above 0
df = df[(df['tempo'] > 0)]  # Some outliers found with 0 tempo
df = df[
    (df['duration_ms'] > 20000) & (df['duration_ms'] <= 1967400)]  # Some outliers found (last one is long but valid)

# df = df[(df['energy'] >= 0) & (df['energy'] <= 1)]  # No outliers found (just in case)
# df = df[(df['speechiness'] >= 0) & (df['speechiness'] <= 1)]  # No outliers found (just in case)
# df = df[(df['acousticness'] >= 0) & (df['acousticness'] <= 1)]  # No outliers found (just in case)
# df = df[(df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1)]  # No outliers found (just in case)
# df = df[(df['liveness'] >= 0) & (df['liveness'] <= 1)]  # No outliers found (just in case)
# df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]  # No outliers found (just in case)
# df = df[(df['popularity'] >= 0) & (df['popularity'] <= 100)]  # No outliers found (just in case)
# df = df[(df['number_of_artists'] >= 1) & (df['number_of_artists'] <= 30)]  # No outliers found (just in case)

# Print min and max values of columns
print("*" * 100, "After removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

# Handle missing values (0,5b) -----------------------------------------------------------------------------------------

# Count missing values in columns
print("*" * 100, "Missing values", "*" * 100)
print(f"Length of dataset: {len(df)}")
print(df.isnull().sum())

# Deal with missing values and columns with no use
df = df.dropna(
    subset=['top_genre', 'popularity', 'number_of_artists'])  # Drop rows with missing values in certain columns
df = df.drop(['name', 'url', 'genres', 'filtered_genres'], axis=1)  # Drop columns with no use

print("*" * 100, "Missing values after removing them", "*" * 100)
print(f"Length of dataset: {len(df)}")
print(df.isnull().sum())

# Column types and encoding (0,5b) ------------------------------------------------------------------------------------

# Print column types
print("*" * 100, "Column types", "*" * 100)
print(df.dtypes)

# Use dummy encoding for top_genre
df = pd.get_dummies(df, columns=['top_genre'], prefix='', prefix_sep='')

# Use Label encoding for emotion
le = LabelEncoder()
df['emotion'] = le.fit_transform(df['emotion'])

print("*" * 100, "Column types", "*" * 100)

# Change boolean columns to float (False = 0, True = 1)
for col in ['explicit', 'ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
            'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial', 'j-pop',
            'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock', 'rockabilly', 'ska',
            'sleep', 'soul']:
    df[col] = df[col].astype(float)
print(df.dtypes)

# Split dataset into X and y (input and output) (1b) -------------------------------------------------------------------

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
print(f"Full dataset: {df.shape}")
print(f"X_train: {X_train.shape}")
print(f"X_valid: {X_valid.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_valid: {y_valid.shape}")
print(f"y_test: {y_test.shape}")

# Print min and max values of columns
print("*" * 100, "Before scaling/standardizing", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(X_train.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(X_train.max(numeric_only=True))

# Plot histograms before scaling (for interval attributes) (excludes explicit, genres columns)
X_train[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists']].hist(bins=70, figsize=(15, 15))
plt.suptitle('Histograms before scaling/standardizing')
plt.show()

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

# Print min and max values of columns
print("*" * 100, "After scaling/standardizing", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(X_train.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(X_train.max(numeric_only=True))

# Plot histograms after scaling (for interval attributes) (excludes explicit, genres)
X_train[['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists']].hist(bins=70, figsize=(15, 15))
plt.suptitle('Histograms after scaling/standardizing')
plt.show()

# Piechart of explicit-----------
sizes = (X_train['explicit'].value_counts() / len(X_train['explicit'])).sort_values(ascending=True)
plt.figure(figsize=(15, 15))
plt.title(f'Explicit [{len(X_train)}]')
plt.pie(sizes, autopct='', labels=None)

numbers = sizes.index
percentages = [f'True: {size * 100:.1f}%' if number == 1.0 else f'False: {size * 100:.1f}%' for number, size in
               zip(numbers, sizes)]
plt.legend(labels=percentages, title="True/False:Percento", loc='center', bbox_to_anchor=(1, 0.5), fontsize='large')
plt.show()

# Piechart of genres-------
sizes = (
    X_train[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall', 'disco', 'edm',
             'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial', 'j-pop', 'j-rock', 'jazz',
             'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']]
    .sum().sort_values(ascending=False).head(32))
plt.figure(figsize=(15, 15))
plt.title(f'Žánre [{len(X_train)}]')
plt.pie(sizes, autopct='', labels=None)

genres = sizes.index
percentages = [f'{genre}: {size / sum(sizes) * 100:.1f}%' for genre, size in zip(genres, sizes)]
plt.legend(labels=percentages, title="Žáner:Percento", loc='center', bbox_to_anchor=(1, 0.5), fontsize='large')
plt.show()

# Train MLP model to predict emotion (1b) ------------------------------------------------------------------------------

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

# Print Results and confusion matrix of results (1b) -------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------------
