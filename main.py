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

# Print min and max values of columns before removing outliers
print("*" * 100, "Before removing outliers", "*" * 100)
print("-" * 10, "Min", "-" * 10)
print(df.min(numeric_only=True))
print("-" * 10, "Max", "-" * 10)
print(df.max(numeric_only=True))

# Remove outliers
df = df[(df['danceability'] >= 0) & (df['danceability'] <= 1)]  # Some values were higher than 1
df = df[(df['energy'] >= 0) & (df['energy'] <= 1)]  # No outliners found (just in case)
df = df[(df['loudness'] >= -60) & (df['loudness'] <= 5)]  # Range should be <-60, 0>, one value is slightly above 0
df = df[(df['speechiness'] >= 0) & (df['speechiness'] <= 1)]  # No outliners found (just in case)
df = df[(df['acousticness'] >= 0) & (df['acousticness'] <= 1)]  # No outliners found (just in case)
df = df[(df['instrumentalness'] >= 0) & (df['instrumentalness'] <= 1)]  # No outliners found (just in case)
df = df[(df['liveness'] >= 0) & (df['liveness'] <= 1)]  # No outliners found (just in case)
df = df[(df['valence'] >= 0) & (df['valence'] <= 1)]  # No outliners found (just in case)
