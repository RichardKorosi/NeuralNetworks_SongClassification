import pandas as pd
import numpy as np
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Dense
from keras.src.optimizers import Adam, SGD
from sklearn.feature_selection import RFE
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
import keras
import seaborn as sns
from keras import layers, Sequential
from imblearn.over_sampling import SMOTE, RandomOverSampler

# ZDROJE KU KODOM ------------------------------------------------------------------------------------------------------
# ======================================================================================================================
# Zdrojove kody z cviceni:
#   Autor: Ing. Vanesa Andicsová
#   Subory:
#       seminar2.py
#       main.py
# ======================================================================================================================
# Grafy, Pomocne funkcie, SMOTE...:
#  Autor: Github Copilot, ChatGPT
#  Grafy, pomocne funkcie  boli vypracoavane za pomoci ChatGPT a GithubCopilota
# ======================================================================================================================

# Uvod -----------------------------------------------------------------------------------------------------------------
# Uvod bol inspirovany zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

pd.options.display.width = None
pd.options.display.max_columns = None
pd.set_option('display.max_rows', 3000)
pd.set_option('display.max_columns', 3000)

pd.set_option('mode.chained_assignment', None)

df = pd.read_csv('./data/zadanie1_dataset.csv')
dfGen = pd.read_csv('./data/zadanie1_dataset.csv')
dfThird = pd.read_csv('./data/zadanie1_dataset.csv')
dfBonus = pd.read_csv('./data/zadanie1_dataset.csv')


# Functions ------------------------------------------------------------------------------------------------------------
def handleOutliersAndMissingValues(dframe, mode=0):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

    # Handle outliers
    print("*" * 100, "Before removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    dframe = dframe[(dframe['danceability'] >= 0) & (dframe['danceability'] <= 1)]
    dframe = dframe[(dframe['loudness'] >= -60) & (
            dframe['loudness'] <= 5)]
    dframe = dframe[(dframe['tempo'] > 0)]
    dframe = dframe[(dframe['duration_ms'] > 20000) & (
            dframe['duration_ms'] <= 1967400)]

    print("*" * 100, "After removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    # Handle missing values
    print("*" * 100, "Missing values", "*" * 100)
    print(f"Length of dataset: {len(dframe)}")
    print(dframe.isnull().sum())

    if mode == 0:
        dframe = dframe.dropna(
            subset=['top_genre', 'popularity', 'number_of_artists'])
        dframe = dframe.drop(['name', 'url', 'genres', 'filtered_genres'], axis=1)
    else:
        dframe = dframe.dropna(
            subset=['top_genre', 'popularity', 'number_of_artists'])
        dframe = dframe.drop(['name', 'url'], axis=1)

    print("*" * 100, "Missing values after removing them", "*" * 100)
    print(f"Length of dataset: {len(dframe)}")
    print(dframe.isnull().sum())

    return dframe


def encodeGenres(dframe, mode=0):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

    if mode == 0:
        # First Part

        # Column types and encoding
        print("*" * 100, "Column types", "*" * 100)
        print(dframe.dtypes)

        dframe = pd.get_dummies(dframe, columns=['top_genre'], prefix='', prefix_sep='')

        le = LabelEncoder()
        dframe['emotion'] = le.fit_transform(df['emotion'])

        print("*" * 100, "Column types", "*" * 100)

        for col in ['explicit', 'ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                    'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial',
                    'j-pop',
                    'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock', 'rockabilly',
                    'ska',
                    'sleep', 'soul']:
            dframe[col] = dframe[col].astype(float)
        print(dframe.dtypes)

        return dframe, le
    elif mode == 1:
        # Second part (EDA)

        le = LabelEncoder()
        dframe['emotion_le'] = le.fit_transform(df['emotion'])

        import ast

        dframe['filtered_genres'] = dframe['filtered_genres'].apply(ast.literal_eval)

        genres_encoded = dframe['filtered_genres'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0)

        dframe = pd.concat([dframe, genres_encoded], axis=1)

        dframe['number_of_genres'] = dframe['genres'].str.split(',').str.len()
        dframe['number_of_genres'] = dframe['number_of_genres'].fillna(0)
        dframe['number_of_genres'] = dframe['number_of_genres'].astype(int)

        return dframe
    else:
        # KERAS
        dframe = pd.get_dummies(dframe, columns=['top_genre'], prefix='', prefix_sep='')
        dframe = pd.get_dummies(dframe, columns=['emotion'], prefix='', prefix_sep='')

        for col in ['explicit', 'ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                    'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial',
                    'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock',
                    'rockabilly', 'ska', 'sleep', 'soul', 'calm', 'energetic', 'happy', 'sad']:
            dframe[col] = dframe[col].astype(float)
        return dframe


def createHistogramsPartOne(X_train, time):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

    attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists']

    X_train[attributes].hist(bins=70, figsize=(15, 15))

    for ax in plt.gcf().get_axes():
        ax.set_xlabel('Hodnota')
        ax.set_ylabel('Počet')
    if time == "before":
        plt.suptitle('Histograms before scaling/standardizing')
    else:
        plt.suptitle('Histograms after scaling/standardizing')
    plt.show()

    return None


def createPiechartsPartOne(X_train):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    # Piechart of explicit
    sizes = (X_train['explicit'].value_counts() / len(X_train['explicit'])).sort_values(ascending=True)
    plt.figure(figsize=(15, 15))
    plt.title(f'Explicit [{len(X_train)}]')
    plt.pie(sizes, autopct='', labels=None)

    numbers = sizes.index
    percentages = [f'True: {size * 100:.1f}%' if number == 1.0 else f'False: {size * 100:.1f}%' for number, size in
                   zip(numbers, sizes)]
    plt.legend(labels=percentages, title="True/False:Percento", loc='center', bbox_to_anchor=(1, 0.5), fontsize='large')
    plt.show()

    # Piechart of genres
    sizes = (
        X_train[
            ['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall', 'disco', 'edm',
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


def restOfFirstPart(dframe):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

    X = dframe.drop(columns=['emotion'])
    y = dframe['emotion']

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=42)

    # Scale and standardize data
    print("*" * 100, "Dataset shapes", "*" * 100)
    print(f"Full dataset: {dframe.shape}")
    print(f"X_train: {X_train.shape}")
    print(f"X_valid: {X_valid.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_valid: {y_valid.shape}")
    print(f"y_test: {y_test.shape}")

    print("*" * 100, "Before scaling/standardizing", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(X_train.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(X_train.max(numeric_only=True))

    createHistogramsPartOne(X_train, "before")
    createPiechartsPartOne(X_train)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    print("*" * 100, "After scaling/standardizing", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(X_train.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(X_train.max(numeric_only=True))

    createHistogramsPartOne(X_train, "after")

    # Train MLP model to predict emotion
    print("*" * 100, "MLP", "*" * 100)
    print(f"Random accuracy: {1 / len(y_train.unique())}")

    clf = MLPClassifier(
        hidden_layer_sizes=(200, 150, 100),
        random_state=1,
        max_iter=250,
        early_stopping=True
    ).fit(X_train, y_train)

    # Print Results and confusion matrix of results
    y_pred = clf.predict(X_train)
    print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred))
    cm_train = confusion_matrix(y_train, y_pred)

    y_pred = clf.predict(X_test)
    print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred))
    cm_test = confusion_matrix(y_test, y_pred)

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


def createCorrelationHeatmaps(dframe, dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    # DFRAME1
    # ALL -> BASICS, TOP_GENRE, EMOTION (LE), EXPLICIT
    correlation_matrix = dframe.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap ALL(top_genre)')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    # ONLY -> TOP_GENRE
    dframeEdit = dframe.drop(
        ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists', 'explicit', 'emotion'], axis=1)
    correlation_matrix = dframeEdit.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap GENRE (top_genre)')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    # DFRAME2
    # ALL -> BASICS, FILTERED_GENRE, EMOTION_LE, EXPLICIT
    dframeGenEdit = dframeGen.drop(['genres', 'filtered_genres', 'top_genre', 'emotion'], axis=1)
    correlation_matrix = dframeGenEdit.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap ALL (filtered_genres)')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    # ONLY -> FILTERED_GENRE
    dframeGenEdit = dframeGen.drop(
        ['genres', 'filtered_genres', 'top_genre', 'emotion', 'danceability', 'energy', 'loudness', 'speechiness',
         'acousticness', 'instrumentalness', 'liveness',
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists', 'number_of_genres', 'explicit',
         'emotion_le'], axis=1)
    correlation_matrix = dframeGenEdit.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap GENRE (filtered_genres)')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    return None


def createAnalysisEnergyLoudness(dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    plt.figure(figsize=(10, 6))
    plt.scatter(dframeGen['energy'], dframeGen['loudness'], color='skyblue')
    plt.title('Scatter Plot')
    plt.xlabel('energy')
    plt.ylabel('loudness')
    plt.show()

    num_intervals = 12
    dframeGen['energy_interval'] = pd.cut(dframeGen['energy'],
                                          bins=np.linspace(dframeGen['energy'].min(), dframeGen['energy'].max(),
                                                           num_intervals + 1))
    average_energy = dframeGen.groupby('energy_interval')['loudness'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(average_energy.index.categories.mid, average_energy.values, marker='o', color='skyblue')
    plt.title('Average loudness vs. energy')
    plt.xlabel('energy')
    plt.ylabel('Average Loudness')
    plt.grid(True)
    plt.show()

    return None


def createAnalysisLivenessSpeechiness(dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    plt.figure(figsize=(10, 6))
    plt.scatter(dframeGen['liveness'], dframeGen['speechiness'], color='skyblue')
    plt.title('Scatter Plot')
    plt.xlabel('liveness')
    plt.ylabel('speechiness')
    plt.show()

    num_intervals = 12
    dframeGen['liveness_interval'] = pd.cut(dframeGen['liveness'],
                                            bins=np.linspace(dframeGen['liveness'].min(), dframeGen['liveness'].max(),
                                                             num_intervals + 1))
    plt.figure(figsize=(10, 10))
    sns.stripplot(x=dframeGen['liveness_interval'], y=dframeGen['speechiness'], palette='viridis')
    plt.title('Strip Plot')
    plt.xlabel('liveness_interval')
    plt.ylabel('speechiness')
    plt.xticks(rotation=25)
    plt.show()

    return None


def createAnalysisDanceabilityTopGenre(dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    plt.figure(figsize=(10, 6))
    plt.scatter(dframeGen['danceability'], dframeGen['top_genre'], color='skyblue')
    plt.title('Scatter Plot')
    plt.xlabel('danceability')
    plt.ylabel('top_genre')
    plt.show()

    means = [dframeGen['danceability'][dframeGen['top_genre'] == genre].mean() for genre in
             dframeGen['top_genre'].unique()]

    genre_means = dict(zip(dframeGen['top_genre'].unique(), means))

    sorted_genres = sorted(dframeGen['top_genre'].unique(), key=lambda genre: genre_means[genre])

    plt.figure(figsize=(15, 10))
    plt.title('Danceability by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Danceability')

    plt.boxplot([dframeGen['danceability'][dframeGen['top_genre'] == genre] for genre in sorted_genres],
                labels=sorted_genres)

    sorted_means = [genre_means[genre] for genre in sorted_genres]
    plt.plot(range(1, len(sorted_genres) + 1), sorted_means, 'rx')
    plt.xticks(rotation=45)
    plt.show()

    return None


def generate_three_pie_charts(ax, labels, ratios, title):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    ax.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)


def generate_pie_chart_genres(ax, labels, ratios, title):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    ax.pie(ratios, labels=None, autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired(range(len(labels))))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title(title)
    ax.legend(labels, loc='lower left')  # Move legend to bottom-right


def createAnalysisTopGenreFilteredGenres(dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    dframeGenEdit = dframeGen[['forro', 'sertanejo', 'j-pop', 'anime', 'funk', 'soul']]
    correlation_matrix = dframeGenEdit.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap Selected Genres')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()

    funk_only = dframeGen[(dframeGen['funk'] == True) & (dframeGen['soul'] == False)]

    soul_only = dframeGen[(dframeGen['funk'] == False) & (dframeGen['soul'] == True)]

    funk_and_soul = dframeGen[(dframeGen['funk'] == True) & (dframeGen['soul'] == True)]

    funk_alone = dframeGen[(dframeGen['funk'] == True) & (dframeGen[
                                                              ['edm', 'house', 'country', 'pop', 'rock', 'soul', 'folk',
                                                               'metal', 'grunge', 'metalcore', 'punk',
                                                               'emo', 'bluegrass', 'ska', 'reggaeton', 'reggae',
                                                               'forro', 'sertanejo', 'industrial', 'hardstyle',
                                                               'trance', 'dancehall', 'jazz', 'dubstep', 'blues',
                                                               'ambient', 'classical', 'disco', 'rockabilly',
                                                               'sleep', 'j-pop', 'anime', 'afrobeat', 'electro',
                                                               'k-pop', 'samba', 'j-rock', 'hardcore', 'grindcore',
                                                               'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                               'gospel']] == 0).all(axis=1)]

    soul_alone = dframeGen[
        (dframeGen['soul'] == True) & (dframeGen[['edm', 'house', 'country', 'pop', 'rock', 'funk', 'folk',
                                                  'metal', 'grunge', 'metalcore', 'punk',
                                                  'emo', 'bluegrass', 'ska', 'reggaeton', 'reggae',
                                                  'forro', 'sertanejo', 'industrial', 'hardstyle',
                                                  'trance', 'dancehall', 'jazz', 'dubstep', 'blues',
                                                  'ambient', 'classical', 'disco', 'rockabilly',
                                                  'sleep', 'j-pop', 'anime', 'afrobeat', 'electro',
                                                  'k-pop', 'samba', 'j-rock', 'hardcore', 'grindcore',
                                                  'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                  'gospel']] == 0).all(axis=1)]

    funk_alone_ratio = len(funk_alone) / len(dframeGen)
    funk_only_ratio = len(funk_only) / len(dframeGen) - funk_alone_ratio
    soul_alone_ratio = len(soul_alone) / len(dframeGen)
    soul_only_ratio = len(soul_only) / len(dframeGen) - soul_alone_ratio
    funk_and_soul_ratio = len(funk_and_soul) / len(dframeGen)

    forro_only = dframeGen[(dframeGen['forro'] == True) & (dframeGen['sertanejo'] == False)]

    sertanejo_only = dframeGen[(dframeGen['forro'] == False) & (dframeGen['sertanejo'] == True)]

    forro_and_sertanejo = dframeGen[(dframeGen['forro'] == True) & (dframeGen['sertanejo'] == True)]

    forro_alone = dframeGen[(dframeGen['forro'] == True) & (dframeGen[
                                                                ['edm', 'house', 'country', 'pop', 'rock', 'soul',
                                                                 'folk',
                                                                 'metal', 'grunge', 'metalcore', 'punk',
                                                                 'emo', 'bluegrass', 'ska', 'reggaeton', 'reggae',
                                                                 'funk', 'sertanejo', 'industrial', 'hardstyle',
                                                                 'trance', 'dancehall', 'jazz', 'dubstep', 'blues',
                                                                 'ambient', 'classical', 'disco', 'rockabilly',
                                                                 'sleep', 'j-pop', 'anime', 'afrobeat', 'electro',
                                                                 'k-pop', 'samba', 'j-rock', 'hardcore', 'grindcore',
                                                                 'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                                 'gospel']] == 0).all(axis=1)]
    sertanejo_alone = dframeGen[(dframeGen['sertanejo'] == True) & (dframeGen[
                                                                        ['edm', 'house', 'country', 'pop', 'rock',
                                                                         'soul', 'folk',
                                                                         'metal', 'grunge', 'metalcore', 'punk',
                                                                         'emo', 'bluegrass', 'ska', 'reggaeton',
                                                                         'reggae',
                                                                         'funk', 'forro', 'industrial', 'hardstyle',
                                                                         'trance', 'dancehall', 'jazz', 'dubstep',
                                                                         'blues',
                                                                         'ambient', 'classical', 'disco', 'rockabilly',
                                                                         'sleep', 'j-pop', 'anime', 'afrobeat',
                                                                         'electro',
                                                                         'k-pop', 'samba', 'j-rock', 'hardcore',
                                                                         'grindcore',
                                                                         'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                                         'gospel']] == 0).all(axis=1)]
    forro_alone_ratio = len(forro_alone) / len(dframeGen)
    forro_only_ratio = len(forro_only) / len(dframeGen) - forro_alone_ratio
    sertanejo_alone_ratio = len(sertanejo_alone) / len(dframeGen)
    sertanejo_only_ratio = len(sertanejo_only) / len(dframeGen) - sertanejo_alone_ratio
    forro_and_sertanejo_ratio = len(forro_and_sertanejo) / len(dframeGen)

    jpop_only = dframeGen[(dframeGen['j-pop'] == True) & (dframeGen['anime'] == False)]

    anime_only = dframeGen[(dframeGen['j-pop'] == False) & (dframeGen['anime'] == True)]

    jpop_and_anime = dframeGen[(dframeGen['j-pop'] == True) & (dframeGen['anime'] == True)]

    jpop_alone = dframeGen[(dframeGen['j-pop'] == True) & (dframeGen[
                                                               ['edm', 'house', 'country', 'pop', 'rock', 'soul',
                                                                'folk',
                                                                'metal', 'grunge', 'metalcore', 'punk',
                                                                'emo', 'bluegrass', 'ska', 'reggaeton', 'reggae',
                                                                'funk', 'sertanejo', 'industrial', 'hardstyle',
                                                                'trance', 'dancehall', 'jazz', 'dubstep', 'blues',
                                                                'ambient', 'classical', 'disco', 'rockabilly',
                                                                'sleep', 'anime', 'afrobeat', 'electro',
                                                                'k-pop', 'samba', 'j-rock', 'hardcore', 'grindcore',
                                                                'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                                'gospel']] == 0).all(axis=1)]
    anime_alone = dframeGen[(dframeGen['anime'] == True) & (dframeGen[
                                                                ['edm', 'house', 'country', 'pop', 'rock', 'soul',
                                                                 'folk',
                                                                 'metal', 'grunge', 'metalcore', 'punk',
                                                                 'emo', 'bluegrass', 'ska', 'reggaeton', 'reggae',
                                                                 'funk', 'sertanejo', 'industrial', 'hardstyle',
                                                                 'trance', 'dancehall', 'jazz', 'dubstep', 'blues',
                                                                 'ambient', 'classical', 'disco', 'rockabilly',
                                                                 'sleep', 'j-pop', 'afrobeat', 'electro',
                                                                 'k-pop', 'samba', 'j-rock', 'hardcore', 'grindcore',
                                                                 'j-idol', 'opera', 'comedy', 'tango', 'techno',
                                                                 'gospel']] == 0).all(axis=1)]
    jpop_alone_ratio = len(jpop_alone) / len(dframeGen)
    jpop_only_ratio = len(jpop_only) / len(dframeGen) - jpop_alone_ratio
    anime_alone_ratio = len(anime_alone) / len(dframeGen)
    anime_only_ratio = len(anime_only) / len(dframeGen) - anime_alone_ratio
    jpop_and_anime_ratio = len(jpop_and_anime) / len(dframeGen)

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    labels1 = ['Soul alone', 'Soul with others', 'Funk alone', 'Funk with others', 'Funk and Soul']
    ratios1 = [soul_alone_ratio, soul_only_ratio, funk_alone_ratio, funk_only_ratio, funk_and_soul_ratio]

    labels2 = ['Forro alone', 'Forro with others', 'Sertanejo alone', 'Sertanejo with others', 'Forro and Sertanejo']
    ratios2 = [forro_alone_ratio, forro_only_ratio, sertanejo_alone_ratio, sertanejo_only_ratio,
               forro_and_sertanejo_ratio]

    labels3 = ['J-Pop alone', 'J-Pop with others', 'Anime alone', 'Anime with others', 'J-Pop and Anime']
    ratios3 = [jpop_alone_ratio, jpop_only_ratio, anime_alone_ratio, anime_only_ratio, jpop_and_anime_ratio]

    generate_pie_chart_genres(axes[0], labels1, ratios1, 'Distribution of Funk and Soul')
    generate_pie_chart_genres(axes[1], labels2, ratios2, 'Distribution of Forro and Sertanejo')
    generate_pie_chart_genres(axes[2], labels3, ratios3, 'Distribution of J-Pop and Anime')

    plt.tight_layout()

    plt.show()

    return None


def createAnalysisComedySpeechiness(dframeGen):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    plt.scatter(dframeGen['comedy'], dframeGen['speechiness'])
    plt.title("Correlation between Comedy and Speechiness")
    plt.xlabel("Comedy")
    plt.ylabel("Speechiness")
    plt.show()

    num_intervals = 20
    plt.figure(figsize=(12, 11))

    dframeGen['speechiness_interval'] = pd.cut(dframeGen['speechiness'], num_intervals)

    grouped_data = dframeGen.groupby('speechiness_interval')['comedy'].mean()

    plt.bar(range(len(grouped_data)), grouped_data, tick_label=grouped_data.index)

    plt.xlabel('Speechiness Intervals')
    plt.ylabel('Proportion of Comedy Entries')
    plt.title('Proportion of Comedy Entries by Speechiness Intervals')

    plt.xticks(rotation=45)
    plt.show()


def secondPart(dframe, dframeGen):
    createCorrelationHeatmaps(dframe, dframeGen)
    createAnalysisEnergyLoudness(dframeGen)
    createAnalysisLivenessSpeechiness(dframeGen)
    createAnalysisDanceabilityTopGenre(dframeGen)
    createAnalysisTopGenreFilteredGenres(dframeGen)
    createAnalysisComedySpeechiness(dframeGen)
    return None


def thirdPartOvertrain(dframe, mode='overtrain'):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py a main.py (vid. ZDROJE KU KODOM)

    X = dframe.drop(columns=['happy', 'sad', 'calm', 'energetic'])
    y = dframe[['happy', 'sad', 'calm', 'energetic']]

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    if mode == 'early_stop':
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model = Sequential()
    model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])

    if mode == 'early_stop':
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    else:
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30)

    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)

    print("*" * 100, "Test and Train accuracy", "*" * 20)
    print(f"Test accuracy: {test_scores[1]:.4f}")
    print(f"Train accuracy: {train_scores[1]:.4f}")

    y_pred_test = model.predict(X_test)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    y_pred_train = model.predict(X_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    class_names = dframe[['happy', 'sad', 'calm', 'energetic']].columns.tolist()

    cm = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on test set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()

    cm = confusion_matrix(np.argmax(y_train.values, axis=1), y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on train set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()

    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    return None


def thirdPartLast(dframe, mode):
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py a main.py (vid. ZDROJE KU KODOM)
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    X = dframe.drop(columns=['happy', 'sad', 'calm', 'energetic'])
    y = dframe[['happy', 'sad', 'calm', 'energetic']]

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    if mode == 'prvy':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    if mode == 'druhy':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0002), metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    if mode == 'treti':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.00005), metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    if mode == 'stvrty':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    if mode == 'piaty':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=SGD(learning_rate=0.00005, momentum=0.9),
                      metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])
    if mode == 'siesty':
        # Train MLP model in Keras
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model = Sequential()
        model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(25, activation='relu'))
        model.add(Dense(4, activation='softmax'))

        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.000005), metrics=['accuracy'])
        history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=200, batch_size=30,
                            callbacks=[early_stopping])

    # Evaluate the model
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)

    print("*" * 100, "Test and Train accuracy", "*" * 20)
    print(f"Test accuracy: {test_scores[1]:.4f}")
    print(f"Train accuracy: {train_scores[1]:.4f}")

    # Plot confusion matrix
    y_pred_test = model.predict(X_test)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    y_pred_train = model.predict(X_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    class_names = dframe[['happy', 'sad', 'calm', 'energetic']].columns.tolist()

    cm = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on test set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()

    cm = confusion_matrix(np.argmax(y_train.values, axis=1), y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on train set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.show()

    # Plot loss and accuracy
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return None


def gridSearch(dframe):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)
    # Tato funkcia bola inspirovana zdrojovim kodom seminar2.py (vid. ZDROJE KU KODOM)

    X = dframe.drop(columns=['emotion'])
    y = dframe['emotion']

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=42)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    # Define parameter grid for grid search
    param_grid = {
        'hidden_layer_sizes': [
            (2,), (10,), (100,),  # Single hidden layer
            (10, 10), (100, 100),  # Two hidden layers
            (10, 10, 10), (100, 100, 100),  # Three hidden layers
        ],
        'max_iter': [50, 100, 200],  # Different max iterations
        'early_stopping': [True, False]
    }

    # Initialize the classifier
    clf = MLPClassifier(random_state=1)

    # Create GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid, cv=3, n_jobs=-1)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best parameters and estimator from grid search
    best_params = grid_search.best_params_
    best_clf = grid_search.best_estimator_
    results_df = pd.DataFrame(grid_search.cv_results_)
    worst_params = results_df.sort_values(by='mean_test_score', ascending=True).iloc[0]['params']

    worst_clf = MLPClassifier(random_state=1, early_stopping=worst_params['early_stopping'],
                              hidden_layer_sizes=worst_params['hidden_layer_sizes'][0],
                              max_iter=worst_params['max_iter'])

    # Train with best and worst parameters
    best_clf.fit(X_train, y_train)
    worst_clf.fit(X_train, y_train)

    # Evaluate the performance on the validation set
    worst_score = worst_clf.score(X_valid, y_valid)

    # Print the best parameters
    print("Best Parameters:", best_params)
    print("Best score:", grid_search.best_score_)
    print("Worst Parameters:", worst_params)
    print("Worst score:", worst_score)

    # Print Results and confusion matrix of results for train set
    y_pred_train = best_clf.predict(X_train)
    print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred_train))
    cm_train = confusion_matrix(y_train, y_pred_train)

    # Print Results and confusion matrix of results for test set
    y_pred_test = best_clf.predict(X_test)
    print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred_test))
    cm_test = confusion_matrix(y_test, y_pred_test)

    class_names = list(le.inverse_transform(best_clf.classes_))

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

    # Print Results and confusion matrix of results for train set
    y_pred_train = worst_clf.predict(X_train)
    print('MLP accuracy on train set: ', accuracy_score(y_train, y_pred_train))
    cm_train = confusion_matrix(y_train, y_pred_train)

    # Print Results and confusion matrix of results for test set
    y_pred_test = worst_clf.predict(X_test)
    print('MLP accuracy on test set: ', accuracy_score(y_test, y_pred_test))
    cm_test = confusion_matrix(y_test, y_pred_test)

    class_names = list(le.inverse_transform(worst_clf.classes_))

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

    return None


def reduceDataframe(dframe):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    bool_columns = dframe[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                           'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house',
                           'industrial', 'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk',
                           'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']].astype(bool)

    filtered_df = dframe[bool_columns.any(axis=1)]

    sampled_df = filtered_df.groupby(bool_columns.columns.tolist(), group_keys=False).apply(
        lambda x: x.sample(min(len(x), 200)))

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df


def balanceDataframe(dframe):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    bool_columns = dframe[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                           'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house',
                           'industrial', 'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk',
                           'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']].astype(bool)

    filtered_df = dframe[bool_columns.any(axis=1)]

    sampled_df = filtered_df.groupby(bool_columns.columns.tolist(), group_keys=False).apply(
        lambda x: x.sample(n=1000, replace=True) if len(x) < 1000 else x.sample(1000)
    )

    sampled_df = sampled_df.reset_index(drop=True)

    return sampled_df


def bonusThird(dframe):
    # Tato funkcia bola vypracovana a upravovana za pomoci ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)
    # Casti kodu SMOTE boli vypracovane pomocou ChatGPT a GithubCopilota (vid. ZDROJE KU KODOM)

    X = dframe.drop(columns=['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                             'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house',
                             'industrial', 'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk',
                             'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul'])
    y = dframe[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house',
                'industrial', 'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk',
                'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']]

    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=20)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=20)

    scaler = MinMaxScaler()

    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_valid = pd.DataFrame(X_valid, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)

    # Train MLP model in Keras
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model = Sequential()
    model.add(Dense(45, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))

    model.add(Dense(32, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0004),
                  metrics=['accuracy'])
    history = model.fit(x=X_train, y=y_train, validation_data=(X_valid, y_valid), epochs=600, batch_size=180,
                        callbacks=[early_stopping])

    # Evaluate the model
    test_scores = model.evaluate(X_test, y_test, verbose=0)
    train_scores = model.evaluate(X_train, y_train, verbose=0)

    print("*" * 100, "Test and Train accuracy", "*" * 20)
    print(f"Test accuracy: {test_scores[1]:.4f}")
    print(f"Train accuracy: {train_scores[1]:.4f}")

    # Plot confusion matrix
    y_pred_test = model.predict(X_test)
    y_pred_test = np.argmax(y_pred_test, axis=1)

    y_pred_train = model.predict(X_train)
    y_pred_train = np.argmax(y_pred_train, axis=1)

    class_names = dframe[['ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                          'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house',
                          'industrial', 'j-pop', 'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk',
                          'reggaeton', 'rock', 'rockabilly', 'ska', 'sleep', 'soul']].columns.tolist()

    cm = confusion_matrix(np.argmax(y_test.values, axis=1), y_pred_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on test set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.xticks(rotation=90)
    plt.show()

    cm = confusion_matrix(np.argmax(y_train.values, axis=1), y_pred_train)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax)
    disp.ax_.set_title("Confusion matrix on train set")
    disp.ax_.set(xlabel='Predicted', ylabel='True')
    plt.xticks(rotation=90)
    plt.show()

    # Plot loss and accuracy
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy/Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    return None


# Results---------------------------------------------------------------------------------------------------------------
df = handleOutliersAndMissingValues(df)
dfGen = handleOutliersAndMissingValues(dfGen, 1)
dfThird = handleOutliersAndMissingValues(dfThird)
df, le = encodeGenres(df)
dfGen = encodeGenres(dfGen, 1)
dfThird = encodeGenres(dfThird, 2)
dfBalanced = balanceDataframe(dfThird)
dfReduced = reduceDataframe(dfThird)
restOfFirstPart(df)
secondPart(df, dfGen)
thirdPartOvertrain(dfThird)
thirdPartOvertrain(dfThird, 'early_stop')
thirdPartLast(dfThird, 'stvrty')
thirdPartLast(dfThird, 'siesty')
gridSearch(df)
bonusThird(dfThird)
bonusThird(dfReduced)
bonusThird(dfBalanced)
