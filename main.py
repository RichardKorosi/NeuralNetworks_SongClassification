import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
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
dfGen = pd.read_csv('./data/zadanie1_dataset.csv')


# Functions ------------------------------------------------------------------------------------------------------------
def handleOutliersAndMissingValues(dframe, mode=0):
    # Handle outliers (0,5b)
    # -----------------------------------------------------------------------------------------------

    # Print min and max values of columns before removing outliers
    print("*" * 100, "Before removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    dframe = dframe[(dframe['danceability'] >= 0) & (dframe['danceability'] <= 1)]  # Some values were higher than 1
    dframe = dframe[(dframe['loudness'] >= -60) & (
            dframe['loudness'] <= 5)]  # Range should be <-60, 0>, one value is slightly above 0
    dframe = dframe[(dframe['tempo'] > 0)]  # Some outliers found with 0 tempo
    dframe = dframe[(dframe['duration_ms'] > 20000) & (
            dframe['duration_ms'] <= 1967400)]  # Some outliers found (last one is long but valid)

    # Print min and max values of columns
    print("*" * 100, "After removing outliers", "*" * 100)
    print("-" * 10, "Min", "-" * 10)
    print(dframe.min(numeric_only=True))
    print("-" * 10, "Max", "-" * 10)
    print(dframe.max(numeric_only=True))

    # Handle missing values (0,5b)
    # -----------------------------------------------------------------------------------------

    # Count missing values in columns
    print("*" * 100, "Missing values", "*" * 100)
    print(f"Length of dataset: {len(dframe)}")
    print(dframe.isnull().sum())

    # Deal with missing values and columns with no use
    if mode == 0:
        dframe = dframe.dropna(
            subset=['top_genre', 'popularity', 'number_of_artists'])  # Drop rows with missing values in certain columns
        dframe = dframe.drop(['name', 'url', 'genres', 'filtered_genres'], axis=1)  # Drop columns with no use
    else:
        dframe = dframe.dropna(
            subset=['top_genre', 'popularity', 'number_of_artists'])  # Drop rows with missing values in certain columns
        dframe = dframe.drop(['name', 'url'], axis=1)  # Drop columns with no use

    print("*" * 100, "Missing values after removing them", "*" * 100)
    print(f"Length of dataset: {len(dframe)}")
    print(dframe.isnull().sum())

    return dframe


def encodeGenres(dframe, mode=0):
    if mode == 0:
        # Column types and encoding (0,5b)
        # ------------------------------------------------------------------------------------ Print column types
        print("*" * 100, "Column types", "*" * 100)
        print(dframe.dtypes)

        # Use dummy encoding for top_genre
        dframe = pd.get_dummies(dframe, columns=['top_genre'], prefix='', prefix_sep='')

        # Use Label encoding for emotion
        le = LabelEncoder()
        dframe['emotion'] = le.fit_transform(df['emotion'])

        print("*" * 100, "Column types", "*" * 100)

        # Change boolean columns to float (False = 0, True = 1)
        for col in ['explicit', 'ambient', 'anime', 'bluegrass', 'blues', 'classical', 'comedy', 'country', 'dancehall',
                    'disco', 'edm', 'emo', 'folk', 'forro', 'funk', 'grunge', 'hardcore', 'house', 'industrial',
                    'j-pop',
                    'j-rock', 'jazz', 'metal', 'metalcore', 'opera', 'pop', 'punk', 'reggaeton', 'rock', 'rockabilly',
                    'ska',
                    'sleep', 'soul']:
            dframe[col] = dframe[col].astype(float)
        print(dframe.dtypes)

        return dframe, le
    else:
        le = LabelEncoder()
        dframe['emotion_le'] = le.fit_transform(df['emotion'])

        import ast

        # Convert the genres column from string to list
        dframe['filtered_genres'] = dframe['filtered_genres'].apply(ast.literal_eval)

        # Perform one-hot encoding on the genres column
        genres_encoded = dframe['filtered_genres'].apply(lambda x: pd.Series([1] * len(x), index=x)).fillna(0)

        # Concatenate the encoded genres with the original dataframe
        dframe = pd.concat([dframe, genres_encoded], axis=1)

        # NUMBER OF ARTISTS NUMBER OF GENRES
        dframe['number_of_genres'] = dframe['genres'].str.split(',').str.len()
        dframe['number_of_genres'] = dframe['number_of_genres'].fillna(0)
        dframe['number_of_genres'] = dframe['number_of_genres'].astype(int)

        return dframe


def createHistogramsPartOne(X_train, time):
    attributes = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                  'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists']

    X_train[attributes].hist(bins=70, figsize=(15, 15))

    # Loop through each subplot and set labels
    for ax in plt.gcf().get_axes():
        ax.set_xlabel('Hodnota')  # Set x-axis label to "hodnota"
        ax.set_ylabel('Počet')  # Set y-axis label to "pocet"
    if time == "before":
        plt.suptitle('Histograms before scaling/standardizing')
    else:
        plt.suptitle('Histograms after scaling/standardizing')
    plt.show()

    return None


def createPiechartsPartOne(X_train):
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
    # Split dataset into X and y (input and output) (1b)
    # -------------------------------------------------------------------

    # Split dataset into X and y
    X = dframe.drop(columns=['emotion'])
    y = dframe['emotion']

    # Split dataset into train, valid and test
    X_train, X_valid_test, y_train, y_valid_test = train_test_split(X, y, shuffle=True, test_size=0.2, random_state=42)
    X_valid, X_test, y_valid, y_test = train_test_split(X_valid_test, y_valid_test, shuffle=True, test_size=0.5,
                                                        random_state=42)

    # Scale and standardize data (0,5b)
    # ------------------------------------------------------------------------------------

    # Print dataset shapes
    print("*" * 100, "Dataset shapes", "*" * 100)
    print(f"Full dataset: {dframe.shape}")
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

    createHistogramsPartOne(X_train, "before")
    createPiechartsPartOne(X_train)

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

    createHistogramsPartOne(X_train, "after")

    # Train MLP model to predict emotion (1b)
    # ------------------------------------------------------------------------------

    print("*" * 100, "MLP", "*" * 100)
    print(f"Random accuracy: {1 / len(y_train.unique())}")

    # 10                0.8639  0.85069
    # 100               0.8777  0.86545
    # 500               0.8792  0.86631
    # 150 150           0.8845  0.86197
    # 150 100 100       0.9035  0.86805
    # 1000 300 200      0.9045  0.86892
    # 200 150 100       0.88978 0.87152

    clf = MLPClassifier(
        hidden_layer_sizes=(200, 150, 100),
        random_state=1,
        max_iter=250,
        early_stopping=True
    ).fit(X_train, y_train)

    # Print Results and confusion matrix of results (1b)
    # -------------------------------------------------------------------

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


def createCorrelationHeatmaps(dframe, dframeGen):
    # DFRAME1 ----------------------------------------------------------------------------------------------------------
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

    # DFRAME2 ----------------------------------------------------------------------------------------------------------
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
         'valence', 'tempo', 'duration_ms', 'popularity', 'number_of_artists', 'number_of_genres', 'explicit'], axis=1)
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
    average_speechiness = dframeGen.groupby('liveness_interval')['speechiness'].mean()

    plt.figure(figsize=(10, 6))
    plt.plot(average_speechiness.index.categories.mid, average_speechiness.values, marker='o', color='skyblue')
    plt.title('Average speechiness vs. liveness')
    plt.xlabel('liveness')
    plt.ylabel('Average speechiness')
    plt.grid(True)
    plt.show()

    return None


def createAnalysisTempoTopGenre(dframeGen):
    plt.figure(figsize=(10, 6))
    plt.scatter(dframeGen['tempo'], dframeGen['top_genre'], color='skyblue')
    plt.title('Scatter Plot')
    plt.xlabel('tempo')
    plt.ylabel('top_genre')
    plt.show()

    avg_bpm_per_genre = dframeGen.groupby('top_genre')['tempo'].mean().reset_index()
    avg_bpm_per_genre = avg_bpm_per_genre.sort_values(by='tempo', ascending=False)

    plt.figure(figsize=(16, 10))
    plt.bar(avg_bpm_per_genre['top_genre'], avg_bpm_per_genre['tempo'], color='skyblue')
    plt.title('Average BPM per Genre')
    plt.xlabel('Genre')
    plt.ylabel('Average BPM')
    plt.xticks(rotation=45)
    plt.show()

    # Calculate means for each genre
    means = [dframeGen['tempo'][dframeGen['top_genre'] == genre].mean() for genre in dframeGen['top_genre'].unique()]
    # Create a dictionary to store genres and their means
    genre_means = dict(zip(dframeGen['top_genre'].unique(), means))
    # Sort genres based on mean values
    sorted_genres = sorted(dframeGen['top_genre'].unique(), key=lambda genre: genre_means[genre])
    # Create a boxplot
    plt.figure(figsize=(15, 10))
    plt.title('Tempo Distribution by Genre')
    plt.xlabel('Genre')
    plt.ylabel('Tempo (BPM)')
    # Create the boxplot using the sorted genres
    plt.boxplot([dframeGen['tempo'][dframeGen['top_genre'] == genre] for genre in sorted_genres],
                labels=sorted_genres)
    # Add means as red dots
    sorted_means = [genre_means[genre] for genre in sorted_genres]
    plt.plot(range(1, len(sorted_genres) + 1), sorted_means, 'rx')
    # Show the plot
    plt.xticks(rotation=45)
    plt.show()

    return None


def generate_three_pie_charts(ax, labels, ratios, title):
    ax.pie(ratios, labels=labels, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title(title)


def createAnalysisTopGenreFilteredGenres(dframeGen):
    dframeGenEdit = dframeGen[['forro', 'sertanejo', 'j-pop', 'anime', 'funk', 'soul']]
    correlation_matrix = dframeGenEdit.corr()
    plt.figure(figsize=(12, 11))
    plt.imshow(correlation_matrix, cmap='viridis', interpolation='none')
    plt.colorbar()
    plt.title('Correlation Heatmap Selected Genres')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.show()
    # Filter rows where only EDM is True
    funk_only = dframeGen[(dframeGen['funk'] == True) & (dframeGen['soul'] == False)]
    # Filter rows where only soul is True
    soul_only = dframeGen[(dframeGen['funk'] == False) & (dframeGen['soul'] == True)]
    # Filter rows where both funk and soul are True
    funk_and_soul = dframeGen[(dframeGen['funk'] == True) & (dframeGen['soul'] == True)]
    # Calculate ratios
    funk_only_ratio = len(funk_only) / len(dframeGen)
    soul_only_ratio = len(soul_only) / len(dframeGen)
    funk_and_soul_ratio = len(funk_and_soul) / len(dframeGen)
    # Filter rows where only forro is True
    forro_only = dframeGen[(dframeGen['forro'] == True) & (dframeGen['sertanejo'] == False)]
    # Filter rows where only sertanejo is True
    sertanejo_only = dframeGen[(dframeGen['forro'] == False) & (dframeGen['sertanejo'] == True)]
    # Filter rows where both forro and sertanejo are True
    forro_and_sertanejo = dframeGen[(dframeGen['forro'] == True) & (dframeGen['sertanejo'] == True)]
    # Calculate ratios
    forro_only_ratio = len(forro_only) / len(dframeGen)
    sertanejo_only_ratio = len(sertanejo_only) / len(dframeGen)
    forro_and_sertanejo_ratio = len(forro_and_sertanejo) / len(dframeGen)
    # Filter rows where only j-pop is True
    jpop_only = dframeGen[(dframeGen['j-pop'] == True) & (dframeGen['anime'] == False)]
    # Filter rows where only anime is True
    anime_only = dframeGen[(dframeGen['j-pop'] == False) & (dframeGen['anime'] == True)]
    # Filter rows where both jpop and anime are True
    jpop_and_anime = dframeGen[(dframeGen['j-pop'] == True) & (dframeGen['anime'] == True)]
    # Calculate ratios
    jpop_only_ratio = len(jpop_only) / len(dframeGen)
    anime_only_ratio = len(anime_only) / len(dframeGen)
    jpop_and_anime_ratio = len(jpop_and_anime) / len(dframeGen)
    # Create a figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    # Data for funk and soul
    labels1 = ['Funk without Soul', 'Soul without Funk', 'Funk and Soul']
    ratios1 = [funk_only_ratio, soul_only_ratio, funk_and_soul_ratio]
    # Data for Forro and Sertanejo
    labels2 = ['Forro without Sertanejo', 'Sertanejo without Forro', 'Forro and Sertanejo']
    ratios2 = [forro_only_ratio, sertanejo_only_ratio, forro_and_sertanejo_ratio]
    # Data for J-Pop and Anime
    labels3 = ['J-pop without Anime', 'Anime without J-pop', 'J-pop and Anime']
    ratios3 = [jpop_only_ratio, anime_only_ratio, jpop_and_anime_ratio]
    # Generate pie charts
    generate_three_pie_charts(axes[0], labels1, ratios1, 'Distribution of Funk and Soul')
    generate_three_pie_charts(axes[1], labels2, ratios2, 'Distribution of Forro and Sertanejo')
    generate_three_pie_charts(axes[2], labels3, ratios3, 'Distribution of J-Pop and Anime')
    # Adjust layout
    plt.tight_layout()
    # Show the combined plot
    plt.show()

    return None


def createAnalysisComedySpeechiness(dframeGen):
    plt.scatter(dframeGen['comedy'], dframeGen['speechiness'])
    plt.title("Correlation between Comedy and Speechiness")
    plt.xlabel("Comedy")
    plt.ylabel("Speechiness")
    plt.show()

    num_intervals = 20
    plt.figure(figsize=(12, 11))
    # Create intervals for speechiness
    dframeGen['speechiness_interval'] = pd.cut(dframeGen['speechiness'], num_intervals)

    # Calculate the proportion of comedy entries in each interval
    grouped_data = dframeGen.groupby('speechiness_interval')['comedy'].mean()

    # Plot a bar chart
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
    createAnalysisTempoTopGenre(dframeGen)
    createAnalysisTopGenreFilteredGenres(dframeGen)
    createAnalysisComedySpeechiness(dframeGen)
    return None


# ----------------------------------------------------------------------------------------------------------------------
df = handleOutliersAndMissingValues(df)
dfGen = handleOutliersAndMissingValues(dfGen, 1)
df, le = encodeGenres(df)
dfGen = encodeGenres(dfGen, 1)
restOfFirstPart(df)
df.to_csv('./data/zadanie1_top_genre.csv', index=False)
dfGen.to_csv('./data/zadanie1_all_genres.csv', index=False)
secondPart(df, dfGen)
