import pandas as pd
import matplotlib.pyplot as plt


def question1():
    comments = pd.read_csv('./data/Comments.csv')

    nb_rows = comments.shape[0]
    nb_cols = comments.shape[1]
    test = comments.columns

    print(f'Le nombre de lignes est {nb_rows}')
    print(f'Le nombre de colonnes est {nb_cols}')
    print(f'Les colonnes sont {test}')


def question2():
    comments = pd.read_csv('./data/Comments.csv')

    comment = comments[comments['postId'] == '192978590727638_722477749883613']

    print(f'Le nombre de commentaires est {comment.shape[0]}')


def question3():
    comments = pd.read_csv('./data/Comments.csv')

    comments['like_count'].hist(bins=comments['like_count'].max())

    plt.show()


def question4():
    comments = pd.read_csv('./data/Comments.csv')

    comments = comments[(comments['INSULT'] >= 0.6) & (comments['SEVERE_TOXICITY'] <= 0.5)]

    print(f'Le nombre de commentaires est {comments.shape[0]}')


def question5():
    comments = pd.read_csv('./data/Comments.csv')

    columns_with_missing_values = [col for col in comments.columns if comments[col].isna().any()]

    print(f'Les colonnes avec des valeurs manquantes sont {columns_with_missing_values}')


def question6():
    comments = pd.read_csv('./data/Comments.csv')

    comments_timestamps = pd.to_datetime(comments['created_time'])
    comments_posted_january_2023 = (comments_timestamps.dt.year == 2023) & (comments_timestamps.dt.month == 1)

    print(f'Le nombre de commentaires postés en janvier 2023 est {comments[comments_posted_january_2023].shape[0]}')