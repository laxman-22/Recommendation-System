import pandas as pd
import numpy as np
import random


def process_data():
    """
    Preprocess the Jester dataset for matrix factorization.
    """
    
    # Config
    val_holdouts = 2
    test_holdouts = 3
    total_holdouts = val_holdouts + test_holdouts
    random.seed(42)
    np.random.seed(42)

    # Load Excel, skip the first column (num_rated)
    df = pd.read_excel('data/jester-data-1.xls', header=None)
    df.index = [f'user_{i}' for i in range(len(df))]
    df.columns = ['total'] + [f'J{i}' for i in range(1, 101)]

    # Adjust for zero indexing
    df = df.replace(99.0, np.nan)
    sub_df = df.iloc[:, [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]]

    # Prepare training matrix and holdouts
    train_matrix = sub_df.copy()
    val_holdout_list = []
    test_holdout_list = []
    
    for user_id in train_matrix.index:
        user_ratings = train_matrix.loc[user_id]

        # Valid jokes are the ones this user rated (not NaN)
        valid_jokes = user_ratings.dropna().index.tolist()

        # Ensure enough ratings to hold out
        if len(valid_jokes) <= total_holdouts + 4:
            continue

        # Randomly select total_holdouts
        selected_holdouts = np.random.choice(valid_jokes, size=total_holdouts, replace=False)
        
        # Split into val and test sets
        val_jokes = selected_holdouts[:val_holdouts]
        test_jokes = selected_holdouts[val_holdouts:]

        for joke_id in val_jokes:
            true_rating = train_matrix.at[user_id, joke_id]
            val_holdout_list.append((user_id, joke_id, true_rating))
            train_matrix.at[user_id, joke_id] = np.nan  # Mask rating

        for joke_id in test_jokes:
            true_rating = train_matrix.at[user_id, joke_id]
            test_holdout_list.append((user_id, joke_id, true_rating))
            train_matrix.at[user_id, joke_id] = np.nan  # Mask rating
    
    # Output: train_matrix, val_holdout_list, test_holdout_list
    return sub_df, train_matrix, val_holdout_list, test_holdout_list
