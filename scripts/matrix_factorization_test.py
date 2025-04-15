import numpy as np
from preprocess import process_data
import pandas as pd
from collections import defaultdict

def accuracy(user_features, item_features, holdout_list, original_matrix):
    """
    Calculate the accuracy of the top-1 joke recommendation for the test set.
    The function reconstructs the rating matrix using the user and item features,
    and then ranks the holdout items for each user based on the predicted scores.
    The accuracy is defined as the proportion of users for whom the top-1 predicted joke
    matches the true rating.

    Parameters:
    user_features (numpy.ndarray): User feature matrix.
    item_features (numpy.ndarray): Item feature matrix.
    holdout_list (list): List of tuples containing user_id, item_id, and true_rating for holdout items.
    original_matrix (pandas.DataFrame): The original rating matrix.

    Returns:
    float: The accuracy of the top-1 joke recommendation.
    """

    # 1) Reconstruct the matrix: U * V^T
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # 2) Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    # 3) Group holdout items by user: user_id -> [(item_id, true_rating), ...]
    user2holdouts = defaultdict(list)
    for user_id, item_id, true_rating in holdout_list:
        user2holdouts[user_id].append((item_id, true_rating))

    # 4) For each user, rank ONLY their holdout items by predicted score
    
    correct_items = 0
    for user_id, holdouts in user2holdouts.items():
        if not holdouts:
            continue
        
        # The list of holdout items for this user
        holdout_items = [t[0] for t in holdouts]
        # The true ratings for those items
        holdout_true_ratings = [t[1] for t in holdouts]
        
        # Get the predicted scores for these holdout items only
        predicted_scores = []
        for item in holdout_items:
            pred_score = reconstructed_df.at[user_id, item]
            predicted_scores.append(pred_score)

        if np.argmax(predicted_scores) == np.argmax(holdout_true_ratings):
            correct_items += 1

    return correct_items / len(user2holdouts)

def test():
    """
    Test the matrix factorization model with loaded user and item features
    """
    
    # load user_features, item_features
    checkpoint = np.load('traditional_ml_model.npz')

    # Extract user and item features from the checkpoint
    user_features = checkpoint['user_features']
    item_features = checkpoint['item_features']

    # Process the data
    original_matrix, train_matrix, val_holdout_list, test_holdout_list = process_data()
    calculated_accuracy= accuracy(user_features, item_features, test_holdout_list, original_matrix)

    print(f'Test accuracy: {calculated_accuracy}')

if __name__ == "__main__":
    test()
