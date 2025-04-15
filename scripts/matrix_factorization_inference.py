import pandas as pd
import numpy as np
from preprocess import process_data

def inference(user_index):
    """
    Inference function to recommend top-1 joke for a given user using matrix factorization.
    It loads the user and item features from a pre-trained model, reconstructs the rating matrix,
    and predicts the top-1 joke for the specified user.     
    Parameters:
    user_index (int): The index of the user for whom to recommend a joke.
    Returns:
    str: The ID of the top-1 recommended joke for the user.   
    """
    
    user_features = np.load('traditional_ml_model.npz')['user_features']
    item_features = np.load('traditional_ml_model.npz')['item_features']
    original_matrix, _, _, test_holdout_list = process_data()

    # Reconstruct the matrix using user and item features
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    predicted_scores = {}

    for user_id, joke_id, true_rating in test_holdout_list:
        if f"user_{user_index}" == user_id:
            predicted_score = reconstructed_df.at[user_id, joke_id]
            predicted_scores[joke_id] = predicted_score

    sorted_jokes = [key for key, value in sorted(predicted_scores.items(), key=lambda x: x[1], reverse=True)]
    top_1_jokes = sorted_jokes[0]

    print(f"Top 1 joke for user_{user_index}: {top_1_jokes}")
    return top_1_jokes

    
if __name__ == "__main__":
    inference(1)



