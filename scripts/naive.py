from preprocess import process_data
import pandas as pd
import numpy as np

def mean_model():
    # Load data
    original_df, train_df, val, holdout_list = process_data()
    # Compute mean ratings for each joke
    joke_means = train_df.mean(skipna=True)
    return joke_means, train_df, holdout_list

def evaluate(joke_means, holdout_list):
    from collections import defaultdict

    # Group holdouts by user
    user_to_holdouts = defaultdict(list)
    for user_id, joke_id, true_rating in holdout_list:
        user_to_holdouts[user_id].append((joke_id, true_rating))

    match_count = 0
    total_users = 0

    for user_id, holdouts in user_to_holdouts.items():
        if len(holdouts) < 3:
            continue  # Skip if not enough holdouts

        # Extract joke_ids and ratings
        joke_ids = [jid for jid, _ in holdouts]
        raw_ratings = [rating for _, rating in holdouts]

        # Normalize true ratings
        normalized_relevance = [(r + 10) for r in raw_ratings]

        # Get predicted mean scores
        raw_predictions = [joke_means.get(jid, 0) for jid in joke_ids]
        normalized_predictions = [(p + 10) for p in raw_predictions]

        # Get index of top joke in both
        top_true_idx = np.argmax(normalized_relevance)
        top_pred_idx = np.argmax(normalized_predictions)

        if top_true_idx == top_pred_idx:
            match_count += 1
        total_users += 1

    accuracy = match_count / total_users if total_users > 0 else 0
    return accuracy


def recommend_top_joke(user):  # For demo, pass in array of user ratings.
    joke_means, data, _ = mean_model()
    user_ratings = list(data.loc[user])
    joke_columns = ['J5', 'J7', 'J8', 'J13', 'J15', 'J16', 'J17', 'J18', 'J19', 'J20']

    unrated_indices = [i for i, r in enumerate(user_ratings) if pd.isna(r)]
    
    if len(unrated_indices) < 1:
        raise ValueError("At least 1 NaN (unrated joke) is required to make a recommendation.")
    
    # Gather predictions for unrated jokes using mean model
    preds = [(joke_columns[i], joke_means.get(joke_columns[i], 0)) for i in unrated_indices]
    
    # Sort by predicted mean rating, descending
    sorted_preds = sorted(preds, key=lambda x: x[1], reverse=True)
    
    # Return the top joke column name
    return sorted_preds[0][0]  # Just the best one

if __name__ == "__main__":
    model_data, _, holdouts = mean_model()
    ndcg_scores = evaluate(model_data, holdouts)

    print(f"Accuracy: {np.mean(ndcg_scores):.3f}")
    print(recommend_top_joke('user_1'))  # Example user for recommendation
