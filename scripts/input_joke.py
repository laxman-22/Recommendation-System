import pandas as pd

def get_best_joke_for_user(user_id):
    """
    Given a CSV of predictions with user_id as index and a target user_id (e.g., 'user_3'),
    return the joke_id with the highest predicted rating for that user.
    """
    df = pd.read_csv('./data/predicted_errors.csv', index_col=0)

    if user_id not in df.index:
        print(f"User '{user_id}' not found in predictions.")
        return None

    # Filter rows for the user
    user_df = df.loc[[user_id]]

    # Get the joke_id with the highest predicted rating
    best_row = user_df.sort_values(by="predicted", ascending=False).iloc[0]
    return best_row["joke_id"]

joke = get_best_joke_for_user("user_4")
