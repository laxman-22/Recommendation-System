from preprocess import process_data
import numpy as np
import pandas as pd
from collections import defaultdict

def train(train_matrix,val_holdout_list, checkpoint_interval=1000, num_epochs=10000, num_features=10, learning_rate=0.01, regularization=0.02):
    """
    Train a matrix factorization model using stochastic gradient descent (SGD) on the training matrix.
    Parameters:
    - train_matrix: DataFrame containing the training data.
    - val_holdout_list: List of tuples (user_id, item_id, true_rating) for validation.
    - checkpoint_interval: Interval for saving model checkpoints.
    - num_epochs: Number of epochs for training.
    - num_features: Number of latent features.
    - learning_rate: Learning rate for SGD.
    - regularization: Regularization parameter.

    Returns:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - final_training_loss: Final training loss.
    - final_val_loss: Final validation loss.
    - final_val_ndcg: Final validation NDCG score.
    - row_means: Row means for normalization.
    - row_stds: Row standard deviations for normalization.
    """
    
    # Convert DataFrame to numpy array
    train_matrix_converted = train_matrix.to_numpy()

    # Z-score normalization

    # Row-wise mean and std, ignoring NaNs
    row_means = np.nanmean(train_matrix_converted, axis=1, keepdims=True)
    row_stds  = np.nanstd(train_matrix_converted, axis=1, keepdims=True)

    # Z-score transform per row
    normalized_train_matrix = (train_matrix_converted - row_means) / (row_stds+1e-8)

    # Set hyperparameters
    num_users, num_items = train_matrix.shape
    
    # Initialize user and item latent feature matrices

    user_features = np.random.normal(scale=1./np.sqrt(num_features), size=(num_users, num_features))
    item_features = np.random.normal(scale=1./np.sqrt(num_features), size=(num_items, num_features))

    # Training loop
    for epoch in range(num_epochs):
        for user_id in range(num_users):
            for item_id in range(num_items):
                if not np.isnan(normalized_train_matrix[user_id][item_id]):
                    # Calculate the prediction error
                    prediction = np.dot(user_features[user_id], item_features[item_id])
                    error = normalized_train_matrix[user_id][item_id] - prediction

                    # Update user and item features
                    user_features[user_id] += learning_rate * (error * item_features[item_id] - regularization * user_features[user_id])
                    item_features[item_id] += learning_rate * (error * user_features[user_id] - regularization * item_features[item_id])

        if epoch > 0 and epoch % checkpoint_interval == 0:
            # Save user_features and item_features as NumPy arrays
            checkpoint_path = f"checkpoint_epoch_{epoch}.npz"
            np.savez(checkpoint_path,
                    user_features=user_features,
                    item_features=item_features)
            print(f"Checkpoint saved: {checkpoint_path}")

        # Print loss every 100 epochs
        if epoch % 100 == 0:
            reconstruction_loss = np.nanmean((normalized_train_matrix - np.dot(user_features, item_features.T)) ** 2)
            regularization_loss = regularization * (np.nansum(user_features**2) + np.nansum(item_features**2))/ np.sum(~np.isnan(normalized_train_matrix))
            training_loss = reconstruction_loss + regularization_loss


            val_loss = cal_l2_loss(user_features, item_features, val_holdout_list, train_matrix, row_means, row_stds)
            val_ndcg = cal_ndcg(user_features, item_features, val_holdout_list, train_matrix,1,row_means)
            print(f'Epoch {epoch}, Training Loss: {training_loss}', 
                  f'Validation Loss: {val_loss}',
                  f"Validation NDCG: {val_ndcg}")


    # Final loss
    final_rec_loss = (np.nanmean((normalized_train_matrix - np.dot(user_features, item_features.T)) ** 2))
    final_reg_loss = regularization * (np.nansum(user_features**2) + np.nansum(item_features**2))/ np.sum(~np.isnan(normalized_train_matrix))
    final_training_loss = final_rec_loss + final_reg_loss
    final_val_loss = cal_l2_loss(user_features, item_features, val_holdout_list, train_matrix, row_means, row_stds)
    final_val_ndcg = cal_ndcg(user_features, item_features, val_holdout_list, train_matrix,1,row_means)

    return user_features, item_features, final_training_loss, final_val_loss, final_val_ndcg, row_means, row_stds

def cal_l2_loss(user_features, item_features, holdout_list, original_matrix, row_means=None, row_stds=None):
    """
    Calculate the L2 loss for the holdout set.
    Parameters:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - holdout_list: List of tuples (user_id, item_id, true_rating) for evaluation.
    - original_matrix: The original DataFrame containing the ratings.
    - row_means: Row means for normalization.
    - row_stds: Row standard deviations for normalization.
    Returns:
    - mse: Mean Squared Error for the holdout set.
    """

    # reconstruct the matrix to DataFrame format
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    user2idx = {uid: idx for idx, uid in enumerate(original_matrix.index)}

    mse_list = []
    for user_id, joke_id, true_rating in holdout_list:
        user_idx = user2idx[user_id]

        # If the user never had a min or max, it might be all NaN. 
        # Typically, you'd skip or handle carefully. 
        mean_val = row_means[user_idx, 0]
        std_val = row_stds[user_idx, 0]

        # Get predicted rating from reconstructed matrix:
        predicted_rating = reconstructed_df.at[user_id, joke_id]

        # Denormalize the true rating
        denormalized_prediction = predicted_rating * (std_val + 1e-8) + mean_val        

        # Compute error in normalized space
        error = denormalized_prediction - true_rating
    
        mse_list.append((error**2))

    mse = np.mean(mse_list)

   
    return mse



def cal_ndcg(user_features, item_features, holdout_list, original_matrix, k=3,row_means=None):
    """
    Calculate the average NDCG@k for a recommendation system
    using matrix factorization.

    Parameters:
    - user_features: User latent features matrix.
    - item_features: Item latent features matrix.
    - holdout_list: List of tuples (user_id, item_id, true_rating) for evaluation.
    - original_matrix: The original DataFrame containing the ratings.
    - k: The rank at which to compute NDCG (default is 3).
    - row_means: Row means for normalization.
    Returns:
    - average_ndcg: The average NDCG@k score across all users.

    """

    # 1) Reconstruct the matrix: U * V^T
    reconstructed_matrix = np.dot(user_features, item_features.T)

    # 2) Convert to DataFrame (same index & columns as original)
    reconstructed_df = pd.DataFrame(
        reconstructed_matrix,
        index=original_matrix.index,
        columns=original_matrix.columns
    )

    # ---------- Helper functions for DCG/NDCG ----------
    def dcg_at_k(relevances, k):
        """
        Compute Discounted Cumulative Gain (DCG) at rank k.
        relevances: array-like of ground truth relevance scores in ranked order.
        """
        relevances = np.array(relevances, dtype=float)[:k]
        if relevances.size:
            discounts = np.log2(np.arange(2, relevances.size + 2))
            return np.sum(relevances / discounts)
        return 0.0

    def ndcg_at_k(ranked_relevances, k):
        """
        Compute Normalized DCG at rank k, given the ground-truth relevances
        in the order of predicted ranking.
        """
        dcg = dcg_at_k(ranked_relevances, k)
        # Sort relevances in descending order to get ideal DCG
        ideal_relevances = np.sort(ranked_relevances)[::-1]
        idcg = dcg_at_k(ideal_relevances, k)
        if idcg == 0.0:
            return 0.0
        return dcg / idcg
    # ---------------------------------------------------

    # 3) Group holdout items by user: user_id -> [(item_id, true_rating), ...]
    user2holdouts = defaultdict(list)
    for user_id, item_id, true_rating in holdout_list:
        user2holdouts[user_id].append((item_id, true_rating))

    user2idx = {uid: idx for idx, uid in enumerate(original_matrix.index)}

    ndcg_scores = []

    # 4) For each user, rank ONLY their holdout items by predicted score
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

        # Normalize the true ratings for this user and make sure all ratings are positive
        user_idx = user2idx[user_id]
        normed_true_ratings = [(r - row_means[user_idx,0])+20 for r in holdout_true_ratings]

        # Sort the holdout items by predicted score (descending)
        # We'll get the indices that sort predicted_scores in descending order
        sorted_indices = np.argsort(predicted_scores)[::-1]
        
        # Re-order the true ratings based on the predicted order
        ranked_relevances = [normed_true_ratings[i] for i in sorted_indices]

        # 4. Compute NDCG@k for this user's holdout-based ranking
        user_ndcg = ndcg_at_k(ranked_relevances, k)
        ndcg_scores.append(user_ndcg)

    # 5) Compute the average NDCG across all users
    average_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return average_ndcg

def main():
    # Load and process data
    original_matrix, train_matrix, val_holdout_list, test_holdout_list = process_data()

    # Train the model
    user_features, item_features, train_mse, val_mse, final_val_ndcg, row_means, row_stds = train(train_matrix,val_holdout_list)

    calculated_ndcg = cal_ndcg(user_features, item_features, test_holdout_list, original_matrix,3,row_means)

    print(f'Train MSE: {train_mse}')
    print(f'Val MSE: {val_mse}')
    print(f'Val NDCG: {final_val_ndcg}')
    print(f'Test NDCG: {calculated_ndcg}')

if __name__ == "__main__":
    main()
