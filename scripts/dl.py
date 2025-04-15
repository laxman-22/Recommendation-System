import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random
from collections import defaultdict

# -------------------------------- Process Data -------------------------------- #
def process_data():
    # Config
    val_holdouts = 2
    test_holdouts = 3
    total_holdouts = val_holdouts + test_holdouts
    random.seed(42)
    np.random.seed(42)

    # Load Excel, skip the first column (num_rated)
    df = pd.read_excel('./data/jester-data-1.xls', header=None)
    df.index = [f'user_{i}' for i in range(len(df))]
    df.columns = ['total'] + [f'J{i}' for i in range(1, 101)]

    df = df.replace(99.0, np.nan)
    sub_df = df.iloc[:, [5, 7, 8, 13, 15, 16, 17, 18, 19, 20]]

    train_matrix = sub_df.copy()
    val_holdout_list = []
    test_holdout_list = []
    
    for user_id in train_matrix.index:
        user_ratings = train_matrix.loc[user_id]
        valid_jokes = user_ratings.dropna().index.tolist()

        if len(valid_jokes) <= total_holdouts + 1:
            continue

        selected_holdouts = np.random.choice(valid_jokes, size=total_holdouts, replace=False)
        val_jokes = selected_holdouts[:val_holdouts]
        test_jokes = selected_holdouts[val_holdouts:]

        for joke_id in val_jokes:
            val_holdout_list.append((user_id, joke_id, train_matrix.at[user_id, joke_id]))
            train_matrix.at[user_id, joke_id] = np.nan

        for joke_id in test_jokes:
            test_holdout_list.append((user_id, joke_id, train_matrix.at[user_id, joke_id]))
            train_matrix.at[user_id, joke_id] = np.nan

    return sub_df, train_matrix, val_holdout_list, test_holdout_list

# ---------------------------- Top-K Hits Accuracy --------------------------- #
def compute_top1_hit_accuracy(test_holdout_list, reconstructed_df):
    """
    Compute Top-1 Hit Accuracy:
    For each user, check if the item with the highest predicted score
    in their test holdouts is also the one with the highest true rating.
    """
    user2preds = defaultdict(list)
    for user_id, joke_id, true_rating in test_holdout_list:
        pred_rating = reconstructed_df.at[user_id, joke_id]
        user2preds[user_id].append((joke_id, true_rating, pred_rating))

    hits = 0
    total = 0

    for user_id, triples in user2preds.items():
        if len(triples) < 2:
            continue  # need at least two items to rank

        # Sort by predicted and true rating
        sorted_by_pred = sorted(triples, key=lambda x: x[2], reverse=True)
        sorted_by_true = sorted(triples, key=lambda x: x[1], reverse=True)

        best_pred_joke = sorted_by_pred[0][0]
        best_true_joke = sorted_by_true[0][0]

        if best_pred_joke == best_true_joke:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0

# -------------------------------- Model -------------------------------- #
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=50):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, latent_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z) * 10  # scale to [-10, 10]

def masked_mse_loss(preds, targets, mask):
    diff = (preds - targets) * mask
    return torch.sum(diff ** 2) / torch.sum(mask)

# -------------------------------- Training -------------------------------- #
def train_autoencoder(inputs, masks, latent_dim, val_holdout_list, num_epochs=100, user_means=None, train_matrix=None):
    model = Autoencoder(input_dim=inputs.shape[1], latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    train_losses, val_rmses = [], []

    for epoch in range(num_epochs):
        model.train()
        noisy_inputs = inputs
        preds = model(noisy_inputs)

        train_loss = masked_mse_loss(preds, inputs, masks)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_rmse = train_loss.sqrt().item()
        train_losses.append(train_rmse)

        # ---------------- Validation RMSE from val_holdout_list ---------------- #
        model.eval()
        with torch.no_grad():
            val_preds = model(inputs).numpy()
        val_reconstructed = val_preds + user_means.values[:, np.newaxis]
        val_df = pd.DataFrame(val_reconstructed, index=train_matrix.index, columns=train_matrix.columns)

        val_errors = []
        for user_id, joke_id, true_rating in val_holdout_list:
            pred = val_df.at[user_id, joke_id]
            val_errors.append((pred - true_rating) ** 2)

        if val_errors:
            val_rmse = np.sqrt(np.mean(val_errors))
        else:
            val_rmse = float('nan')

        val_rmses.append(val_rmse)

        if epoch % 10 == 0:
            print(f"[Latent {latent_dim}] Epoch {epoch}, Train RMSE: {train_rmse:.4f}, Val RMSE: {val_rmse:.4f}")

    return model, train_losses, val_rmses

# -------------------------------- Main Script -------------------------------- #
def main():
    sub_df, train_matrix, val_holdout_list, test_holdout_list = process_data()
    mask = ~train_matrix.isna()
    user_means = train_matrix.mean(axis=1)
    normalized_data = train_matrix.sub(user_means, axis=0)
    input_data = normalized_data.fillna(0).values.astype(np.float32)
    inputs = torch.tensor(input_data)
    masks = torch.tensor(mask.values.astype(np.float32))

    model, train_losses, val_rmses = train_autoencoder(inputs, masks, latent_dim=50, val_holdout_list=val_holdout_list, num_epochs=300, user_means=user_means, train_matrix=train_matrix)

    plt.plot(train_losses, label="Train RMSE")
    plt.plot(val_rmses, label="Val RMSE")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title("Training vs Validation")
    plt.legend()
    plt.grid(True)
    plt.show()

    model.eval()
    with torch.no_grad():
        reconstructed = model(inputs).numpy()
    reconstructed += user_means.values[:, np.newaxis]
    reconstructed_df = pd.DataFrame(reconstructed, index=train_matrix.index, columns=train_matrix.columns)

    errors = []
    for user_id, joke_id, true_rating in test_holdout_list:
        pred = reconstructed_df.at[user_id, joke_id]
        errors.append((user_id, joke_id, true_rating, pred, (pred - true_rating) ** 2))

    error_df = pd.DataFrame(errors, columns=["user_id", "joke_id", "true_rating", "predicted", "squared_error"])
    error_df["abs_error"] = (error_df["true_rating"] - error_df["predicted"]).abs()
    test_rmse = np.sqrt(error_df["squared_error"].mean())

    print(f"\n📊 Test RMSE: {test_rmse:.4f}")
    print("\n✅ Best Predictions:")
    print(error_df.sort_values(by="abs_error"))

    print("\n❌ Worst Predictions:")
    print(error_df.sort_values(by="abs_error", ascending=False))

    joke_errors = error_df.groupby("joke_id")["squared_error"].mean().apply(np.sqrt).sort_values()
    print("\n📊 Joke-wise RMSE:")
    print(joke_errors)

    error_df.sort_values(by="abs_error").to_csv("predicted_errors.csv", index=False)

    top1_acc = compute_top1_hit_accuracy(test_holdout_list, reconstructed_df)
    print(f"🎯 Top-1 Hit Accuracy: {top1_acc:.4f}")

if __name__ == "__main__":
    main()
