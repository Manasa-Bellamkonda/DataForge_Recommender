import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class UserKNN:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix = None
        self.user_similarity = None
        self.user_ids = None
        self.user_means = None
        self.global_mean = None

    def fit(self, train_df):
        # Global fallback
        self.global_mean = train_df["rating"].mean()

        # User-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        )

        # Compute user means
        self.user_means = self.user_item_matrix.mean(axis=1)

        # Mean-center ratings
        matrix_centered = self.user_item_matrix.sub(self.user_means, axis=0)

        # Fill missing values with 0 for similarity
        matrix_filled = matrix_centered.fillna(0)

        # Compute cosine similarity
        self.user_similarity = cosine_similarity(matrix_filled)

        # Store user ids
        self.user_ids = self.user_item_matrix.index.tolist()

    def predict(self, df):
        predictions = []

        for _, row in df.iterrows():
            user = row["user_id"]
            item = row["item_id"]

            # Cold-start user
            if user not in self.user_item_matrix.index:
                predictions.append(self.global_mean)
                continue

            # Cold-start item
            if item not in self.user_item_matrix.columns:
                predictions.append(self.user_means[user])
                continue

            user_index = self.user_ids.index(user)
            similarity_scores = self.user_similarity[user_index]

            item_ratings = self.user_item_matrix[item]

            # Build neighbor dataframe
            temp_df = pd.DataFrame({
                "user_id": self.user_ids,
                "similarity": similarity_scores,
                "rating": item_ratings.values
            }).dropna()

            # Sort by similarity
            temp_df = temp_df.sort_values(by="similarity", ascending=False)

            # Remove self
            temp_df = temp_df[temp_df["user_id"] != user]

            # Keep only positive similarities
            temp_df = temp_df[temp_df["similarity"] > 0]

            # Take top-k
            top_k = temp_df.head(self.k)

            if top_k.empty:
                pred = self.user_means[user]
            else:
                weighted_sum = 0
                sim_sum = 0

                for _, neighbor in top_k.iterrows():
                    neighbor_id = neighbor["user_id"]
                    sim = neighbor["similarity"]
                    rating = neighbor["rating"]

                    weighted_sum += sim * (rating - self.user_means[neighbor_id])
                    sim_sum += sim

                if sim_sum == 0:
                    pred = self.user_means[user]
                else:
                    pred = self.user_means[user] + (weighted_sum / sim_sum)

            # Clip prediction to valid range
            pred = min(5, max(1, pred))

            predictions.append(pred)

        return predictions

class ItemKNN:
    def __init__(self, k=20):
        self.k = k
        self.user_item_matrix = None
        self.item_similarity = None
        self.item_ids = None
        self.global_mean = None

    def fit(self, train_df):
        self.global_mean = train_df["rating"].mean()

        # Create user-item matrix
        self.user_item_matrix = train_df.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating"
        )

        # Fill missing values with 0 for similarity
        matrix_filled = self.user_item_matrix.fillna(0)

        # Compute item similarity (transpose!)
        self.item_similarity = cosine_similarity(matrix_filled.T)

        self.item_ids = self.user_item_matrix.columns.tolist()

    def predict(self, df):
        predictions = []

        for _, row in df.iterrows():
            user = row["user_id"]
            item = row["item_id"]

            # Cold start item
            if item not in self.user_item_matrix.columns:
                predictions.append(self.global_mean)
                continue

            # Cold start user
            if user not in self.user_item_matrix.index:
                predictions.append(self.global_mean)
                continue

            item_index = self.item_ids.index(item)
            similarity_scores = self.item_similarity[item_index]

            user_ratings = self.user_item_matrix.loc[user]

            temp_df = pd.DataFrame({
                "item_id": self.item_ids,
                "similarity": similarity_scores,
                "rating": user_ratings.values
            }).dropna()

            # Remove self
            temp_df = temp_df[temp_df["item_id"] != item]

            # Keep positive similarities
            temp_df = temp_df[temp_df["similarity"] > 0]

            temp_df = temp_df.sort_values(by="similarity", ascending=False)

            top_k = temp_df.head(self.k)

            if top_k.empty:
                pred = user_ratings.mean()
            else:
                weighted_sum = (top_k["similarity"] * top_k["rating"]).sum()
                sim_sum = top_k["similarity"].sum()
                pred = weighted_sum / sim_sum

            pred = min(5, max(1, pred))
            predictions.append(pred)

        return predictions