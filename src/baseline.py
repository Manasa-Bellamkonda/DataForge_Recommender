import numpy as np


class GlobalMeanModel:
    def __init__(self):
        self.global_mean = None

    def fit(self, train_df):
        self.global_mean = train_df["rating"].mean()

    def predict(self, df):
        return np.full(len(df), self.global_mean)


class BiasModel:
    def __init__(self):
        self.global_mean = None
        self.user_bias = None
        self.item_bias = None

    def fit(self, train_df):
        self.global_mean = train_df["rating"].mean()

        self.user_bias = (
            train_df.groupby("user_id")["rating"].mean()
            - self.global_mean
        )

        self.item_bias = (
            train_df.groupby("item_id")["rating"].mean()
            - self.global_mean
        )

    def predict(self, df):
        predictions = []

        for _, row in df.iterrows():
            u = row["user_id"]
            i = row["item_id"]

            bu = self.user_bias.get(u, 0)
            bi = self.item_bias.get(i, 0)

            pred = self.global_mean + bu + bi
            predictions.append(pred)

        return predictions