import numpy as np


class HybridRecommender:
    def __init__(self, bias_model, user_knn_model, item_knn_model,
                 alpha=0.3, beta=0.4):
        """
        alpha = weight for bias model
        beta  = weight for user KNN
        (1 - alpha - beta) = weight for item KNN
        """
        self.bias_model = bias_model
        self.user_knn_model = user_knn_model
        self.item_knn_model = item_knn_model
        self.alpha = alpha
        self.beta = beta

    def predict(self, df):

        bias_preds = self.bias_model.predict(df)
        user_preds = self.user_knn_model.predict(df)
        item_preds = self.item_knn_model.predict(df)

        hybrid_preds = []

        gamma = 1 - self.alpha - self.beta

        for b, u, i in zip(bias_preds, user_preds, item_preds):
            pred = (
                self.alpha * b +
                self.beta * u +
                gamma * i
            )
            hybrid_preds.append(pred)

        return hybrid_preds