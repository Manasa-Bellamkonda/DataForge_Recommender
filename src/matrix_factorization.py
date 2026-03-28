import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors=20, lr=0.01, reg=0.02, n_epochs=20):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs

    def fit(self, train_df):

        self.global_mean = train_df["rating"].mean()

        self.users = train_df["user_id"].unique()
        self.items = train_df["item_id"].unique()

        self.user_map = {u: i for i, u in enumerate(self.users)}
        self.item_map = {i: j for j, i in enumerate(self.items)}

        n_users = len(self.users)
        n_items = len(self.items)

        # Initialize latent factors
        self.P = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = np.random.normal(0, 0.1, (n_items, self.n_factors))

        # Initialize biases
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)

        for epoch in range(self.n_epochs):

            for _, row in train_df.iterrows():

                u = row["user_id"]
                i = row["item_id"]
                r = row["rating"]

                if u not in self.user_map or i not in self.item_map:
                    continue

                u_idx = self.user_map[u]
                i_idx = self.item_map[i]

                pred = (
                    self.global_mean
                    + self.b_u[u_idx]
                    + self.b_i[i_idx]
                    + np.dot(self.P[u_idx], self.Q[i_idx])
                )

                err = r - pred

                # Update biases
                self.b_u[u_idx] += self.lr * (err - self.reg * self.b_u[u_idx])
                self.b_i[i_idx] += self.lr * (err - self.reg * self.b_i[i_idx])

                # Update latent factors
                P_old = self.P[u_idx].copy()

                self.P[u_idx] += self.lr * (
                    err * self.Q[i_idx] - self.reg * self.P[u_idx]
                )

                self.Q[i_idx] += self.lr * (
                    err * P_old - self.reg * self.Q[i_idx]
                )

    def predict(self, df):

        preds = []

        for _, row in df.iterrows():

            u = row["user_id"]
            i = row["item_id"]

            if u not in self.user_map or i not in self.item_map:
                preds.append(self.global_mean)
                continue

            u_idx = self.user_map[u]
            i_idx = self.item_map[i]

            pred = (
                self.global_mean
                + self.b_u[u_idx]
                + self.b_i[i_idx]
                + np.dot(self.P[u_idx], self.Q[i_idx])
            )

            # Clip to rating scale
            pred = min(5, max(1, pred))

            preds.append(pred)

        return preds