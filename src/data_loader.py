import pandas as pd
import os


def load_movielens_100k(data_path=None):
    """
    Load MovieLens 100K dataset.
    Automatically detects project root.
    """

    # Get project root (parent of src folder)
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    if data_path is None:
        data_path = os.path.join(base_dir, "data", "raw", "ml-100k")

    # -------------------------
    # Load Ratings
    # -------------------------
    ratings_cols = ["user_id", "item_id", "rating", "timestamp"]

    ratings = pd.read_csv(
        os.path.join(data_path, "u.data"),
        sep="\t",
        names=ratings_cols
    )

    # -------------------------
    # Load Movies
    # -------------------------
    movies_cols = [
        "item_id", "title", "release_date", "video_release_date",
        "IMDb_URL", "unknown", "Action", "Adventure", "Animation",
        "Children's", "Comedy", "Crime", "Documentary", "Drama",
        "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
        "Romance", "Sci-Fi", "Thriller", "War", "Western"
    ]

    movies = pd.read_csv(
        os.path.join(data_path, "u.item"),
        sep="|",
        names=movies_cols,
        encoding="latin-1"
    )

    # -------------------------
    # Load Users
    # -------------------------
    users_cols = ["user_id", "age", "gender", "occupation", "zip_code"]

    users = pd.read_csv(
        os.path.join(data_path, "u.user"),
        sep="|",
        names=users_cols
    )

    return ratings, movies, users

if __name__ == "__main__":
    ratings, movies, users = load_movielens_100k()

    print("Ratings shape:", ratings.shape)
    print("Movies shape:", movies.shape)
    print("Users shape:", users.shape)

    print("\nSample Ratings:")
    print(ratings.head())