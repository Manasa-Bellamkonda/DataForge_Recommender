import os
import pandas as pd
from sklearn.model_selection import train_test_split


def train_validation_test_split(ratings, test_size=0.1, val_size=0.1, random_state=42):
    """
    Splits ratings into train, validation, and test sets.
    """

    # First split: train+val and test
    train_val, test = train_test_split(
        ratings,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Adjust validation size relative to remaining data
    val_adjusted = val_size / (1 - test_size)

    train, val = train_test_split(
        train_val,
        test_size=val_adjusted,
        random_state=random_state,
        shuffle=True
    )

    return train, val, test


def save_splits(train, val, test, output_path="data/processed"):
    """
    Save train, validation, and test splits to CSV files.
    """

    os.makedirs(output_path, exist_ok=True)

    train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    val.to_csv(os.path.join(output_path, "validation.csv"), index=False)
    test.to_csv(os.path.join(output_path, "test.csv"), index=False)

    print("Splits saved successfully.")