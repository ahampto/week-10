import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor



def train_linear_regression():
    # Load data
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)

    # Drop missing values
    df = df.dropna(subset=["rating", "100g_USD"])

    X = df[["100g_USD"]].values
    y = df["rating"].values

    lr = LinearRegression()
    lr.fit(X, y)

    # Save model
    with open("model_1.pickle", "wb") as f:
        pickle.dump(lr, f)

    print("model_1.pickle saved successfully!")


def roast_category(roast_value):
    """
    Map roast types to numeric categories.
    Missing or unknown values can return np.nan.
    """
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast_value, np.nan)


def train_decision_tree():
    
    url = "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/data/coffee_analysis.csv"
    df = pd.read_csv(url)

    # Create roast category numeric column
    df["roast_cat"] = df["roast"].map(roast_category)

    # Drop rows missing required features
    df = df.dropna(subset=["rating", "100g_USD", "roast_cat"])

    # Define features and target
    X = df[["100g_USD", "roast_cat"]].values
    y = df["rating"].values

    dtr = DecisionTreeRegressor(random_state=42)
    dtr.fit(X, y)

    # Save model
    with open("model_2.pickle", "wb") as f:
        pickle.dump(dtr, f)

    print("model_2.pickle saved successfully!")



if __name__ == "__main__":
    train_linear_regression()
    train_decision_tree()
