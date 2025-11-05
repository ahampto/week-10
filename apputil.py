import pandas as pd
import numpy as np
import pickle

# Roast mapping (same as in train.py)
def roast_category(roast_value):
    mapping = {
        "Light": 0,
        "Medium-Light": 1,
        "Medium": 2,
        "Medium-Dark": 3,
        "Dark": 4
    }
    return mapping.get(roast_value, np.nan)


def predict_rating(df_X):
    """
    df_X: pandas DataFrame with columns:
        - "100g_USD" (numerical)
        - "roast" (text, can be missing or unknown)
    
    Returns: numpy array of predicted ratings
    """
    # Load models
    with open("model_1.pickle", "rb") as f:
        lr = pickle.load(f)
    with open("model_2.pickle", "rb") as f:
        dtr = pickle.load(f)

    df = df_X.copy()

    # Map roast to numeric category
    df["roast_cat"] = df["roast"].map(roast_category)

    # Prepare predictions array
    y_pred = np.zeros(len(df))

    mask_tree = df["roast_cat"].notna()
    if mask_tree.any():
        X_tree = df.loc[mask_tree, ["100g_USD", "roast_cat"]].values
        y_pred[mask_tree] = dtr.predict(X_tree)

    mask_lr = ~mask_tree
    if mask_lr.any():
        X_lr = df.loc[mask_lr, ["100g_USD"]].values
        y_pred[mask_lr] = lr.predict(X_lr)

    return y_pred
