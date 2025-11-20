from app.inference.feature_selection import select_features
from app.inference.load_data import load_dataset


def preprocess_pipeline():
    target_col = "Exited"
    df = load_dataset()
    processed_df = select_features(df, target_col)
    print(f"Data after dropping weak features has {processed_df.shape[1]} columns")
    return processed_df


if __name__ == "__main__":
    df_final = preprocess_pipeline()
    print(df_final.head())
