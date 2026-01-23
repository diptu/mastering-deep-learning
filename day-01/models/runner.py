import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
from model_selection.train_test_split import train_test_split_scratch
from .simple_linear_regression import LinearRegression


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PATH = BASE_DIR / "data" / "Housing.csv"

    cols_to_keep = ["price", "area"]
    df = pd.read_csv(DATA_PATH)[cols_to_keep]

    # --- Prepare data ---
    X = df["area"].values
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split_scratch(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")

    # --- Train model ---
    model = LinearRegression(lr=0.01, epochs=2000)
    model.fit(X_train, y_train)

    # --- Draw regression line ---
    X_line = np.linspace(X.min(), X.max(), 100)
    y_line = model.predict(X_line)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, alpha=0.5, label="Data points")
    plt.plot(X_line, y_line, color="red", label="Regression line")

    plt.xlabel("Area (sq ft)")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Simple Linear Regression (Area vs Price)")
    plt.show()

    print(f"Learned w: {model.w:.4f}")
    print(f"Learned b: {model.b:.4f}")


if __name__ == "__main__":
    main()
