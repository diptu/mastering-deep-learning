import numpy as np


def train_test_split_scratch(X, y, test_size=0.2, random_state=None):
    # Convert inputs to numpy arrays for consistent indexing
    X = np.array(X)
    y = np.array(y)

    if len(X) != len(y):
        raise ValueError("Features and labels must have the same length.")

    if random_state:
        np.random.seed(random_state)

    # 1. Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    # 2. Determine split boundary
    test_set_size = int(len(X) * test_size)

    # 3. Partition indices
    test_indices = indices[:test_set_size]
    train_indices = indices[test_set_size:]

    # 4. Slice data
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    return X_train, X_test, y_train, y_test


# --- Example Usage ---
if __name__ == "__main__":
    # Mock data: 10 samples
    X = np.array([[i, i * 2] for i in range(10)])
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])

    X_train, X_test, y_train, y_test = train_test_split_scratch(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Total samples: {len(X)}")
    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
