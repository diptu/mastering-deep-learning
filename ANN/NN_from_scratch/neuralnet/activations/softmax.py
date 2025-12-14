import numpy as np


def softmax(x):
    """
    Computes the numerically stable Softmax function.

    Prevents floating-point overflow by shifting logits using the max value of each sample.

    :param x: A 2D NumPy array of shape (n_samples, n_classes).
    :return: A 2D NumPy array of probabilities (summing to 1 along axis 1).
    """

    shiftx = x - x.max(axis=1, keepdims=True)

    # 2. Standard Softmax calculation using the shifted values.
    exp_values = np.exp(shiftx)

    # 3. Normalization: Sum the exponentials along the class dimension (axis=1)
    norm_val = np.sum(exp_values, axis=1, keepdims=True)

    # Optional: Print for debugging (removed in production code)
    # print(f"norm_val: {norm_val} ")

    prob = exp_values / norm_val
    return prob


if __name__ == "__main__":
    # Test 1: Small values (Your original test case)
    X_small = np.array([[1, 2, 3, 4], [4, 5, 6, 7]])

    res_small = softmax(X_small)
    print("--- Test 1: Small Values ---")
    print("Input Logits X:")
    print(X_small)
    print("\nSoftmax Output (Probabilities):")
    print(res_small)
    print(f"\nVerification (Total Sums per row): {res_small.sum(axis=1)}")

    print("\n" + "=" * 30 + "\n")
