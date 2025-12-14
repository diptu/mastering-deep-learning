import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from neuralnet.models.sequential import Sequential
from neuralnet.layers.dense import Dense
from neuralnet.activations.relu import ReLU
from neuralnet.losses.cross_entropy import SparseCrossEntropy
from neuralnet.optimizers.sgd import SGD


def main():
    X, y = make_classification(
        n_samples=100,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42,
    )

    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Sequential([Dense(20, 64), ReLU(), Dense(64, 32), ReLU(), Dense(32, 3)])
    loss_fn = SparseCrossEntropy()
    optimizer = SGD(model.parameters(), lr=0.01)

    epochs = 50
    batch_size = 32

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

        epoch_loss = 0.0

        for i in range(0, len(X_train), batch_size):
            x_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            logits = model.forward(x_batch)
            loss = loss_fn.forward(logits, y_batch)

            grad = loss_fn.backward()
            model.backward(grad)

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss

        print(f"Epoch {epoch + 1:02d} | Loss: {epoch_loss:.4f}")

    def accuracy(model, X, y):
        logits = model.forward(X)
        preds = np.argmax(logits, axis=1)
        return (preds == y).mean()

    train_acc = accuracy(model, X_train, y_train)
    test_acc = accuracy(model, X_test, y_test)

    print(f"Train Accuracy: {train_acc:.3f}")
    print(f"Test Accuracy:  {test_acc:.3f}")


if __name__ == "__main__":
    main()
