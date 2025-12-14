from activations.relu import ReLU
from layers.dense import Dense
from models.sequential import Sequential
from losses.cross_entropy import SparseCrossEntropy
from optimizers.sgd import SGD
from engine.trainer import Trainer

model = Sequential(
    [
        Dense(784, 64),
        ReLU(),
        Dense(64, 32),
        ReLU(),
        Dense(32, 10),
    ]
)

loss = SparseCrossEntropy()
optimizer = SGD(model.parameters(), lr=0.01)

trainer = Trainer(model, loss, optimizer)
