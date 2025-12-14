class Trainer:
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

    def train_step(self, x, y):
        logits = self.model.forward(x)
        loss = self.loss.forward(logits, y)

        grad = self.loss.backward()
        self.model.backward(grad)

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss
