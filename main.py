

from sklearn.datasets import make_moons
from src.initializers.xavier import XavierInitializer

from src.loss.base_loss import BinaryCrossEntropy
from src.network import Network

lr = 0.1
epochs = 10
batch_size = 16

# generate 2d classification dataset
X, y = make_moons(n_samples=10000, shuffle=True,
                  noise=0.00, random_state=42)
X = X.tolist()
y = y.tolist()

net = Network(2, 12, 1)
initializer = XavierInitializer(net)
loss_fn = BinaryCrossEntropy()

# Init weights
initializer.init()

for epoch in range(epochs):
    epoch_losses = []
    for i in range(0, len(X), batch_size):
        batch_X = X[i:i+batch_size]
        batch_y = y[i:i+batch_size]
        net.zero_grad()

        batch_loss = 0
        preds = net.forward(batch_X)
        flattened_preds = [x[0] for x in preds]
        loss = loss_fn(flattened_preds, batch_y)
        batch_loss += loss
        net.backward(loss_fn.backward(flattened_preds, batch_y))
        net.step(lr)
        epoch_losses.append(batch_loss / batch_size)

    print(f"Epoch {epoch+1}, Average Loss: {sum(epoch_losses) / len(epoch_losses):.4f}")

# Test
correct_predictions = 0
total_predictions = 0

for i in range(0, len(X), batch_size):
    batch_X = X[i:i+batch_size]
    batch_y = y[i:i+batch_size]
    preds = net.forward(batch_X)
    flattened_preds = [x[0] for x in preds]

    for pred, true_label in zip(flattened_preds, batch_y):
        predicted_label = 1 if pred > 0.5 else 0
        if predicted_label == true_label:
            correct_predictions += 1
        total_predictions += 1

accuracy = correct_predictions / total_predictions
print(f"Accuracy: {accuracy:.4f}")
