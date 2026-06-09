from sklearn.neural_network import MLPClassifier
from src.config import SEED

def build_mlp(n_layers):
    hidden_layers = tuple([2**x for x in range(n_layers, 1, -1)])

    return MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation="relu",
        solver="adam",
        random_state=SEED,
        max_iter=500,
        early_stopping=True,
    )
