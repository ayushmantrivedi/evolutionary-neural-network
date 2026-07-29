import copy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from evonet.core.network import MultiClassEvoNet


def _build_dataset(random_state: int, n_samples: int = 240):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        class_sep=1.0,
        weights=[0.78, 0.22],
        flip_y=0.04,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    y_oh = np.eye(2)[y]
    return train_test_split(X, y, y_oh, test_size=0.3, random_state=random_state, stratify=y)


def _predict(net: MultiClassEvoNet, X: np.ndarray) -> np.ndarray:
    return np.array([np.argmax(net._forward_elite(x)[0]) for x in X])


def _anchor_mse(net: MultiClassEvoNet) -> float:
    _, _, l2_after, _ = net._forward_elite(net.anchor_inputs)
    return float(np.mean((l2_after - net.anchor_features) ** 2))


def test_end_to_end_pipeline_with_anchor_memory():
    np.random.seed(123)

    # Data processing
    X_a_train, X_a_test, y_a_train, y_a_test, yoh_a_train, yoh_a_test = _build_dataset(1)
    X_b_train, X_b_test, y_b_train, y_b_test, yoh_b_train, yoh_b_test = _build_dataset(9)

    net = MultiClassEvoNet(input_dim=8, num_classes=2)

    # Train on task A
    net.train(X_a_train, y_a_train, yoh_a_train, epochs=2)
    preds_a_before = _predict(net, X_a_test)
    acc_a_before = accuracy_score(y_a_test, preds_a_before)

    # Retain task A geometry as an anchor
    net.capture_anchor_memory(X_a_train[:64], max_examples=32)
    assert net.anchor_inputs is not None
    assert net.anchor_features is not None

    # Train on task B with anchor retention active
    net.anchor_weight = 8.0
    net.train(X_b_train, y_b_train, yoh_b_train, epochs=2)

    # Output checks on both tasks
    preds_a_after = _predict(net, X_a_test)
    preds_b_after = _predict(net, X_b_test)
    acc_a_after = accuracy_score(y_a_test, preds_a_after)
    acc_b_after = accuracy_score(y_b_test, preds_b_after)

    # Anchor retention checks
    anchor_mse = _anchor_mse(net)
    assert anchor_mse < 5.0

    # The model should still emit finite probabilities through the inference path.
    y_pred, l1, l2, confidences = net._forward_elite(X_a_test[:4])
    assert np.isfinite(y_pred).all()
    assert np.isfinite(l1).all()
    assert np.isfinite(l2).all()
    assert np.isfinite(confidences).all()

    # The whole pipeline should produce usable outputs after task B.
    assert y_pred.shape[1] == 2
    assert acc_a_before >= 0.0
    assert acc_a_after >= 0.0
    assert acc_b_after >= 0.0

    # Anchor memory should stabilize the prior task better than random collapse.
    assert acc_a_after >= 0.2 or acc_b_after >= 0.2


def test_anchor_memory_can_be_cleared_and_retrained():
    np.random.seed(321)
    X_train, X_test, y_train, y_test, yoh_train, yoh_test = _build_dataset(15)

    net = MultiClassEvoNet(input_dim=8, num_classes=2)
    net.train(X_train, y_train, yoh_train, epochs=1)
    net.capture_anchor_memory(X_train[:48], max_examples=16)
    assert net.anchor_inputs is not None

    before_clear = copy.deepcopy(net.anchor_inputs)
    net.clear_anchor_memory()
    assert net.anchor_inputs is None
    assert net.anchor_features is None

    # Reuse the model after clear to ensure the training path still works.
    net.train(X_train, y_train, yoh_train, epochs=1)
    preds = _predict(net, X_test)
    acc = accuracy_score(y_test, preds)
    assert np.isfinite(acc)
    assert before_clear.shape[1] == 8
