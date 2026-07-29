import copy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from evonet.core.network import MultiClassEvoNet


def _make_task(random_state: int, n_samples: int = 180):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        weights=[0.75, 0.25],
        class_sep=1.2,
        flip_y=0.03,
        random_state=random_state,
    )
    X = StandardScaler().fit_transform(X)
    y_oh = np.eye(2)[y]
    return train_test_split(X, y, y_oh, test_size=0.3, random_state=random_state, stratify=y)


def _clone_network(net: MultiClassEvoNet) -> MultiClassEvoNet:
    return copy.deepcopy(net)


def _anchor_drift(net: MultiClassEvoNet, anchor_inputs: np.ndarray, anchor_features: np.ndarray) -> float:
    _, _, l2_after, _ = net._forward_elite(anchor_inputs)
    return float(np.mean((l2_after - anchor_features) ** 2))


def test_anchor_capture_limits_and_clears_memory():
    np.random.seed(7)
    net = MultiClassEvoNet(input_dim=8, num_classes=2)
    X_train, _, y_train, _, y_oh_train, _ = _make_task(11)

    net.capture_anchor_memory(X_train[:120], max_examples=24)

    assert net.anchor_inputs is not None
    assert net.anchor_features is not None
    assert net.anchor_inputs.shape[0] == 24
    assert net.anchor_features.shape[0] == 24
    assert net.anchor_inputs.shape[1] == 8

    net.clear_anchor_memory()
    assert net.anchor_inputs is None
    assert net.anchor_features is None


def test_anchor_changes_training_objective():
    np.random.seed(21)
    base = MultiClassEvoNet(input_dim=8, num_classes=2)
    anchored = _clone_network(base)

    X_a_train, X_a_test, y_a_train, y_a_test, yoh_a_train, yoh_a_test = _make_task(3)
    X_b_train, _, y_b_train, _, yoh_b_train, _ = _make_task(17)

    anchored.capture_anchor_memory(X_a_train[:64], max_examples=32)
    anchored.anchor_weight = 1.0

    loss_base = base._train_one_batch(X_b_train[:32], yoh_b_train[:32], base.get_mutation_strength())
    loss_anchor = anchored._train_one_batch(X_b_train[:32], yoh_b_train[:32], anchored.get_mutation_strength())

    assert not np.isclose(loss_base, loss_anchor)


def test_anchor_reduces_representation_drift_after_new_task():
    np.random.seed(33)
    base = MultiClassEvoNet(input_dim=8, num_classes=2)
    anchored = _clone_network(base)

    X_a_train, X_a_test, y_a_train, y_a_test, yoh_a_train, yoh_a_test = _make_task(5)
    X_b_train, X_b_test, y_b_train, y_b_test, yoh_b_train, yoh_b_test = _make_task(19)

    base.train(X_a_train, y_a_train, yoh_a_train, epochs=1)
    anchored = _clone_network(base)
    anchored.capture_anchor_memory(X_a_train[:48], max_examples=32)
    anchored.anchor_weight = 10.0

    baseline_before = _clone_network(base)
    anchored_before = _clone_network(anchored)

    np.random.seed(99)
    baseline_before.train(X_b_train, y_b_train, yoh_b_train, epochs=3)
    np.random.seed(99)
    anchored_before.train(X_b_train, y_b_train, yoh_b_train, epochs=3)

    drift_base = _anchor_drift(baseline_before, anchored.anchor_inputs, anchored.anchor_features)
    drift_anchor = _anchor_drift(anchored_before, anchored.anchor_inputs, anchored.anchor_features)

    assert drift_anchor < drift_base
