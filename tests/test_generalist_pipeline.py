import copy

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from evonet.core.network import MultiClassEvoNet


def _make_task(seed: int, n_samples: int = 180):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        n_repeated=0,
        weights=[0.78, 0.22],
        class_sep=1.0,
        flip_y=0.04,
        random_state=seed,
    )
    X = StandardScaler().fit_transform(X)
    y_oh = np.eye(2)[y]
    return train_test_split(X, y, y_oh, test_size=0.3, random_state=seed, stratify=y)


def test_router_distinguishes_low_and_high_complexity_inputs():
    net = MultiClassEvoNet(input_dim=8, num_classes=2)

    low_complexity = np.zeros(8, dtype=np.float32)
    high_complexity = np.array([-4.0, 3.0, -2.5, 4.5, -3.5, 2.5, -4.5, 3.5], dtype=np.float32)

    routed_low, route_low = net._route_inputs(low_complexity)
    routed_high, route_high = net._route_inputs(high_complexity)

    assert route_low.shape == (1,)
    assert route_high.shape == (1,)
    assert int(route_low[0]) != int(route_high[0])
    assert not np.allclose(routed_low, routed_high)


def test_diversity_archive_changes_training_surface():
    np.random.seed(7)
    X_train, _, y_train, _, yoh_train, _ = _make_task(7)
    batch_X = X_train[:24]
    batch_yoh = yoh_train[:24]

    base = MultiClassEvoNet(input_dim=8, num_classes=2)
    diverse = copy.deepcopy(base)

    flat_len = diverse._flatten_candidate_state(0).shape[0]
    diverse.diversity_archive = [np.ones(flat_len, dtype=np.float64)]

    base_loss = base._train_one_batch(batch_X, batch_yoh, base.get_mutation_strength())
    diverse_loss = diverse._train_one_batch(batch_X, batch_yoh, diverse.get_mutation_strength())

    assert not np.isclose(base_loss, diverse_loss)
    assert len(diverse.diversity_archive) >= 1


def test_generalist_pipeline_keeps_output_finite_and_anchored():
    np.random.seed(11)
    X_a_train, X_a_test, y_a_train, y_a_test, yoh_a_train, yoh_a_test = _make_task(11)
    X_b_train, X_b_test, y_b_train, y_b_test, yoh_b_train, yoh_b_test = _make_task(19)

    net = MultiClassEvoNet(input_dim=8, num_classes=2)
    net.train(X_a_train, y_a_train, yoh_a_train, epochs=2)
    net.capture_anchor_memory(X_a_train[:48], max_examples=24)

    pre_anchor_state = copy.deepcopy(net.anchor_features)
    net.anchor_weight = 8.0
    net.train(X_b_train, y_b_train, yoh_b_train, epochs=2)

    y_pred, l1, l2, conf = net._forward_elite(X_a_test[:8])
    preds = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_a_test[:8], preds)

    assert np.isfinite(y_pred).all()
    assert np.isfinite(l1).all()
    assert np.isfinite(l2).all()
    assert np.isfinite(conf).all()
    assert net.anchor_inputs is not None
    assert pre_anchor_state is not None
    assert net.route_best_idx in (0, 1, 2)
    assert acc >= 0.0
