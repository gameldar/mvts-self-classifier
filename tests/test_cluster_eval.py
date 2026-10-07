import numpy as np
import pytest

from cluster_eval import confusion_matrix, format_report, mapping_accuracy, purity


def test_confusion_matrix_counts_behaviour_against_cluster():
    behaviours = np.array(["resting", "resting", "walking", "walking"])
    clusters = np.array([0, 0, 1, 1])

    matrix = confusion_matrix(behaviours, clusters, num_clusters=2)

    assert matrix.loc["resting", 0] == 2
    assert matrix.loc["resting", 1] == 0
    assert matrix.loc["walking", 1] == 2


def test_confusion_matrix_keeps_clusters_the_model_never_used():
    behaviours = np.array(["resting", "walking"])
    clusters = np.array([0, 0])

    matrix = confusion_matrix(behaviours, clusters, num_clusters=4)

    assert list(matrix.columns) == [0, 1, 2, 3]
    assert matrix[3].sum() == 0


def test_purity_is_one_when_clusters_are_pure():
    behaviours = np.array(["resting", "resting", "walking", "walking"])
    clusters = np.array([3, 3, 1, 1])

    matrix = confusion_matrix(behaviours, clusters, num_clusters=4)

    assert purity(matrix) == pytest.approx(1.0)


def test_purity_is_a_half_when_every_cluster_is_evenly_mixed():
    behaviours = np.array(["resting", "walking", "resting", "walking"])
    clusters = np.array([0, 0, 1, 1])

    matrix = confusion_matrix(behaviours, clusters, num_clusters=2)

    assert purity(matrix) == pytest.approx(0.5)


def test_mapping_accuracy_ignores_how_clusters_are_numbered():
    behaviours = np.array(["resting", "resting", "walking", "walking"])
    permuted = np.array([1, 1, 0, 0])

    matrix = confusion_matrix(behaviours, permuted, num_clusters=2)

    assert mapping_accuracy(matrix) == pytest.approx(1.0)


def test_mapping_accuracy_penalises_collapse_onto_one_cluster():
    behaviours = np.array(["resting", "resting", "walking", "walking"])
    collapsed = np.array([0, 0, 0, 0])

    matrix = confusion_matrix(behaviours, collapsed, num_clusters=2)

    # Only one behaviour can claim the single used cluster, so half the rows are lost.
    assert mapping_accuracy(matrix) == pytest.approx(0.5)


def test_mapping_accuracy_is_never_below_chance_for_a_balanced_split():
    behaviours = np.array(["a", "a", "b", "b"])
    clusters = np.array([0, 1, 0, 1])

    matrix = confusion_matrix(behaviours, clusters, num_clusters=2)

    assert mapping_accuracy(matrix) == pytest.approx(0.5)


def test_report_names_each_behaviour_and_both_metrics():
    behaviours = np.array(["resting", "walking"])
    clusters = np.array([0, 1])

    report = format_report(confusion_matrix(behaviours, clusters, num_clusters=2))

    assert "resting" in report
    assert "walking" in report
    assert "purity" in report
    assert "accuracy" in report
