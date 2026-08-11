import pytest

from aquila.interface.problem import TaskAllocationProblem


def test_rejects_empty_weights():
    with pytest.raises(ValueError):
        TaskAllocationProblem(task_weights=[])


def test_rejects_nonpositive_weights():
    with pytest.raises(ValueError):
        TaskAllocationProblem(task_weights=[1.0, -2.0])


def test_default_task_ids():
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0, 3.0])
    assert problem.task_ids == ["task-0", "task-1", "task-2"]


def test_mismatched_ids_length_rejected():
    with pytest.raises(ValueError):
        TaskAllocationProblem(task_weights=[1.0, 2.0], task_ids=["only-one"])


def test_ising_coefficients_symmetric_count():
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0, 3.0, 4.0])
    coeffs = problem.ising_coefficients()
    # n choose 2 pairs, each i<j exactly once
    assert len(coeffs) == 6
    assert coeffs[(0, 1)] == pytest.approx(2.0 * 1.0 * 2.0)


def test_cost_of_assignment_matches_manual_computation():
    problem = TaskAllocationProblem(task_weights=[3.0, 5.0, 2.0])
    spins = [1, -1, 1]
    expected = (3.0 - 5.0 + 2.0) ** 2
    assert problem.cost_of_assignment(spins) == pytest.approx(expected)


def test_cost_of_assignment_rejects_wrong_length():
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        problem.cost_of_assignment([1, -1, 1])


def test_cost_of_assignment_rejects_invalid_spin_values():
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0])
    with pytest.raises(ValueError):
        problem.cost_of_assignment([1, 0])


def test_clusters_from_assignment():
    problem = TaskAllocationProblem(task_weights=[1.0, 2.0, 3.0], task_ids=["a", "b", "c"])
    cluster_a, cluster_b = problem.clusters_from_assignment([1, -1, 1])
    assert cluster_a == ["a", "c"]
    assert cluster_b == ["b"]


@pytest.mark.parametrize(
    "bitstring,num_bits,expected",
    [
        (0, 3, [-1, -1, -1]),
        (1, 3, [1, -1, -1]),
        (5, 3, [1, -1, 1]),  # 5 = 0b101
        (7, 3, [1, 1, 1]),
    ],
)
def test_bitstring_to_spins(bitstring, num_bits, expected):
    assert TaskAllocationProblem.bitstring_to_spins(bitstring, num_bits) == expected
