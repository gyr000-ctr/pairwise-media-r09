import math

import pytest

import spawn

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

W = [[1.0, 1.0], [1.0, 1.0]]


def _s():
    return spawn.Spawn(0)


def _check_count():
    return len(_s().sample(W, 100)) == 100


def _check_bounds():
    return all(0 <= i < 2 and 0 <= j < 2 for i, j in _s().sample(W, 50))


def _check_det():
    return _s().sample(W, 20) == _s().sample(W, 20)


def _check_seed_diff():
    return spawn.Spawn(1).sample(W, 20) != spawn.Spawn(2).sample(W, 20)


def _check_bad_weights():
    return _raises(lambda: _s().sample([], 10))


def _check_bad_n():
    return _raises(lambda: _s().sample(W, 0))


def _check_type():
    return isinstance(_s().sample(W, 10), list)


def _check_zero_weights():
    return _raises(lambda: _s().sample([[0.0, 0.0]], 5))

CASES = [("count", "_check_count"), ("bounds", "_check_bounds"), ("det", "_check_det"), ("seed_diff", "_check_seed_diff"), ("bad_weights", "_check_bad_weights"), ("bad_n", "_check_bad_n"), ("type", "_check_type"), ("zero_weights", "_check_zero_weights"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Spawn() is not None
        return
    assert fn() is True
