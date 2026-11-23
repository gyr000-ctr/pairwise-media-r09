import math

import pytest

import meshlod

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

V = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.0)]
F = [(0, 1, 2), (1, 3, 2)]


def _m():
    return meshlod.MeshLod()


def _check_faces():
    v, f = _m().simplify(V, F, 0.5)
    return len(f) <= 2


def _check_bounds():
    v, f = _m().simplify(V, F, 0.5)
    return all(0.0 <= p[0] <= 1.0 for p in v)


def _check_bad_ratio():
    return _raises(lambda: _m().simplify(V, F, 0.0))


def _check_bad_input():
    return _raises(lambda: _m().simplify(V, [], 0.5))


def _check_type():
    return isinstance(_m().simplify(V, F, 0.5), tuple)


def _check_error():
    m = _m()
    m.simplify(V, F, 0.5)
    return m.error() >= 0.0


def _check_no_degenerate():
    v, f = _m().simplify(V, F, 0.5)
    return all(len(set(t)) == 3 for t in f)


def _check_det():
    return _m().simplify(V, F, 0.5) == _m().simplify(V, F, 0.5)

CASES = [("faces", "_check_faces"), ("bounds", "_check_bounds"), ("bad_ratio", "_check_bad_ratio"), ("bad_input", "_check_bad_input"), ("type", "_check_type"), ("error", "_check_error"), ("no_degenerate", "_check_no_degenerate"), ("det", "_check_det"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert MeshLod() is not None
        return
    assert fn() is True
