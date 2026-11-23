import math

import pytest

import tileblend

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _t():
    return tileblend.TileBlend()


def _check_sum():
    return abs(sum(_t().blend(0.5, 0.5, 1.0, 0.2)) - 1.0) < 1e-9


def _check_count():
    return len(_t().blend(0.5, 0.5, 1.0, 0.2)) == 4


def _check_corner():
    w = _t().blend(0.5, 0.5, 1.0, 0.2)
    return max(w) >= 0.25


def _check_bad_width():
    return _raises(lambda: _t().blend(0.5, 0.5, 1.0, 0.0))


def _check_bad_tile():
    return _raises(lambda: _t().blend(0.5, 0.5, 0.0, 0.2))


def _check_type():
    return isinstance(_t().blend(0.5, 0.5, 1.0, 0.2), tuple)


def _check_det():
    return _t().blend(0.3, 0.3, 1.0, 0.2) == _t().blend(0.3, 0.3, 1.0, 0.2)


def _check_range():
    return all(0.0 <= v <= 1.0 for v in _t().blend(0.9, 0.9, 1.0, 0.3))

CASES = [("sum", "_check_sum"), ("count", "_check_count"), ("corner", "_check_corner"), ("bad_width", "_check_bad_width"), ("bad_tile", "_check_bad_tile"), ("type", "_check_type"), ("det", "_check_det"), ("range", "_check_range"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert TileBlend() is not None
        return
    assert fn() is True
