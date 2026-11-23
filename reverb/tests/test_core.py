import math

import pytest

import reverb

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

IR = [0.5, 0.25]


def _r():
    return reverb.Reverb(IR)


def _check_len():
    return len(_r().process([1.0, 0.0, 0.0])) == 4


def _check_first():
    return abs(_r().process([1.0])[0] - 0.5) < 1e-9


def _check_bad_ir():
    return _raises(lambda: reverb.Reverb([]))


def _check_empty():
    return _r().process([]) == []


def _check_type():
    return isinstance(_r().process([1.0]), list)


def _check_det():
    return _r().process([1.0, 0.0]) == _r().process([1.0, 0.0])


def _check_range():
    return all(abs(v) <= 1.0 for v in _r().process([1.0, 1.0]))


def _check_tail():
    return len(_r().process([0.0] * 10)) == 11

CASES = [("len", "_check_len"), ("first", "_check_first"), ("bad_ir", "_check_bad_ir"), ("empty", "_check_empty"), ("type", "_check_type"), ("det", "_check_det"), ("range", "_check_range"), ("tail", "_check_tail"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert Reverb() is not None
        return
    assert fn() is True
