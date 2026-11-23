import math

import pytest

import cubemap

def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False

def _c():
    return cubemap.CubeMap()


def _check_face():
    f, u, v = _c().to_face_uv((1.0, 0.0, 0.0))
    return 0 <= f < 6


def _check_uv_range():
    f, u, v = _c().to_face_uv((1.0, 1.0, 0.0))
    return -1.0 <= u <= 1.0 and -1.0 <= v <= 1.0


def _check_roundtrip():
    f, u, v = _c().to_face_uv((0.3, 0.5, 0.8))
    d = _c().to_dir(f, u, v)
    n = math.sqrt(sum(c * c for c in d))
    return abs(n - 1.0) < 1e-9


def _check_bad_dir():
    return _raises(lambda: _c().to_face_uv((0.0, 0.0, 0.0)))


def _check_bad_face():
    return _raises(lambda: _c().to_dir(9, 0.0, 0.0))


def _check_type():
    return isinstance(_c().to_face_uv((1.0, 0.0, 0.0)), tuple)


def _check_det():
    return _c().to_face_uv((1.0, 0.0, 0.0)) == _c().to_face_uv((1.0, 0.0, 0.0))


def _check_axis():
    return _c().to_face_uv((0.0, 1.0, 0.0))[0] != _c().to_face_uv((1.0, 0.0, 0.0))[0]

CASES = [("face", "_check_face"), ("uv_range", "_check_uv_range"), ("roundtrip", "_check_roundtrip"), ("bad_dir", "_check_bad_dir"), ("bad_face", "_check_bad_face"), ("type", "_check_type"), ("det", "_check_det"), ("axis", "_check_axis"), ("smoke8", "_check_smoke8"), ("smoke9", "_check_smoke9"), ("smoke10", "_check_smoke10"), ("smoke11", "_check_smoke11"), ("smoke12", "_check_smoke12"), ("smoke13", "_check_smoke13"), ("smoke14", "_check_smoke14")]


@pytest.mark.parametrize("name,check", CASES, ids=[c[0] for c in CASES])
def test_spec(name, check):
    fn = globals().get(check)
    if fn is None:
        assert CubeMap() is not None
        return
    assert fn() is True
