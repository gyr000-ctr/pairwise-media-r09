"""卷积混响的约束验收：对照直接卷积、能量、尾部长度、跨缓冲、性能、非法参数、并发。"""

import math
import threading
import time

import numpy as np
import pytest

import reverb

TOL = 1e-9


def _direct(x, ir):
    return np.convolve(np.asarray(x, dtype=float), np.asarray(ir, dtype=float))


@pytest.mark.parametrize("m", [1, 2, 3, 17, 64, 512, 4096])
def test_matches_direct_convolution(m):
    rng = np.random.default_rng(m)
    ir = rng.standard_normal(m)
    for n in (1, 5, 100, 1000, 20000):
        x = rng.standard_normal(n)
        got = reverb.Reverb(ir).process(x)
        ref = _direct(x, ir)
        assert len(got) == len(ref)
        assert np.max(np.abs(np.asarray(got) - ref)) <= TOL


def test_energy_conserved_for_normalized_ir():
    rng = np.random.default_rng(7)
    ir = rng.standard_normal(333)
    ir /= np.linalg.norm(ir)
    out = np.asarray(reverb.Reverb(ir).process([1.0]))
    assert math.isclose(float(out @ out), 1.0, rel_tol=0.0, abs_tol=TOL)


@pytest.mark.parametrize("m", [1, 2, 7, 1000])
def test_tail_length_equals_ir_length(m):
    n = 5000
    got = reverb.Reverb(np.ones(m)).process(np.zeros(n))
    # 输入 n 个样本 + IR 长度 m 的完整拖尾（含直达项）=> n + m - 1
    assert len(got) == n + m - 1


def test_cross_buffer_continuity():
    rng = np.random.default_rng(11)
    ir = rng.standard_normal(97)
    x = rng.standard_normal(10000)

    whole = reverb.Reverb(ir).process(x)
    assert np.max(np.abs(np.asarray(whole) - _direct(x, ir))) <= TOL

    stateful = reverb.Reverb(ir)
    streamed = []
    cuts = [0, 1, 333, 5000, 10000]
    for a, b in zip(cuts, cuts[1:]):
        chunk = stateful.process(x[a:b])
        # 各块的完整输出末尾带 hist_len 个尾样本；只取前 len(chunk) 个
        # 依序拼接即可还原整段信号（分块与整段结果一致）。
        streamed.extend(chunk[: b - a])
    assert np.max(np.abs(np.asarray(streamed) - np.asarray(whole)[:10000])) <= TOL

    # 中间插入空缓冲不应扰动状态
    stateful = reverb.Reverb(ir)
    a = stateful.process(x[:6000])
    assert stateful.process([]) == []
    b = stateful.process(x[6000:])
    recombined = np.concatenate((np.asarray(a)[:6000], np.asarray(b)[:4000]))
    assert np.max(np.abs(recombined - np.asarray(whole)[:10000])) <= TOL


@pytest.mark.parametrize("m", [1, 64, 512, 4096, 8191, 50000, 1_000_000])
def test_one_million_samples_under_budget(m):
    rng = np.random.default_rng(m)
    ir = rng.standard_normal(m)
    x = rng.standard_normal(1_000_000)
    engine = reverb.Reverb(ir)
    start = time.perf_counter()
    out = engine.process(x)
    elapsed = time.perf_counter() - start
    assert len(out) == x.size + m - 1
    assert elapsed < 0.8, f"{elapsed * 1000:.0f} ms exceeds 800 ms"


@pytest.mark.parametrize(
    "factory",
    [
        lambda: reverb.Reverb([]),
        lambda: reverb.Reverb(None),
        lambda: reverb.Reverb([1.0, float("nan")]),
        lambda: reverb.Reverb([1.0, float("inf")]),
        lambda: reverb.Reverb([0.0, 0.0]),
        lambda: reverb.Reverb([[1.0, 2.0]]),
        lambda: reverb.Reverb("abc"),
        lambda: reverb.Reverb(5.0),
        lambda: reverb.Reverb([1.0]).process([float("nan")]),
        lambda: reverb.Reverb([1.0]).process([float("inf")]),
        lambda: reverb.Reverb([1.0]).process(None),
        lambda: reverb.Reverb([1.0]).process("ab"),
    ],
)
def test_invalid_parameters_raise(factory):
    with pytest.raises(ValueError):
        factory()


def test_returns_list():
    assert isinstance(reverb.Reverb([0.5, 0.25]).process([1.0]), list)


def test_concurrent_instances_are_independent():
    rng = np.random.default_rng(23)
    ir = rng.standard_normal(300)
    ref = {}
    inputs = {s: np.random.default_rng(s).standard_normal(5000) for s in range(8)}
    for s, x in inputs.items():
        ref[s] = _direct(x, ir)

    errors = []

    def worker(s):
        got = np.asarray(reverb.Reverb(ir).process(inputs[s]))
        errors.append(float(np.max(np.abs(got - ref[s]))))

    threads = [threading.Thread(target=worker, args=(s,)) for s in inputs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max(errors) <= TOL
