"""reverb 核心 API：基于 FFT 分块重叠保留（overlap-save）的卷积混响。

约定
----
- IR 原样使用，不做增益归一化：对单位冲激输入，输出恰为 IR 本身
  （因此其能量与 IR 的能量一致）；调用方若需要单位能量 IR，自行归一化。
- ``process(buffer)`` 返回 ``list``，长度为 ``len(buffer) + len(IR) - 1``
  （输入为空时返回 ``[]``）：前 ``len(buffer)`` 个样本是本缓冲区的卷积
  结果，其余 ``len(IR) - 1`` 个是尚未耗尽的混响尾部样本，总拖尾跨度
  覆盖完整的 IR 长度。
- 同一实例跨多次 ``process`` 调用保留滤波器历史，因此把信号拆成多块
  依次处理，与一次性整段处理逐样本一致（每块只取前 ``len(chunk)`` 个
  输出再拼接即为完整卷积），即"跨缓冲连续"。
- 非有限值（IR 或样本中的 NaN/Inf）与非法参数（空 IR、非一维、非数值、
  零能量 IR 等）统一抛出 ``ValueError``。
- 实例状态只在该实例的 ``process`` 内读写，并用锁串行化同一实例的并发
  调用；不同实例之间没有共享的可变状态，因此并发安全。
"""

from __future__ import annotations

import threading

import numpy as np

__all__ = ["Reverb"]

# 每个 overlap-save 分块希望包含的有效样本数（决定 hop）。
# 取值需在 FFT 开销与分块数量之间折中：1e6 样本在 ~4k hop 下仅需数百块。
_TARGET_HOP = 4096


def _is_sequence_like(value):
    try:
        iter(value)
    except TypeError:
        return False
    return True


def _validate_ir(ir):
    """把 IR 转为只读的一维 float64 数组，非法时抛 ``ValueError``。"""
    if ir is None or not _is_sequence_like(ir):
        raise ValueError("ir must be a non-empty sequence of real numbers")

    if isinstance(ir, np.ndarray):
        if np.iscomplexobj(ir):
            raise ValueError("ir must contain only real numbers")
        try:
            arr = ir.astype(np.float64, copy=True)
        except (TypeError, ValueError):
            raise ValueError("ir must contain only real numbers") from None
    else:
        try:
            arr = np.asarray(list(ir), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("ir must contain only real numbers") from None

    if arr.ndim != 1:
        raise ValueError("ir must be one-dimensional")
    if arr.size == 0:
        raise ValueError("ir must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("ir must contain only finite values")
    if not np.any(arr):
        raise ValueError("ir must not be all zeros")

    arr.setflags(write=False)
    return arr


def _validate_samples(samples):
    """把输入缓冲转为一维 float64 数组，非法时抛 ``ValueError``。"""
    if samples is None or isinstance(samples, (str, bytes, bytearray)):
        raise ValueError("samples must be a sequence of finite real numbers")

    if isinstance(samples, np.ndarray):
        if np.iscomplexobj(samples):
            raise ValueError("samples must contain only real numbers")
        try:
            arr = samples.astype(np.float64, copy=False)
        except (TypeError, ValueError):
            raise ValueError("samples must contain only real numbers") from None
    else:
        if not _is_sequence_like(samples):
            raise ValueError("samples must be a sequence of finite real numbers")
        try:
            arr = np.asarray(list(samples), dtype=np.float64)
        except (TypeError, ValueError):
            raise ValueError("samples must contain only real numbers") from None

    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("samples must contain only finite values")
    if arr.ndim != 1:
        raise ValueError("samples must be one-dimensional")
    return arr


class Reverb:
    """分块重叠保留（overlap-save）卷积混响器。"""

    def __init__(self, ir):
        ir_arr = _validate_ir(ir)
        self._m = int(ir_arr.size)

        # 每个分块至少携带 _TARGET_HOP 个有效样本，大 IR 时与 IR 同阶：
        # FFT 尺寸取 2 的幂，容纳 m-1 个历史样本 + 至少 max(_TARGET_HOP, m)
        # 个新样本，避免 IR 接近 2 的幂时 hop 过小导致分块数暴涨。
        target = max(_TARGET_HOP, self._m)
        nfft = 1
        while nfft < self._m - 1 + target:
            nfft <<= 1
        self._nfft = nfft
        self._hop = nfft - self._m + 1  # 每块产出的有效样本数

        # 构造后只读，可被各线程安全共享。
        self._ir_fft = np.fft.rfft(ir_arr, nfft)
        # 输入历史（最近 m-1 个已处理样本），保证跨缓冲区连续。
        self._hist = np.zeros(self._m - 1, dtype=np.float64)

        # 同一实例的并发 process 串行化；跨实例无共享状态。
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ API
    def process(self, samples):
        """处理一个缓冲区，返回 ``list``。"""
        data = _validate_samples(samples)
        with self._lock:
            return self._process_locked(data)

    # ------------------------------------------------------------- internals
    def _process_locked(self, data):
        m = self._m
        hist_len = m - 1

        if data.size == 0:
            # 空缓冲区不产生输出，也不扰动滤波器状态。
            return []

        n_in = int(data.size)
        nfft = self._nfft
        hop = self._hop
        ir_fft = self._ir_fft

        # 补零设计：前端接上一缓冲区的输入历史，后端补 hist_len 个零，
        # 使补零输入产生的混响尾部也落在输出范围内。
        padded = np.concatenate(
            (self._hist, data, np.zeros(hist_len, dtype=np.float64))
        )
        n_blocks = (padded.size + hop - 1) // hop
        if padded.size < n_blocks * hop + hist_len:
            padded = np.concatenate(
                (
                    padded,
                    np.zeros(
                        n_blocks * hop + hist_len - padded.size, dtype=np.float64
                    ),
                )
            )

        # 每个 overlap-save 块产出 hop 个有效样本；输出下标与 padded 对齐。
        out = np.empty(n_blocks * hop, dtype=np.float64)
        for k in range(n_blocks):
            start = k * hop
            block = padded[start : start + hop + hist_len]
            spec = np.fft.rfft(block, nfft)
            spec *= ir_fft
            conv = np.fft.irfft(spec, nfft)
            # 每块前 m-1 个样本是循环卷积的重叠段，丢弃。
            out[start : start + hop] = conv[hist_len:]

        # 更新输入历史（只取真实输入，不含补零）。
        if n_in >= hist_len:
            self._hist = data[n_in - hist_len :].copy()
        else:
            self._hist = np.concatenate((self._hist[n_in:], data))

        # overlap-save 丢弃每块前 hist_len 个重叠样本后，out[0] 正对应
        # padded[hist_len]，即本缓冲区的第一个输入样本。因此本缓冲区结果
        # 加上其后 hist_len 个尾部样本就是 out 的前 n_in + hist_len 个值。
        return [float(v) for v in out[: n_in + hist_len]]
