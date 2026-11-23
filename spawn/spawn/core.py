"""spawn 核心 API：按权重密度场采样刷怪点。

采样点落在二维权重网格的格子上，每个格子被抽中的概率与其权重成正比
（有放回抽样）。同一种子产生完全一致的采样序列；每次调用使用独立的
随机数生成器，实例无共享可变状态，因此并发调用安全。
"""

import math
import random
from bisect import bisect_right


class Spawn:
    """密度场采样器。

    Parameters
    ----------
    seed:
        随机种子，原样保存。同一种子 + 同一入参的 :meth:`sample` 调用
        总是返回相同结果。
    """

    def __init__(self, seed):
        self.seed = seed

    def sample(self, weights, n):
        """按权重密度场采样 ``n`` 个刷怪点。

        Parameters
        ----------
        weights:
            非空矩形二维网格（行优先的列表/元组），元素为非负有限数值。
            至少有一个正权重。
        n:
            采样点数，正整数。

        Returns
        -------
        list[tuple[int, int]]
            ``(行, 列)`` 坐标列表，长度恰为 ``n``，坐标恒在网格范围内。

        Raises
        ------
        TypeError
            权重容器结构类型不合法（非二维列表/元组、元素非数值）。
        ValueError
            入参取值不合法（空网格、行列不齐、NaN/Inf、负权重、
            权重总和为 0、``n`` 非正整数等）。
        """
        cdf, ncols, total = _build_cdf(weights)
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError("n 必须是正整数")

        # 每次调用独立 RNG：无共享可变状态，天然可重入、并发安全，
        # 同时保证同种子 + 同入参可逐点复现。
        rng = random.Random(self.seed)
        random_ = rng.random
        bisect = bisect_right
        last = len(cdf) - 1

        points = []
        for _ in range(n):
            # 钳制：极大 total 下 u*total 可能舍入到 total 本身，
            # bisect_right 会越界，此时落到最后一个正权重格子。
            idx = bisect(cdf, random_() * total)
            if idx > last:
                idx = last
            points.append(divmod(idx, ncols))
        return points


def _build_cdf(weights):
    """校验权重网格并构建累积分布（CDF）。

    返回 ``(cdf, 列数, 权重总和)``；CDF 中零权重格子表现为平台，
    配合 ``bisect_right`` 保证永远不会被抽中。
    """
    if not isinstance(weights, (list, tuple)):
        raise TypeError("weights 必须是二维列表/元组")
    if len(weights) == 0:
        raise ValueError("weights 不能为空网格")

    ncols = None
    cdf = []
    total = 0.0

    for row in weights:
        if not isinstance(row, (list, tuple)):
            raise TypeError("weights 的每一行必须是列表/元组")
        if len(row) == 0:
            raise ValueError("weights 不能包含空行")
        if ncols is None:
            ncols = len(row)
        elif len(row) != ncols:
            raise ValueError("weights 各行长度必须一致（矩形网格）")

        for w in row:
            if not isinstance(w, (int, float)) or isinstance(w, bool):
                raise TypeError("权重必须是数值（int 或 float）")
            w = float(w)
            if not math.isfinite(w):
                raise ValueError("权重必须是有限值，不接受 NaN/Inf")
            if w < 0.0:
                raise ValueError("权重不能为负数")
            total += w
            cdf.append(total)

    if total <= 0.0:
        raise ValueError("权重总和必须大于 0")
    if not math.isfinite(total):
        raise ValueError("权重总和非有限（溢出），请缩小权重尺度")

    return cdf, ncols, total
