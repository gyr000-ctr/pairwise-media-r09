"""tileblend 核心 API。

瓦片过渡混合：过渡带内权重双线性连续且和为 1，过渡带外取单一瓦片。
TileBlend 无状态（__slots__，无可变成员），blend 为纯函数，天然并发安全。
"""

import math

__all__ = ["TileBlend"]


def _axis_weight(u, wn):
    """一维过渡权重：返回“上侧”瓦片的线性斜坡权重 a ∈ [0, 1]。

    斜坡以瓦片边界为中心、总宽 wn（归一化），边界处 a=0.5，
    两侧分别线性过渡到 0 / 1，保证跨边界连续。
    """
    t = (u + 0.5) % 1.0
    a = (t - 0.5) / wn + 0.5
    if a <= 0.0:
        return 0.0
    if a >= 1.0:
        return 1.0
    return a


class TileBlend:
    """瓦片混合权重计算器（无状态，线程安全）。"""

    __slots__ = ()

    def blend(self, x, y, tile_size, width):
        """返回 4 元组 (w00, w10, w01, w11)：采样点周围四个瓦片的双线性权重。

        权重和恒为 1；过渡带外退化为单一瓦片（某一分量为 1，其余为 0）。
        """
        try:
            xf = float(x)
            yf = float(y)
            ts = float(tile_size)
            wd = float(width)
        except (TypeError, ValueError) as exc:
            raise TypeError("blend 参数必须是数值类型 (numeric required): %r" % (exc,)) from exc
        if not (math.isfinite(xf) and math.isfinite(yf)):
            raise ValueError("x/y 必须是有限数值 (finite required)")
        if not math.isfinite(ts) or ts <= 0.0:
            raise ValueError("tile_size 必须是正的有限数值 (positive finite), got %r" % (tile_size,))
        if not math.isfinite(wd) or wd <= 0.0:
            raise ValueError("width 必须是正的有限数值 (positive finite), got %r" % (width,))
        wn = wd / ts
        if wn > 1.0:
            wn = 1.0  # 过渡带宽于瓦片时钳制，斜坡仍连续
        ax = _axis_weight(xf / ts, wn)
        ay = _axis_weight(yf / ts, wn)
        bx = 1.0 - ax
        by = 1.0 - ay
        return (bx * by, ax * by, bx * ay, ax * ay)
