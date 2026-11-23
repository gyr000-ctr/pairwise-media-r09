"""cubemap 核心 API：方向 <-> 面/UV 换算。

约定（OpenGL 立方体贴图，跨面接缝连续）：
    面 0:+X  1:-X  2:+Y  3:-Y  4:+Z  5:-Z
    u = sc/|ma|, v = tc/|ma|，ma 为最大绝对分量。
面选择按最大绝对分量，平局规则确定：x 优先于 y，y 优先于 z；
同轴正负由分量符号决定。CubeMap 无状态，方法可并发调用。
"""

import math as _math
import sys as _sys

__all__ = [
    "CubeMap",
    "CubeMapError",
    "ZeroDirectionError",
    "NonFiniteDirectionError",
    "InvalidFaceError",
    "InvalidUVError",
]

_FLOAT_MAX = _sys.float_info.max
_sqrt = _math.sqrt


class CubeMapError(ValueError):
    """cubemap 参数错误基类（兼容既有的 ValueError 捕获）。"""


class ZeroDirectionError(CubeMapError):
    """零向量方向。"""


class NonFiniteDirectionError(CubeMapError):
    """方向含 NaN/Inf。"""


class InvalidFaceError(CubeMapError):
    """非法面索引。"""


class InvalidUVError(CubeMapError):
    """非法 UV（非有限值）。"""


class CubeMap:
    """立方体映射换算器。无状态、线程安全。"""

    __slots__ = ()

    def to_face_uv(self, d):
        """方向到面/UV。返回 (face, u, v)，u/v ∈ [-1, 1]。"""
        x, y, z = d
        ax = abs(x)
        ay = abs(y)
        az = abs(z)
        if not (ax <= _FLOAT_MAX and ay <= _FLOAT_MAX and az <= _FLOAT_MAX):
            raise NonFiniteDirectionError(
                "direction contains NaN or Inf: %r" % (d,)
            )
        # 最大分量选择；平局时 x 优先于 y、y 优先于 z（确定性规则）。
        if ax >= ay and ax >= az:
            axis = 0
            m = ax
        elif ay >= az:
            axis = 1
            m = ay
        else:
            axis = 2
            m = az
        if m == 0.0:
            raise ZeroDirectionError("zero direction vector: %r" % (d,))
        if axis == 0:
            if x > 0.0:
                return (0, -z / m, -y / m)
            return (1, z / m, -y / m)
        if axis == 1:
            if y > 0.0:
                return (2, x / m, z / m)
            return (3, x / m, -z / m)
        if z > 0.0:
            return (4, x / m, -y / m)
        return (5, -x / m, -y / m)

    def to_dir(self, face, u, v):
        """面/UV 到方向。返回单位向量 (x, y, z)。"""
        if not (abs(u) <= _FLOAT_MAX and abs(v) <= _FLOAT_MAX):
            raise InvalidUVError("non-finite uv: (%r, %r)" % (u, v))
        if face == 0:
            x = 1.0
            y = -v
            z = -u
        elif face == 1:
            x = -1.0
            y = -v
            z = u
        elif face == 2:
            x = u
            y = 1.0
            z = v
        elif face == 3:
            x = u
            y = -1.0
            z = -v
        elif face == 4:
            x = u
            y = -v
            z = 1.0
        elif face == 5:
            x = -u
            y = -v
            z = -1.0
        else:
            raise InvalidFaceError("invalid face index: %r" % (face,))
        n = _sqrt(x * x + y * y + z * z)
        return (x / n, y / n, z / n)
