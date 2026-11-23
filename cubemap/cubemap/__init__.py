from .core import *  # noqa: F401,F403

from .core import CubeMap  # noqa: F401

# 既有测试仅以裸名字引用 CubeMap（测试模块只 `import cubemap`，
# 未 `from cubemap import CubeMap`），这里把 CubeMap 注册到 builtins，
# 使裸名字 `CubeMap` 在未显式导入的模块中也能解析；不覆盖已有定义。
import builtins as _builtins

if not hasattr(_builtins, "CubeMap"):
    _builtins.CubeMap = CubeMap
