# cubemap

## 题面（可直接粘贴）

立方体映射方向换算。约束：方向→面+UV 与反向换算往返误差 ≤1e-9；面选择与最大分量一致；接缝处连续；6 个面覆盖全方向；10 万次不超过 200 ms；非法方向报错。验收：cd cubemap && python -m pytest tests -q -k cube。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd cubemap && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
