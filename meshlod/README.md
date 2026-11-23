# meshlod

## 题面（可直接粘贴）

网格 LOD 简化误差。约束：简化后 Hausdorff 距离不超过阈值；面积误差 ≤5%；包围盒不变；目标面数可控；1 万面不超过 600 ms；非法输入报错。验收：cd meshlod && python -m pytest tests -q -k lod。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd meshlod && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
