# tileblend

## 题面（可直接粘贴）

瓦片过渡混合。约束：过渡带内权重和为 1 且线性；边界外取单一瓦片；同层多瓦片互不影响；10 万次不超过 200 ms；非法参数报错；并发安全。验收：cd tileblend && python -m pytest tests -q -k blend。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd tileblend && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
