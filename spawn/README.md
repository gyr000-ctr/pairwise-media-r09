# spawn

## 题面（可直接粘贴）

刷怪点密度场。约束：采样点密度与权重场成正比（卡方检验 ≤5%）；同种子可复现；边界不越界；1 万点不超过 400 ms；非法权重报错。验收：cd spawn && python -m pytest tests -q -k spawn。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd spawn && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
