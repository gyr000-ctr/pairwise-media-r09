# reverb

## 题面（可直接粘贴）

卷积混响。约束：与直接卷积对照误差 ≤1e-9；IR 归一化后能量守恒；尾部长度等于 IR 长度；跨缓冲连续；100 万样本不超过 800 ms；非法参数报错。验收：cd reverb && python -m pytest tests -q -k reverb。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd reverb && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：音视频处理
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
