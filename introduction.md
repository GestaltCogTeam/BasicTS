### Overall

**BasicTSRunner**

**一个runner，三个入口：**
- train
- eval (_test，即此前的test_pipeline，不希望被用户直接调用，而是被封装进eval内部)
- validate (基本用不到)

**两个loop：**
- train_loop：训练流程
- eval_loop：验证、测试流程，由于测试阶段不再需要concat，两个流程可以共用绝大部分代码。

**pipeline：**
以train_loop为例，
1. runner on_step_start
2. 用户（callback）on_step_start
3. Taskflow 前处理
4. runner forward
5. 用户 on_compute_loss (可以定制化修改forward_return以便计算loss)
6. runner计算loss (TODO:支持在forward_return中直接给loss)
7. runner backward (TODO:需要提供更多hook才能支持梯度累积等操作，如on_optimizer_step)
8. Taskflow 后处理（准备计算指标）
9. runner计算指标
10. 用户 on_step_end
11. runner on_step_end

**数据格式**

数据应该有固定的格式:
```python
{
    'inputs': torch.Tensor,
    'targets': torch.Tensor, [Optional]
    'inputs_timestamps': torch.Tensor, [Optional]
    'targets_timestamps': torch.Tensor [Optional]
}
```

### Design idea

各个任务需要的特殊操作：（point-wise权重指需要按有效点加权、sample-wise指需要按batch大小加权）

|          | 前处理                 | 前传             | 计算loss        | 计算指标前       | 计算指标        |
| -------- | ---------------------- | ---------------- | --------------- | ---------------- | --------------- |
| 预测     | 归一化                 | inputs、targets   | point-wise权重  | 反归一化（可选） | point-wise权重  |
| 分类     | 归一化                 | inputs、targets   | sample-wise权重 | argmax           | sample-wise权重 |
| 插补     | 先mask再归一化         | 可能额外多传mask | point-wise权重  | 反归一化（可选） | point-wise权重  |
| 异常检测 | 先mask（可选）再归一化 | 可能额外多传mask | point-wise权重  | 根据残差计算Topk | point-wise权重 |

训练流程、任务、策略解耦成三个层次：
1. Runner负责整体pipline，与数据、任务和策略无关。
2. Taskflow负责与任务相关的处理，需要实现三个方法：preprocess、postprocess、get_weight
3. Callback负责与策略相关的设计，如课程学习、早停、梯度累计等，需要实现相应位置的hook函数

这样的设计下，90%的用户只用接触runner，30%的用户只用接触runner和callback，只有非常少部分的用户才会深入到taskflow逻辑。