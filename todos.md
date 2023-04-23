# BasicTS v0.2 版本开发计划

## Format

- 梳理BasicTS的功能和意义
    - 梳理一个标准的DL模型的训练/验证/测试pipeline。（用户并不需要理解这个pipeline是如何构建的，因为这需要讲解easytorch。用户只需要学会这个pipeline是如何使用以及扩展、修改的就可以了。）
    - BasicTS的目标：标准化这样的pipeline，用户只需要专注于模型设计。
    - 另外，还提供灵活且丰富的接口，支撑灵活的模型验证。
 
- 梳理Code Base Convention
    - 代码库的组织结构。
    - 代码库的约定习惯。
    - 如何修改/扩展标准的pipeline。
    - 如何进行模型验证。    

- 清理不符合规范的代码
    - 代码注释。 ✅
    - 统一数据预处理的结果。 ✅
    - 清理Runner。 ✅
        1. 梳理时间序列预测的Pipeline，并标注对应的CFG选项、接口设计
        2. 修改Scripts
            2.1 添加Norm Each Channel选项 ✅
            2.2 统一Temporal Feature的格式 ✅
        3. 修改arch
            3.1 修改Xformer的数据处理 ✅
        4. 修改runner
            4.1 NULL Value提到General Parameters里面去 ✅
            4.2 添加Rescale选项（General Parameters） ✅
            4.3 Evaluate单独写一个函数 ✅
            4.4 ReScale单独写一个函数 ✅
            4.5 整理Setup Graph，使其更合理，并使其推理脚本更加简洁易懂 ✅
            4.6 SETUP_Graph 选项应该放在MODEL里面 ✅
            4.7 Format一下CFG.TRAIN和CFG.get("TRAIN")以及cfg["TRAIN] ✅
            4.8 XFormer的Metrics和Loss需要适配 ✅
        5. 删除没有特殊定义的Runner ✅

- 添加icon，并整理README。 ✅

## New Feature

- 支持独立加载模型。✅
- 支持更多的Runner内部函数的解耦操作，从而支持更灵活的模型验证。✅
- 支持自定义是否re-scale。✅
- 将ETT数据集按照现有工作的习惯划分重新处理。 ✅

## Documentation

- Getting Started
    - 安装
    - 快速上手训练
    - 配置文件修改
    - 模型验证
    - 内置样例库
    - Benchmark
- Documentation
    - Code Base Convention
        - 代码库的组织结构。
        - 代码库的约定习惯。
        - 如何修改/扩展标准的pipeline。
        - 如何进行模型验证。
    - 数据结构和约定
    - 模型接口约定
    - Runner接口约定
    - Loss接口约定
    - Metric接口约定
    - 工具函数utils介绍
-  API Reference
    - Runner
    - Model
    - Loss
    - Metric
    - Utils
- How to (Tutorials)
    - 如何可视化数据和分析数据
    - 如何加载训练好的模型，并进行后续可视化分析（基于Jupyter）
    - 预测结构可视化和分析
    - 新手教程
        - 任务介绍
        - 数据介绍
        - 模型介绍
- Additional Information
    - FAQ
    - Contributing
    - License
    - Citation
    - Acknowledgement
