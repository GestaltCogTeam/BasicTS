# 贡献指引

## 简介

本文将指引您如何一同建设 BasicTS！

敏感信息和安全漏洞等信息请不要公开发布于issue等位置，请直接将相关信息发送到duyifan at ict.ac.cn。

我们推荐使用英文进行交流。

## Issues

### 如何报告 issues

请在issues中使用多种相关的关键词搜索您的问题，以确保您的问题尚未被报告。

如果您的问题确实尚未被报告，请在这里[提交一个issue](https://github.com/GestaltCogTeam/BasicTS/issues/new/choose)，提交时请选择合适的issues类型以便进行分类管理。

请提供清晰且简明的说明，即使步骤很简单，也请确保我们能够复现该行为。

### issues 分类

目前提供的issues 分类如下:

* `bug report`：提交Bug或者您遇到的错误信息
* `feature or enhancement request`：请求新功能特性。请详细的描述您希望在BasicTS中使用的功能及其使用场景，使任何阅读该issue的人都能理解其应如何实现这个功能。
* `security issue`：请不要在公开issue中讨论安全类问题，安全问题请直接发送相关信息至duyifan at ict.ac.cn。

## 提交您的贡献

1. 在提交代码变更之前，请先在[issues](https://github.com/GestaltCogTeam/BasicTS/issues)中充分讨论您的想法，以确保其必要性和可行性。
2. Fork 此仓库并为您的工作创建一个新分支。
3. 进行开发时，请添加清晰的代码注释以解释您的方法，并尽量遵循现有的代码约定。
4. 请遵循以下的[代码格式检查](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/contribution_guideline_cn.md#%E4%BB%A3%E7%A0%81%E6%A0%BC%E5%BC%8F%E6%A3%80%E6%9F%A5)指南。
5. 提交一个指向 `main` 分支的 PR（Pull Request），并关联与之相关的issue。请提供详细背景信息，并遵循以下的[PR格式指南](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/contribution_guideline_cn.md#%E4%BB%A3%E7%A0%81%E6%A0%BC%E5%BC%8F%E6%A3%80%E6%9F%A5)。

我们会尽快评估 PR，并与您协作整合您的贡献。请耐心等待，因为评估PR需要时间。一旦通过，您的代码将被合并！

### 代码格式检查

请注意：由于baseline部分可能包含一些外部实现，因此这部分并不强制进行代码检查，但也请尽量保持代码的高效整洁。

项目使用 `Pylint`进行代码格式检查，并使用 `isort`对代码依赖部分进行自动排序。为保证代码一致性，请遵循以下步骤：

1. 安装 [Pylint](https://pylint.org/#install) 和 [isort](https://pycqa.github.io/isort/index.html#installing-isort)。
2. 在项目根目录[运行pyline](https://docs.pylint.org/run.html)进行格式检查。（pylintrc位于项目根目录中）
3. 在项目根目录[运行isort](https://pycqa.github.io/isort/index.html#using-isort)进行依赖排序。（isort.cfg位于项目根目录中）

### PR格式指南

请尽量写明您的 PR 以便于评估。

1. 请保持PR 专注于事前讨论好的代码，不包含其他的无关部分。
2. 请允许维护者编辑您的 PR。如此维护者可以在之后负责合并 PR，而无需您操作。
3. PR 标题请使用 `<mark> <title>` 的格式。可用的mark如下：

| mark       | 含义         |
| ---------- | ------------ |
| docs: ✏️ | 文档变更     |
| fix: 🐛    | bug修复      |
| tests: 📏  | 测试         |
| feat: 🎸   | 新功能特性   |
| chore: 🤖  | 项目杂项     |
| style: 💄  | 代码风格维护 |
