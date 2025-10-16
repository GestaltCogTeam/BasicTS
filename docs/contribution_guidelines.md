# Contribution Guidelines

## Introduction

This document explains how to contribute changes to the BasicTS project.
Sensitive security-related issues should be reported to duyifan at ict.ac.cn.

## Issues

### How to report issues

Please search the issues on the issue tracker with a variety of related keywords to ensure that your issue has not already been reported.

If your issue has not been reported yet, [open an issue](https://github.com/GestaltCogTeam/BasicTS/issues/new/choose) and choose a issue type to report.
Please write clear and concise instructions so that we can reproduce the behavior — even if it seems obvious.

### Types of issues

Typically, issues fall in one of the following categories:

* `bug report`: Report errors or unexpected behavior.
* `feature or enhancement request`: Propose something new. You should describe this feature in enough detail that anyone who reads the issue can understand how it is supposed to be implemented.
* `security issue`:  Please do not file such issues on the public tracker and send a mail to duyifan at ict.ac.cn instead.

## Contribution

1. Before taking on significant code changes, please discuss your ideas on [issues](https://github.com/GestaltCogTeam/BasicTS/issues) to ensure the necessity and feasibility.
2. Fork the repository and create a new branch for your work.
3. Make changes with clear code comments explaining your approach. Try to follow existing conventions in the code.
4. Follow the [Code Formatting and Linting](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/contribution_guidelines.md#code-formatting-and-linting) guide below.
5. Open a PR into `main` linking any related issues. Provide detailed context on your changes. Please follow the [Pull request format](https://github.com/GestaltCogTeam/BasicTS/blob/master/tutorial/contribution_guidelines.md#pull-request-format) guide below.

We will review PRs when possible and work with you to integrate your contribution. Please be patient as reviews take time. Once approved, your code will be merged.

### Code Formatting and Linting

NOTE:  Since the baseline part may contain some external code, this part does not require mandatory code formatting and linting.

We use `Pylint` for code formatting and `isort` for import sorting . To ensure consistency across contributions, please adhere to the following steps:

1. Install [Pylint](https://pylint.org/#install)  and [isort ](https://pycqa.github.io/isort/index.html#installing-isort)if needed.
2. [Run pyline](https://docs.pylint.org/run.html) to analyze your code in the project root directory (cause the pylintrc is here).
3. [Run isort](https://pycqa.github.io/isort/index.html#using-isort) to sort the import part in the project root directory (cause the isort.cfg is here).

### Pull request format

Please try to make your pull request easy to review for us.

1. Don't make changes unrelated to your PR. Make it simple.
2. Allow edits by maintainers. This way, the maintainers will take care of merging the PR later on instead of you.
3. PR title please use `<mark> <title>` format. Available marks are here:

| mark       | meaning                     |
| ---------- | --------------------------- |
| docs: ✏️ | document                    |
| fix: 🐛    | bug fix                     |
| tests: 📏  | tests                       |
| feat: 🎸   | features or enhancements    |
| chore: 🤖  | chores of project           |
| style: 💄  | changes releated code style |
