# GitHub 归档说明

本项目采用“Git 仓库 + GitHub Release 全量快照”两层归档。

## Git 仓库包含什么

仓库保存源代码、配置、脚本、Markdown 文档、点云地图、模型描述、论文材料和其他适合进行
版本管理的文件。以下本地生成目录不直接进入 Git 历史：

- `results/`：训练权重、检查点与评估明细；
- `build/`、`install/`、`log/`、`build_context_v2/`：ROS 2 构建产物和日志；
- Python 与编辑器缓存。

这样可以保持仓库可正常克隆和查看。上述目录并未丢失，它们保存在同版本 GitHub Release
的全量分卷归档中。

## 全量 Release 快照

Release 标签：`workspace-snapshot-2026-08-23`

归档文件名格式：

```text
work-full-20260823.tar.part-aa
work-full-20260823.tar.part-ab
...
work-full-20260823.SHA256SUMS
```

每个分卷小于 GitHub Release 的 2 GiB 单文件限制。SHA-256 清单必须与所有分卷一起下载。

## macOS / Linux 恢复方式

将所有分卷和 `work-full-20260823.SHA256SUMS` 放在同一目录：

```bash
shasum -a 256 -c work-full-20260823.SHA256SUMS
mkdir work-restored
cat work-full-20260823.tar.part-* | tar -xf - -C work-restored
```

恢复后得到创建快照时的完整 `work` 工作区内容，包括训练结果、权重、构建产物和日志。
快照不包含 `.git` 目录；如需 Git 历史，请单独克隆仓库，再按需要把 Release 内容恢复到
克隆目录。

## 归档边界

- 快照日期：2026-08-23；
- 快照在本目录首次 `git init` 之前创建，因此不包含 Git 内部对象；
- GitHub 仓库设为 private；
- 未在工作区发现密码、GitHub token、AWS key 或私钥文件；
- 论文 PDF 和实验模型仅用于该私有科研项目归档，不应把仓库直接改成 public。
