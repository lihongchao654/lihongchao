# 主震模拟

本仓库包含用于人工模拟地震动和相关脚本与示例数据。

## 目录（部分）
- `人工模拟地震动.py` / `人工模拟地震动_纯Python版.py`：模拟脚本
- `地震动加速度谱.csv`、`地震事件汇总.csv`：示例数据
- `run.bat` / `run.ps1`：示例运行脚本

## 快速开始

在 Windows 上运行（PowerShell）：

```powershell
cd "C:\\Users\\李鸿超的小本本\\Desktop\\仓库"
# 用 bat
.\\run.bat
# 或用 PowerShell 脚本
.\\run.ps1
```

如果你使用 Python 环境，建议创建虚拟环境并安装依赖（若仓库包含 `requirements.txt`）：

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

## 提交与协作
- 提交时可以使用 `git commit -m "message"` 跳过编辑器。
- 编辑器已配置为 `code --wait`，将使用 VS Code 编辑提交信息并等待关闭后继续。

## 联系
如需帮助，请联系：`lihongchao@stu.xju.edu.cn`
