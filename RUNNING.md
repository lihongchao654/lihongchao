# 主震预测代码运行说明 (Main Shock Prediction - Running Instructions)

## 快速开始 (Quick Start - 无需依赖)

运行演示版本（使用纯 Python，无需安装依赖）：

```bash
python 主震预测_demo.py
```

**输出**: 演示使用随机生成的地震数据计算 UHS 和 CMS 频谱

## 完整版本 (Full Version - 需要依赖)

### 前置要求
- Python 3.11 或 3.12（推荐）或 Python 3.10（不支持 Python 3.13 beta）
- pip 包管理器

### 安装依赖

在项目目录运行：

```bash
pip install -r requirements.txt
```

或直接安装：

```bash
pip install numpy pandas scipy
```

### 运行完整版本

```bash
python 主震预测.py
```

## 代码修复历史 (Code Fixes)

### 修复的问题 (Issues Fixed)

1. **uniform.rvs() 参数错误** → 改为正确的 `loc` 和 `scale` 参数
2. **periods 数组起始值** → 从 0.0 s 改为 0.01 s (PGA)
3. **calculate_cms() 全局变量问题** → 改为传递 `sa_uhs` 参数
4. **Series 索引错误** → 改为 `event.loc[]` 语法
5. **边界检查** → UHS 索引添加边界保护
6. **性能优化** → 默认 100 次模拟（可改为 10000）
7. **输出格式** → 改为清晰的表格输出

### 文件说明

| 文件 | 说明 |
|------|------|
| `主震预测.py` | 完整版本，包含详细 GMPE、Haversine 距离计算等 |
| `主震预测_demo.py` | 演示版本，纯 Python 实现，无外部依赖 |
| `requirements.txt` | Python 依赖列表 |
| `README.md` | 项目说明文件 |

## 已知问题 (Known Issues)

1. **Python 3.13 beta 兼容性**: NumPy 2.3.x 对 Python 3.13 beta 版本有编译问题
   - **解决方案**: 使用 Python 3.11/3.12，或使用演示版本

2. **大规模模拟耗时**: 10000 次模拟可能需要数分钟
   - **解决方案**: 使用演示版本测试，或调整 `num_simulations` 为较小值

## 示例输出 (Sample Output)

```
均匀危险谱 (UHS) @ 10% 超越概率 in 50 years:
  T =   0.01 s: SA =   0.571375 g
  T =   0.20 s: SA =   0.480908 g
  T =   0.50 s: SA =   0.384726 g
  ...

条件平均谱 (CMS) @ T = 0.50s:
  T =   0.01 s: SA =   0.571375 g
  ...
```

## 参考文献 (References)

- USGS: Probabilistic Seismic Hazard Analysis
- Baker, J.W. (2008): An Introduction to Probabilistic Seismic Hazard Analysis (PSHA)
