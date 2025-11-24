# 地震模拟系统 - 使用指南

## 📋 项目说明

本项目包含完整的**地震蒙特卡洛事件模拟 + 地震动波形模拟系统**。

### 文件结构

| 文件 | 说明 |
|------|------|
| `人工模拟地震动_纯Python版.py` | ✅ **推荐使用** - 纯 Python 版本，无需科学计算库编译 |
| `人工模拟地震动.py` | 原始完整版本（需 NumPy/SciPy/Matplotlib） |
| `setup_env_with_python311.ps1` | PowerShell 环境配置脚本 |
| `README.md` | 本文件 |

---

## 🚀 快速开始

### 推荐方案：使用纯 Python 版本

**优点**
- ✅ 无需安装 C 编译器
- ✅ 无依赖版本冲突
- ✅ 立即可运行
- ✅ 完整的地震模拟功能

**运行步骤**

在 PowerShell 中执行：

```powershell
# 进入项目目录
cd d:\opensees\1

# 激活虚拟环境（如已激活则跳过）
.\.venv\Scripts\Activate.ps1

# 运行纯 Python 版本
python 人工模拟地震动_纯Python版.py
```

**预期输出**
```
===========================================================
纯 Python 版地震模拟系统 - 演示版本
（无需 NumPy/SciPy 编译依赖）
===========================================================

[1] 初始化研究区域...
[2] 创建地震蒙特卡洛模拟器...
[3] 设置地震统计参数...
[4] 运行蒙特卡洛地震事件模拟...
[5] 获取统计摘要...

模拟结果统计
===========================================================
总模拟次数: 3
每次模拟平均事件数: 3.33 ± 2.62
...
```

---

## 🔧 功能说明

### 地震蒙特卡洛模拟

基于**平稳泊松模型**的地震事件模拟：

- **震级分布**：古腾堡-里希特（Gutenberg-Richter）关系
- **深度分布**：对数正态分布
- **空间分布**：均匀分布（可扩展为高斯分布）
- **发生时间**：泊松过程

### 地震动模型

**点源模型**（AB95 双拐角频率）：

- 位移谱计算
- 路径效应（几何衰减、吸收）
- 场地效应（Kappa 衰减）
- 加速度谱输出

### 统计分析

自动生成：
- 震级统计（平均、标准差、范围）
- 深度统计
- 发生率统计
- 模型使用统计

---

## 📊 示例：自定义模拟

```python
from 人工模拟地震动_纯Python版 import SeismicMonteCarloSimple

# 1. 定义研究区域
region = {
    'lon_min': -90.5, 'lon_max': -89.5,
    'lat_min': 35.0, 'lat_max': 35.5,
    'depth_min': 5, 'depth_max': 25
}

# 2. 创建模拟器
simulator = SeismicMonteCarloSimple(region, time_period=100.0)

# 3. 设置参数
simulator.set_seismic_parameters(
    annual_rate=0.05,      # 年发生率
    b_value=1.0,           # Gutenberg-Richter b 值
    M_min=4.0, M_max=8.0,  # 震级范围
    depth_params={'type': 'lognormal', 'mean': 12, 'std': 6},
    spatial_params={'type': 'uniform'}
)

# 4. 运行模拟
results = simulator.run_monte_carlo_simulation(n_simulations=5, random_seed=42)

# 5. 获取统计
summary = simulator.get_summary_statistics()
print(f"平均事件数: {summary['mean_events_per_simulation']:.2f}")
print(f"平均震级: {summary['magnitude_stats']['mean']:.2f}")
```

---

## ⚠️ 原始版本环境配置（高级用户）

如果需要运行原始的 NumPy 版本（获得更好的性能和绘图功能），需要解决环境问题。

### 原因

Windows 上 NumPy 2.3.5 需要 C 编译器才能从源代码编译。系统当前：
- ❌ 缺少 Microsoft Visual C++ Build Tools
- ❌ Python 3.13 + NumPy 二进制兼容性问题

### 解决方案

#### 方案 B1：安装 Microsoft Build Tools（推荐）

1. 下载 Microsoft Build Tools for C++：
   - 官方链接：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist
   - 选择"Microsoft Visual C++ 2015-2022 Redistributable (x64)"

2. 安装后重启系统

3. 重新安装 NumPy：
   ```powershell
   .\.venv\Scripts\Activate.ps1
   pip install --upgrade --force-reinstall numpy pandas scipy matplotlib
   ```

4. 运行原始版本：
   ```powershell
   python 人工模拟地震动.py
   ```

#### 方案 B2：使用 Conda（最稳妥，推荐）

1. 安装 Miniconda：
   - 下载：https://docs.conda.io/en/latest/miniconda.html
   - 选择 Windows x64 版本

2. 创建隔离环境：
   ```powershell
   conda create -n seismic_env python=3.11 -y
   conda activate seismic_env
   conda install numpy pandas scipy matplotlib -y
   ```

3. 运行原始版本：
   ```powershell
   python 人工模拟地震动.py
   ```

---

## 📈 性能对比

| 特性 | 纯 Python 版 | NumPy 版 |
|------|------------|---------|
| 启动速度 | ⚡ 快 | ⚡ 快 |
| 大规模模拟（>10000 事件） | 🐢 慢（几秒） | 🚀 快（毫秒） |
| 绘图功能 | ❌ 无 | ✅ 完整 |
| 依赖管理 | ✅ 简单 | ⚠️ 复杂 |
| Windows 兼容性 | ✅ 完美 | ⚠️ 需编译器 |

**建议**：
- 🎯 日常使用、学习、小规模测试 → 使用**纯 Python 版**
- 📊 大规模研究、生成论文图表 → 升级到 **NumPy 版**

---

## 🐛 常见问题

**Q: 为什么纯 Python 版本加速度值这么大？**

A: 这是简化模型的特点。原始 NumPy 版本使用更复杂的物理模型和单位换算。纯 Python 版本演示的是相对值和模型框架。

**Q: 能否修改模拟参数？**

A: 可以！编辑脚本最后的 `main()` 函数中的参数即可。参考"自定义模拟"部分。

**Q: 多次运行结果是否相同？**

A: 使用 `random_seed=42` 时结果相同。不指定 seed 时每次随机。

**Q: 原始版本有什么优势？**

A: 
- 高性能数值计算（处理百万级事件）
- 完整的频谱绘图和可视化
- 更精确的物理模型
- 导出结果到 CSV/图像

---

## 📚 参考文献

- **Gutenberg, B., & Richter, C. F. (1944)**. Frequency of earthquakes in California. *Bulletin of the Seismological Society of America*, 34(4), 185-188.
- **Atkinson, G. M., & Boore, D. M. (1995)**. Ground-motion relations for eastern North America. *Bulletin of the Seismological Society of America*, 85(1), 17-30.
- **Wells, D. L., & Coppersmith, K. J. (1994)**. New empirical relationships among magnitude, rupture length, rupture width, rupture area, and surface displacement. *Bulletin of the Seismological Society of America*, 84(4), 974-1002.

---

## 📝 更新日志

### v1.0.0 (2024-11-24)

- ✅ 纯 Python 版本完成
- ✅ 蒙特卡洛地震事件模拟
- ✅ 地震动点源模型
- ✅ 完整统计分析
- ✅ 原始 NumPy 版本（需环境配置）

---

## 💡 联系与反馈

如有问题或建议，欢迎反馈！

---

**最后更新**：2024 年 11 月 24 日
