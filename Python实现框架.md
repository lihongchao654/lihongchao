# 您的Python代码改进方案

## 1. 针对现有文件的分析与建议

### 1.1 已有文件分析

```
您的仓库结构：
主震模拟/
├── 主震加速度谱模拟与处理.py
├── 综合可视化工具.py
├── 地震动加速度谱.csv
├── 地震事件汇总.csv
├── 条件均值谱.csv
├── 危险性曲线.csv
└── 一致危险谱_50年20%.csv
```

**关键问题**：
- 文件名中"50年20%"应理解为什么？
  - 若为50年20%超越概率 → 约239年回归周期
  - 通常地震规范用50年2%或50年10%
  - **建议确认**: 这是否应为"50年2%"

### 1.2 数据完整性检查清单

| 数据文件 | 完整性 | 应包含内容 | 改进建议 |
|--------|--------|----------|--------|
| 加速度谱.csv | ✓ | 周期T vs 谱加速度Sa | 添加计算方法注记 |
| 事件汇总.csv | ✓ | 震级M、距离r | 缺少深度H信息 |
| 条件均值谱.csv | ✓ | CMS(T) | 缺少标准差列 |
| 危险性曲线.csv | ✓ | 周期T vs 危险率λ | 好 |
| UHS_50年20%.csv | △ | UHS曲线 | **需澄清超越概率** |

---

## 2. 论文方法在代码中的具体映射

### 2.1 核心算法对应表

| 论文章节 | 核心方法 | 您代码应包含的模块 | 优先级 |
|--------|--------|------------------|--------|
| 2.1 源模型 | AB95_BC混合 | `SeismicSource` 类 | 高 |
| 2.2 UHS生成 | 危险性积分 | `HazardCalculator.compute_uhs()` | 高 |
| 2.3 CMS计算 | 条件均值 | `SpectrumCalculator.compute_cms()` | 高 |
| 2.3 记录选择 | 贪心优化 | `GroundMotionSelector.optimize()` | 中 |
| 2.3 缩放处理 | 多因子缩放 | `Scaler.two_factor_scale()` | 中 |

### 2.2 现有代码的功能诊断

**您的 `主震加速度谱模拟与处理.py` 应该包含**：

```python
# 现有可能的功能
✓ 地震动模拟
✓ 响应谱计算  
✓ 结果存储

# 可能缺失的功能
✗ GMPE模型（需引入CB08或Atkinson-Boore）
✗ CMS的标准差计算
✗ 相关系数模型
✗ 贪心优化算法
✗ 两阶段缩放处理
```

---

## 3. 改进代码框架

### 3.1 完整的模块架构

```python
# =============================================================================
# 地震动模拟与选择系统（基于论文方法）
# =============================================================================

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 第一部分：GMPE模型与地震动预测
# ─────────────────────────────────────────────────────────────────────────────

class GMPE_CB08:
    """
    Campbell-Boore 2008 Ground Motion Prediction Equation
    """
    
    def __init__(self):
        """初始化CB08模型参数"""
        # 基准参数（可从表格读入）
        self.T_ref = 1.0  # 参考周期
        self.Vs30_ref = 760  # 参考场地速度
        
    def predict(self, T, M, R_rup, R_jb, R_x, Vs30=760, 
                mechanism='unspecified', dip=90):
        """
        CB08 GMPE预测
        
        参数:
            T: 周期(秒)
            M: 震级(Mw)
            R_rup: 地震破裂距(km)
            R_jb: 最近断层距(km)
            R_x: 沿断层走向距离(km)
            Vs30: 30米平均剪切波速(m/s)
            mechanism: 断层机制('SS', 'RV', 'NM', 'unspecified')
            dip: 断层倾角(度)
        
        返回:
            ln_Sa: 对数谱加速度(g的对数)
            sigma: 标准差(对数单位)
        """
        # CB08的具体系数需要查表
        # 这里提供框架，实现需查阅论文附录
        
        # 1. 震级项
        f_M = self._magnitude_term(T, M)
        
        # 2. 距离项
        f_R = self._distance_term(T, M, R_rup, R_jb)
        
        # 3. 场地效应
        f_site = self._site_term(T, Vs30)
        
        # 4. 随机项
        f_random = self._random_term(T)
        
        # 5. 其他修正
        f_other = self._other_corrections(T, mechanism, dip, R_x)
        
        # 合成
        ln_Sa = f_M + f_R + f_site + f_random + f_other
        sigma = self._compute_sigma(T)
        
        return np.exp(ln_Sa), sigma
    
    def _magnitude_term(self, T, M):
        """震级缩放项"""
        # 参数来自CB08表格
        if M <= 5.5:
            return 0.0  # 简化处理
        else:
            return 0.5 * (M - 5.5)  # 示意
    
    def _distance_term(self, T, M, R_rup, R_jb):
        """距离衰减项"""
        # c1项，取决于T
        c1 = -2.5 if T < 0.5 else -2.0  # 示意值
        R_ref = np.sqrt(R_rup**2 + 10**2)
        return c1 * np.log10(R_ref)
    
    def _site_term(self, T, Vs30):
        """场地效应修正"""
        if Vs30 >= self.Vs30_ref:
            return 0.0
        else:
            # 土壤放大
            ln_vs_ratio = np.log(Vs30 / self.Vs30_ref)
            return 0.1 * ln_vs_ratio  # 简化示意
    
    def _random_term(self, T):
        """随机项（通常为0）"""
        return 0.0
    
    def _other_corrections(self, T, mechanism, dip, R_x):
        """其他修正（断层机制、悬挂墙效应等）"""
        return 0.0
    
    def _compute_sigma(self, T):
        """计算标准差"""
        # CB08标准差通常0.55-0.70
        if T < 0.5:
            return 0.55
        elif T < 2.0:
            return 0.60
        else:
            return 0.68


# ─────────────────────────────────────────────────────────────────────────────
# 第二部分：危险性分析与UHS计算
# ─────────────────────────────────────────────────────────────────────────────

class HazardCalculator:
    """
    概率地震危险分析与UHS生成
    """
    
    def __init__(self, gmpe_model):
        self.gmpe = gmpe_model
        self.earthquakes = None  # 地震目录
        
    def set_earthquake_catalog(self, catalog_df):
        """
        设置地震目录
        
        参数:
            catalog_df: DataFrame with columns [ID, M, r, depth, rate]
        """
        self.earthquakes = catalog_df
        
    def compute_hazard_curve(self, T, sa_levels, site_params):
        """
        计算单个周期的危险性曲线
        
        参数:
            T: 周期(秒)
            sa_levels: 谱加速度水准数组(g)
            site_params: 场地参数字典 {Vs30, lon, lat, ...}
        
        返回:
            sa_levels: 谱加速度水准(g)
            probabilities: 年超越概率
        """
        
        annual_prob = np.zeros_like(sa_levels)
        
        for i, sa in enumerate(sa_levels):
            # 积分: λ(Sa>a) = ∫∫ λ_M(m) * P(Sa>a|m,r) dm dr
            hazard_sum = 0.0
            
            for _, eq in self.earthquakes.iterrows():
                M = eq['M']
                r = eq['r']
                annual_rate = eq.get('rate', 1.0/1000)  # 默认1000年1次
                
                # 预测Sa及其标准差
                Sa_mean, sigma = self.gmpe.predict(
                    T, M, r, r, 0, 
                    Vs30=site_params.get('Vs30', 760)
                )
                
                # 计算P(Sa > a | M, r)
                # 假设对数正态分布
                ln_sa = np.log(sa)
                ln_sa_mean = np.log(Sa_mean)
                z = (ln_sa - ln_sa_mean) / sigma
                
                # 标准正态分布的生存函数
                from scipy.stats import norm
                prob_exceed = 1 - norm.cdf(z)
                
                hazard_sum += annual_rate * prob_exceed
            
            annual_prob[i] = hazard_sum
        
        return sa_levels, annual_prob
    
    def compute_uhs(self, periods, exceedance_prob, site_params):
        """
        生成一致危险谱 (UHS)
        
        参数:
            periods: 周期数组(秒)
            exceedance_prob: 目标年超越概率(如0.0004 = 50年2%)
            site_params: 场地参数
        
        返回:
            periods: 周期数组
            uhs: UHS谱加速度(g)
        """
        
        uhs = np.zeros_like(periods)
        
        for i, T in enumerate(periods):
            # 对该周期生成危险性曲线
            sa_range = np.logspace(-2, 1, 100)  # Sa从0.01到10g
            sa_levels, prob_exceed = self.compute_hazard_curve(
                T, sa_range, site_params
            )
            
            # 找到与目标超越概率对应的Sa值
            # 使用插值
            from scipy.interpolate import interp1d
            f_interp = interp1d(prob_exceed[::-1], sa_levels[::-1], 
                               fill_value='extrapolate')
            uhs[i] = f_interp(exceedance_prob)
        
        return periods, uhs


# ─────────────────────────────────────────────────────────────────────────────
# 第三部分：条件均值谱(CMS)计算
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumCalculator:
    """
    响应谱计算与相关分析
    """
    
    def __init__(self, gmpe_model):
        self.gmpe = gmpe_model
    
    def compute_spectrum(self, motion_time_series, dt, periods=None):
        """
        计算响应谱 (Newmark-β方法)
        
        参数:
            motion_time_series: 加速度时间序列(g)
            dt: 时间步长(秒)
            periods: 周期数组，如为None则自动生成
        
        返回:
            periods: 周期数组
            Sa: 谱加速度(g)
        """
        if periods is None:
            periods = np.logspace(-2, 1, 80)
        
        Sa = np.zeros_like(periods)
        
        for i, T in enumerate(periods):
            # 计算频率和阻尼比
            f = 1 / T
            omega = 2 * np.pi * f
            xi = 0.05  # 标准阻尼比
            
            # Newmark-β积分（简化）
            # 实际应用应使用专业库如OpenSees或Openseespy
            max_disp = self._compute_max_displacement(
                motion_time_series, dt, omega, xi
            )
            
            Sa[i] = omega**2 * max_disp / 9.81  # 转换为g
        
        return periods, Sa
    
    def _compute_max_displacement(self, acceleration, dt, omega, xi):
        """Newmark-β方法计算最大位移"""
        # 简化实现，建议使用专业库
        from scipy.integrate import odeint
        
        n_steps = len(acceleration)
        gamma = 0.5
        beta = 0.25
        
        u = np.zeros(n_steps)
        v = np.zeros(n_steps)
        
        a_prev = 0
        
        for i in range(1, n_steps):
            a_curr = acceleration[i]
            a_bar = gamma * (a_curr - a_prev) / (beta * dt) + \
                   (1 - gamma / (2*beta)) * v[i-1]
            
            dt_eff = 1 / (omega**2 + 2*xi*omega*a_bar/dt)
            
            u[i] = u[i-1] + dt*v[i-1] + dt**2*(1-2*beta)*a_prev/(2*beta) + \
                   beta*dt**2*a_curr/beta
            v[i] = v[i-1] + dt*(1-gamma)*a_prev + gamma*dt*a_curr
            
            a_prev = a_curr
        
        return np.max(np.abs(u))
    
    def compute_cms(self, T1, Sa1, T2, M, r, site_params):
        """
        计算条件均值谱 (CMS)
        
        参数:
            T1: 参考周期(秒)，通常1.0s
            Sa1: 参考周期的谱加速度(g)
            T2: 目标周期(秒)
            M: 震级(Mw)
            r: 距离(km)
            site_params: 场地参数
        
        返回:
            cms_value: 条件均值谱加速度(g)
            cms_std: 条件标准差(对数单位)
        """
        
        # 1. 获取无条件的均值和标准差
        Sa1_mean, sigma1 = self.gmpe.predict(T1, M, r, r, 0, **site_params)
        Sa2_mean, sigma2 = self.gmpe.predict(T2, M, r, r, 0, **site_params)
        
        # 2. 计算相关系数 (Jayaram & Baker)
        rho = self._jayaram_baker_correlation(T1, T2)
        
        # 3. 计算条件参数
        # μ_2|1 = μ_2 + ρ·(σ_2/σ_1)·(ln(Sa1) - μ_1)
        
        ln_sa1_obs = np.log(Sa1)
        ln_sa1_mean = np.log(Sa1_mean)
        ln_sa2_mean = np.log(Sa2_mean)
        
        ln_sa2_given_sa1 = ln_sa2_mean + rho * (sigma2/sigma1) * \
                           (ln_sa1_obs - ln_sa1_mean)
        
        # σ_2|1 = σ_2·√(1 - ρ²)
        sigma2_given_sa1 = sigma2 * np.sqrt(1 - rho**2)
        
        cms_value = np.exp(ln_sa2_given_sa1)
        
        return cms_value, sigma2_given_sa1
    
    def _jayaram_baker_correlation(self, T1, T2):
        """
        Jayaram & Baker (2008)相关系数模型
        
        参数:
            T1, T2: 周期(秒)
        
        返回:
            rho: 相关系数(0-1)
        """
        
        T_min = min(T1, T2)
        T_max = max(T1, T2)
        
        # 相关系数参数
        c1 = 1.0
        c2 = 0.57
        
        # 相关性函数
        if T_min < c2:
            rho = c1 - (1 - c1) * (np.log(T_max/T_min) / np.log(c2/T_min))**0.5
        else:
            rho = c1
        
        # 限制在[0, 1]
        rho = np.clip(rho, 0, 1)
        
        return rho


# ─────────────────────────────────────────────────────────────────────────────
# 第四部分：地震动记录选择与缩放
# ─────────────────────────────────────────────────────────────────────────────

class GroundMotionSelector:
    """
    基于CMS的地震记录选择与贪心优化
    """
    
    def __init__(self, spectrum_calc):
        self.spectrum_calc = spectrum_calc
        self.records_db = None  # 地震记录数据库
        
    def load_record_database(self, records_file):
        """
        加载地震记录数据库
        
        参数:
            records_file: CSV文件，包含地震记录的元数据和时间序列
        """
        self.records_db = pd.read_csv(records_file)
    
    def greedy_select(self, n_records, cms_target, periods, 
                     weights=None, scale_range=(0.5, 2.0)):
        """
        贪心算法选择最优记录组合
        
        参数:
            n_records: 目标选择的记录数量
            cms_target: 目标CMS (周期 vs CMS值)
            periods: 周期数组
            weights: 误差权重 [w_mean, w_std, w_corr]
            scale_range: 缩放因子范围
        
        返回:
            selected_records: 选中的记录索引列表
            errors: 误差历史
        """
        
        if weights is None:
            weights = [2, 2, 1]  # 论文中的权重比例
        
        n_total = len(self.records_db)
        selected = []
        candidates = list(range(n_total))
        errors = []
        
        for iteration in range(n_records):
            best_error = float('inf')
            best_record = None
            best_scale = None
            
            for rec_idx in candidates:
                # 计算该记录加入后的总体误差
                test_set = selected + [rec_idx]
                
                # 尝试不同的缩放因子
                for scale in np.linspace(scale_range[0], scale_range[1], 20):
                    # 获取记录的响应谱
                    rec_spectrum = self._get_record_spectrum(
                        rec_idx, periods
                    )
                    rec_spectrum_scaled = rec_spectrum * scale
                    
                    # 计算组合误差
                    error = self._compute_ensemble_error(
                        test_set, rec_spectrum_scaled, 
                        cms_target, periods, weights
                    )
                    
                    if error < best_error:
                        best_error = error
                        best_record = rec_idx
                        best_scale = scale
            
            # 添加最优记录
            selected.append(best_record)
            candidates.remove(best_record)
            errors.append(best_error)
            
            print(f"迭代 {iteration+1}/{n_records}: " + 
                  f"选择记录{best_record}, 缩放因子={best_scale:.2f}, " +
                  f"误差={best_error:.4f}")
        
        return selected, errors
    
    def _get_record_spectrum(self, record_idx, periods):
        """获取记录的响应谱"""
        # 从数据库加载响应谱
        # 实际实现需要加载时间序列并计算谱
        pass
    
    def _compute_ensemble_error(self, record_indices, added_spectrum,
                                cms_target, periods, weights):
        """
        计算记录集合相对于CMS的误差
        
        参数:
            record_indices: 选中记录的索引
            added_spectrum: 新增记录的缩放后响应谱
            cms_target: 目标CMS
            periods: 周期数组
            weights: 权重系数
        
        返回:
            total_error: 加权总误差
        """
        
        w_mean, w_std, w_corr = weights
        
        # 计算集合均值
        ensemble_spectra = []
        for idx in record_indices:
            spec = self._get_record_spectrum(idx, periods)
            ensemble_spectra.append(spec)
        
        ensemble_spectra.append(added_spectrum)
        ensemble_array = np.array(ensemble_spectra)
        
        ensemble_mean = np.mean(ensemble_array, axis=0)
        ensemble_std = np.std(np.log(ensemble_array), axis=0)
        
        # 计算CMS均值和标准差
        cms_mean = cms_target['sa']  # 假设列名为'sa'
        cms_std = cms_target.get('std', np.std(np.log(cms_mean)))
        
        # 误差项
        error_mean = np.sum((np.log(ensemble_mean) - 
                           np.log(cms_mean))**2)
        
        error_std = np.sum((ensemble_std - cms_std)**2)
        
        # 简化：先不计算相关性误差
        error_corr = 0
        
        total_error = w_mean * error_mean + w_std * error_std + \
                     w_corr * error_corr
        
        return total_error
    
    def two_factor_scale(self, record_spectrum, cms_target, periods):
        """
        两阶段缩放处理
        
        参数:
            record_spectrum: 原始响应谱
            cms_target: 目标CMS
            periods: 周期数组
        
        返回:
            scaled_spectrum: 缩放后的响应谱
            scale_factors: 缩放因子(可能是数组)
        """
        
        # 分界周期
        T_boundary = 0.5
        
        # 短周期缩放因子
        short_period_mask = periods < T_boundary
        alpha = np.mean(cms_target[short_period_mask]) / \
               np.mean(record_spectrum[short_period_mask])
        
        # 长周期缩放因子
        long_period_mask = periods > T_boundary
        beta = np.mean(cms_target[long_period_mask]) / \
              np.mean(record_spectrum[long_period_mask])
        
        # 构建线性变化的缩放因子
        scale_factors = np.ones_like(periods)
        for i, T in enumerate(periods):
            if T < T_boundary:
                scale_factors[i] = alpha
            else:
                scale_factors[i] = beta + (alpha - beta) * \
                                  (T_boundary - T) / (T_boundary)
        
        scaled_spectrum = record_spectrum * scale_factors
        
        return scaled_spectrum, scale_factors


# ─────────────────────────────────────────────────────────────────────────────
# 第五部分：主程序与工作流
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """
    完整的地震动模拟与选择工作流
    """
    
    # 1. 初始化模型
    gmpe = GMPE_CB08()
    hazard_calc = HazardCalculator(gmpe)
    spectrum_calc = SpectrumCalculator(gmpe)
    selector = GroundMotionSelector(spectrum_calc)
    
    # 2. 加载地震目录
    earthquake_catalog = pd.DataFrame({
        'ID': range(1, 51),
        'M': np.random.uniform(4, 8, 50),
        'r': np.random.uniform(10, 200, 50),
        'depth': np.random.uniform(5, 20, 50),
        'rate': np.ones(50) * 1/1000
    })
    hazard_calc.set_earthquake_catalog(earthquake_catalog)
    
    # 3. 设置场地参数
    site_params = {
        'Vs30': 760,
        'lon': -89.3,
        'lat': 35.1,
        'mechanism': 'unspecified'
    }
    
    # 4. 生成周期数组
    periods = np.logspace(-2, 1, 80)  # 0.01 ~ 10 秒
    
    # 5. 计算UHS (50年2% = 0.0004年-1)
    periods_uhs, uhs = hazard_calc.compute_uhs(
        periods, 0.0004, site_params
    )
    
    print(f"生成UHS: {len(periods_uhs)}个周期")
    
    # 6. 计算CMS (M=8.0, r=40km)
    T_ref = 1.0
    Sa_ref = uhs[periods == T_ref][0]
    
    cms_values = []
    for T in periods:
        cms, cms_std = spectrum_calc.compute_cms(
            T_ref, Sa_ref, T, 8.0, 40, site_params
        )
        cms_values.append(cms)
    
    cms_df = pd.DataFrame({
        'Period': periods,
        'Sa': cms_values
    })
    
    print(f"生成CMS: {len(periods)}个周期")
    
    # 7. 保存结果
    uhs_df = pd.DataFrame({'Period': periods_uhs, 'Sa': uhs})
    uhs_df.to_csv('UHS_computed.csv', index=False)
    cms_df.to_csv('CMS_computed.csv', index=False)
    
    # 8. 可视化
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.loglog(periods_uhs, uhs, 'b-', linewidth=2, label='UHS (50年2%)')
    plt.xlabel('周期 (秒)')
    plt.ylabel('谱加速度 (g)')
    plt.title('一致危险谱')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.loglog(periods, cms_values, 'r-', linewidth=2, 
               label='CMS (M=8.0, r=40km)')
    plt.xlabel('周期 (秒)')
    plt.ylabel('谱加速度 (g)')
    plt.title('条件均值谱')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('Spectra_Comparison.png', dpi=300)
    print("图表已保存为 Spectra_Comparison.png")
    
    print("\n✓ 完成！结果已保存到CSV文件")


if __name__ == '__main__':
    main()
```

### 3.2 代码使用示例

```python
# 示例：快速计算
from groundmotion_system import *

# 初始化
gmpe = GMPE_CB08()
spec_calc = SpectrumCalculator(gmpe)

# 计算某一周期的CMS
cms, std = spec_calc.compute_cms(
    T1=1.0,           # 参考周期
    Sa1=0.2,          # 参考谱加速度(g)
    T2=0.3,           # 目标周期
    M=8.0,            # 震级
    r=40,             # 距离(km)
    site_params={'Vs30': 760}
)

print(f"T=0.3s的CMS: {cms:.3f}g, 标准差: {std:.3f}")
```

---

## 4. 与现有文件的集成方案

### 4.1 数据导入

```python
# 读取您已有的数据
def load_existing_data():
    # 加速度谱
    df_accel = pd.read_csv('地震动加速度谱.csv')
    
    # 条件均值谱
    df_cms_existing = pd.read_csv('条件均值谱.csv')
    
    # 危险性曲线
    df_hazard = pd.read_csv('危险性曲线.csv')
    
    # UHS
    df_uhs = pd.read_csv('一致危险谱_50年20%.csv')
    # ⚠️ 确认这里的"20%"是否有误
    
    return df_accel, df_cms_existing, df_hazard, df_uhs

# 数据验证
df_accel, df_cms, df_hazard, df_uhs = load_existing_data()
print("数据加载完成")
print(f"加速度谱数据点: {len(df_accel)}")
print(f"CMS数据点: {len(df_cms)}")
print(f"危险曲线数据点: {len(df_hazard)}")
```

### 4.2 结果对比与验证

```python
# 比较论文方法的计算结果与您的现有数据

def compare_results(df_existing, df_computed):
    """
    对比现有数据与新计算结果
    """
    
    # 插值对齐周期
    periods_common = np.intersect1d(
        df_existing['Period'].values,
        df_computed['Period'].values
    )
    
    # 计算RMSE
    rmse = np.sqrt(np.mean(
        (np.log(df_existing['Sa']) - 
         np.log(df_computed['Sa']))**2
    ))
    
    print(f"RMSE (对数单位): {rmse:.4f}")
    print(f"对应的加速度相对误差约: ±{(np.exp(rmse)-1)*100:.1f}%")
    
    # 可视化对比
    plt.figure(figsize=(10, 6))
    plt.loglog(df_existing['Period'], df_existing['Sa'], 
               'o-', label='现有数据', alpha=0.7)
    plt.loglog(df_computed['Period'], df_computed['Sa'], 
               's-', label='新计算结果', alpha=0.7)
    plt.xlabel('周期(秒)')
    plt.ylabel('谱加速度(g)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('result_comparison.png', dpi=300)
```

---

## 5. 优先实现清单

### 优先级1（本周完成）
- [ ] 实现基本的CB08 GMPE
- [ ] 测试CMS计算的正确性
- [ ] 与您的条件均值谱.csv对比验证

### 优先级2（次周）
- [ ] 实现完整的危险性曲线计算
- [ ] UHS生成及验证
- [ ] 与UHS_50年20%.csv对比

### 优先级3（第三周）
- [ ] 贪心优化算法实现
- [ ] 地震记录选择功能
- [ ] 两阶段缩放处理

### 优先级4（优化阶段）
- [ ] 性能优化
- [ ] 文档完善
- [ ] GUI界面

---

## 6. 常见实现问题

### 问题1：周期分辨率
**症状**：UHS曲线出现锯齿状
**原因**：周期数量不足或不均匀分布
**解决**：使用对数均匀间隔，至少60-80个周期

### 问题2：CMS与UHS差异过大
**症状**：条件均值谱明显低于UHS
**原因**：相关系数计算有误，或GMPE参数不对
**解决**：验证Jayaram-Baker相关系数，检查CB08参数表

### 问题3：记录选择不收敛
**症状**：贪心算法误差不下降
**原因**：候选记录库不足，或权重系数设置不当
**解决**：增加候选记录，调整权重比例到2:2:1

---

**生成日期**：2025年11月26日  
**适用于**：您的Python实现改进
