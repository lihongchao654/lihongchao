#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Python 版本的完整地震模拟系统
包含地震危险性分析：一致危险谱和条件均值谱计算
"""

import sys
import math
import random
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 处理 Windows 控制台编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ============================================================================
# 纯 Python 数学工具库（替代 NumPy）
# ============================================================================

@dataclass
class Array:
    """简单的 Python 数组包装类（模拟 NumPy 数组）"""
    data: List[float]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def __setitem__(self, idx, val):
        self.data[idx] = val
    
    def sum(self):
        return sum(self.data)
    
    def mean(self):
        return sum(self.data) / len(self.data) if self.data else 0
    
    def std(self):
        if not self.data or len(self.data) == 1:
            return 0
        m = self.mean()
        return math.sqrt(sum((x - m) ** 2 for x in self.data) / len(self.data))
    
    def min(self):
        return min(self.data) if self.data else 0
    
    def max(self):
        return max(self.data) if self.data else 0
    
    def __repr__(self):
        return f"Array({self.data})"


class SimpleMath:
    """简单的数学函数库（替代 SciPy）"""
    
    @staticmethod
    def exp(x):
        """指数函数"""
        return math.exp(x)
    
    @staticmethod
    def log(x):
        """自然对数"""
        return math.log(x) if x > 0 else -999
    
    @staticmethod
    def log10(x):
        """以 10 为底的对数"""
        return math.log10(x) if x > 0 else -999
    
    @staticmethod
    def sqrt(x):
        """平方根"""
        return math.sqrt(x) if x >= 0 else 0
    
    @staticmethod
    def sin(x):
        """正弦"""
        return math.sin(x)
    
    @staticmethod
    def cos(x):
        """余弦"""
        return math.cos(x)
    
    @staticmethod
    def pi():
        """圆周率"""
        return math.pi
    
    @staticmethod
    def exp_negative(x):
        """负指数 e^(-x)"""
        try:
            return math.exp(-x)
        except:
            return 0


# ============================================================================
# 地震蒙特卡洛模拟器（纯 Python 版）
# ============================================================================

class SeismicMonteCarloSimple:
    """
    基于泊松模型的地震蒙特卡洛模拟器（纯 Python 版本）
    """
    
    def __init__(self, region_bounds: Dict, time_period: float = 1000.0):
        self.region_bounds = region_bounds
        self.time_period = time_period
        self.math_lib = SimpleMath()
        
        self.seismic_params = {
            'annual_rate': 0.1,
            'b_value': 1.0,
            'M_min': 4.0,
            'M_max': 8.0,
            'depth_distribution': {'type': 'lognormal', 'mean': 10, 'std': 5},
            'spatial_distribution': {'type': 'uniform'},
        }
        
        self.simulation_results = None
        
    def set_seismic_parameters(self, annual_rate: float, b_value: float, 
                             M_min: float, M_max: float, 
                             depth_params: Dict, spatial_params: Dict):
        self.seismic_params.update({
            'annual_rate': annual_rate,
            'b_value': b_value,
            'M_min': M_min,
            'M_max': M_max,
            'depth_distribution': depth_params,
            'spatial_distribution': spatial_params
        })
    
    def gutenberg_richter_magnitude(self, n_events: int) -> List[float]:
        """古腾堡-里希特震级分布"""
        b_value = self.seismic_params['b_value']
        M_min = self.seismic_params['M_min']
        M_max = self.seismic_params['M_max']
        
        beta = b_value * self.math_lib.log(10)
        C = 1 - self.math_lib.exp_negative(beta * (M_max - M_min))
        
        magnitudes = []
        for _ in range(n_events):
            u = random.random()
            m = M_min - (1/beta) * self.math_lib.log(1 - u * C)
            magnitudes.append(m)
        
        return magnitudes
    
    def generate_depth_distribution(self, n_events: int) -> List[float]:
        """
        生成深度分布
        
        改进方法：使用更真实的Gamma分布而不是对数正态分布
        地震深度通常表现为Gamma分布特征
        """
        depth_params = self.seismic_params['depth_distribution']
        depth_min = self.region_bounds['depth_min']
        depth_max = self.region_bounds['depth_max']
        
        depths = []
        
        if depth_params['type'] == 'gamma':
            # 使用Gamma分布（更符合地震学特征）
            # 参数化：α=形状参数，β=尺度参数
            # 均值 = α*β，方差 = α*β^2
            
            mean = depth_params.get('mean', 12)
            std = depth_params.get('std', 6)
            
            # 计算Gamma分布参数
            alpha = (mean / std) ** 2
            beta = std ** 2 / mean
            
            for _ in range(n_events):
                # 使用Marsaglia and Tsang的算法采样Gamma分布
                if alpha >= 1:
                    d = alpha - 1/3
                    c = 1 / math.sqrt(9 * d)
                else:
                    d = alpha
                    c = 1 / (3 * math.sqrt(d))
                
                while True:
                    u1 = random.random()
                    if u1 > 1e-10:  # 避免log(0)
                        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * random.random())
                        v = 1 + c * z
                        if v > 0:
                            u2 = random.random()
                            if u2 <= 1 - 0.0331 * z ** 4:
                                break
                            if math.log(u2) <= 0.5 * z ** 2 + d * (1 - v + math.log(v)):
                                break
                
                depth = d * v * beta
                depths.append(depth)
        
        elif depth_params['type'] == 'truncated_normal':
            # 截断正态分布
            mean = depth_params.get('mean', 12)
            std = depth_params.get('std', 6)
            
            for _ in range(n_events):
                while True:
                    depth = random.gauss(mean, std)
                    if depth_min <= depth <= depth_max:
                        depths.append(depth)
                        break
        
        else:
            # 默认均匀分布
            depths = [random.uniform(depth_min, depth_max) for _ in range(n_events)]
        
        # 限制范围
        depths = [min(max(d, depth_min), depth_max) for d in depths]
        return depths
    
    def generate_spatial_distribution(self, n_events: int) -> Tuple[List[float], List[float]]:
        """生成空间分布"""
        spatial_params = self.seismic_params['spatial_distribution']
        lon_min = self.region_bounds['lon_min']
        lon_max = self.region_bounds['lon_max']
        lat_min = self.region_bounds['lat_min']
        lat_max = self.region_bounds['lat_max']
        
        longitudes = [random.uniform(lon_min, lon_max) for _ in range(n_events)]
        latitudes = [random.uniform(lat_min, lat_max) for _ in range(n_events)]
        
        return longitudes, latitudes
    
    def generate_occurrence_times(self, n_events: int) -> List[float]:
        """生成发生时间（泊松分布）"""
        annual_rate = self.seismic_params['annual_rate']
        
        if annual_rate <= 0:
            raise ValueError("年均发生率必须大于 0")
        
        # 指数分布采样
        inter_event_times = [-self.math_lib.log(random.random()) / annual_rate for _ in range(n_events)]
        occurrence_times = []
        cumsum = 0
        for iet in inter_event_times:
            cumsum += iet
            if cumsum <= self.time_period:
                occurrence_times.append(cumsum)
        
        return occurrence_times
    
    def run_monte_carlo_simulation(self, n_simulations: int = 1, 
                                 random_seed: Optional[int] = None) -> Dict:
        """运行蒙特卡洛地震事件模拟"""
        if random_seed is not None:
            random.seed(random_seed)
        
        all_simulations = []
        
        for sim_idx in range(n_simulations):
            annual_rate = self.seismic_params['annual_rate']
            expected_events = annual_rate * self.time_period
            
            # 泊松分布：生成事件数
            n_events = 0
            prob_sum = self.math_lib.exp_negative(expected_events)
            threshold = random.random()
            cumsum = prob_sum
            while cumsum < threshold:
                n_events += 1
                prob_sum *= expected_events / n_events
                cumsum += prob_sum
            
            if n_events == 0:
                simulation = {
                    'occurrence_times': [],
                    'magnitudes': [],
                    'longitudes': [],
                    'latitudes': [],
                    'depths': []
                }
            else:
                occurrence_times = self.generate_occurrence_times(n_events)
                magnitudes = self.gutenberg_richter_magnitude(len(occurrence_times))
                longitudes, latitudes = self.generate_spatial_distribution(len(occurrence_times))
                depths = self.generate_depth_distribution(len(occurrence_times))
                
                simulation = {
                    'occurrence_times': occurrence_times,
                    'magnitudes': magnitudes,
                    'longitudes': longitudes,
                    'latitudes': latitudes,
                    'depths': depths
                }
            
            all_simulations.append(simulation)
        
        self.simulation_results = {
            'parameters': self.seismic_params.copy(),
            'region_bounds': self.region_bounds.copy(),
            'time_period': self.time_period,
            'n_simulations': n_simulations,
            'simulations': all_simulations
        }
        
        return self.simulation_results
    
    def get_summary_statistics(self) -> Dict:
        """获取统计摘要"""
        if self.simulation_results is None:
            raise ValueError("请先运行蒙特卡洛模拟")
        
        all_magnitudes = []
        all_depths = []
        event_counts = []
        
        for sim in self.simulation_results['simulations']:
            event_counts.append(len(sim['occurrence_times']))
            all_magnitudes.extend(sim['magnitudes'])
            all_depths.extend(sim['depths'])
        
        summary = {
            'total_simulations': len(self.simulation_results['simulations']),
            'mean_events_per_simulation': self._mean(event_counts),
            'std_events_per_simulation': self._std(event_counts),
            'min_events_per_simulation': min(event_counts) if event_counts else 0,
            'max_events_per_simulation': max(event_counts) if event_counts else 0,
            'magnitude_stats': {
                'mean': self._mean(all_magnitudes),
                'std': self._std(all_magnitudes),
                'min': min(all_magnitudes) if all_magnitudes else 0,
                'max': max(all_magnitudes) if all_magnitudes else 0,
            },
            'depth_stats': {
                'mean': self._mean(all_depths),
                'std': self._std(all_depths),
                'min': min(all_depths) if all_depths else 0,
                'max': max(all_depths) if all_depths else 0,
            }
        }
        
        return summary
    
    @staticmethod
    def _mean(data):
        return sum(data) / len(data) if data else 0
    
    @staticmethod
    def _std(data):
        if not data or len(data) == 1:
            return 0
        m = sum(data) / len(data)
        return math.sqrt(sum((x - m) ** 2 for x in data) / len(data))


# ============================================================================
# 简化的地震动模型
# ============================================================================

def improved_ground_motion_model(frequency: float, magnitude: float, distance: float, 
                                 random_seed: Optional[float] = None) -> float:
    """
    改进的地震动预测模型（基于Boore-Joyner-Fumal经验模型的简化版本）
    
    参数:
    - frequency: 频率 (Hz)
    - magnitude: 震级 (M)
    - distance: 震源到台站的最近距离 (km)
    - random_seed: 随机波动（单位sigma）
    """
    math_lib = SimpleMath()
    
    # ===== 地震矩和拐角频率计算 =====
    # 地震矩 (dyne-cm)
    M0 = 10 ** (1.5 * magnitude + 9.105)
    
    # 拐角频率 (Hz) - 基于Brune模型
    # f_c = (4.9e6 * Vs * (σ/M0)^(1/3)) / (10^7)
    Vs = 3.5  # km/s - 剪切波速度
    stress_drop = 100 * 1e5  # dyne/cm^2，假设应力降为100 bar
    f_c = 4.9e6 * Vs * (stress_drop / M0) ** (1/3) / 1e7
    
    # ===== 位移谱计算（改进的Brune谱） =====
    # 使用更精确的高频衰减
    if frequency > 0:
        brune_spectrum = M0 / (4 * 3.14159 * Vs * 1e5 * (1 + (frequency/f_c)**2))
    else:
        brune_spectrum = M0 / (4 * 3.14159 * Vs * 1e5)
    
    # ===== 路径衰减和几何展开 =====
    # 计算有效距离（考虑震源深度）
    depth = 15  # 假设平均深度 km
    R_eff = math.sqrt(distance**2 + depth**2)
    
    # 几何衰减（考虑近场和远场差异）
    if R_eff < 10:
        # 近场：球面波传播
        geometric_spread = 1.0 / max(R_eff, 1.0)
    else:
        # 远场：圆柱形波传播（Q衰减）
        geometric_spread = 1.0 / math.sqrt(max(R_eff, 1.0))
    
    # ===== 频率相关的品质因子衰减 =====
    # Q(f) = Q_0 * f^η，其中Q_0=100, η=0.8（典型值）
    Q_0 = 100
    eta = 0.8
    Q_f = Q_0 * (frequency ** eta) if frequency > 0 else Q_0
    
    # 体波衰减（以π计算）
    v_wave = 3.5  # km/s
    t_travel = R_eff / v_wave
    kappa_factor = 3.14159 * frequency * t_travel / Q_f
    q_attenuation = math_lib.exp_negative(kappa_factor)
    
    # ===== 高频滤波和辐射效应 =====
    # 高频滤波（kappa值）- 考虑浅层衰减
    kappa = 0.035
    high_freq_filter = math_lib.exp_negative(3.14159 * frequency * kappa)
    
    # 辐射效应因子（与震源机制相关，这里使用平均值）
    radiation_factor = 0.55  # 平均值
    
    # ===== 频率转换因子 =====
    # 从位移谱转换到加速度谱：乘以(2π*f)^2
    freq_factor = (2 * 3.14159 * frequency) ** 2 if frequency > 0 else 1.0
    
    # ===== 单位转换和缩放 =====
    # 位移谱：cm，加速度谱：cm/s^2
    # 1 dyne = 1 g*cm/s^2 / 980.665
    unit_conversion = 1.0 / 980.665
    
    # ===== 综合地震动计算 =====
    acceleration = (freq_factor * brune_spectrum * geometric_spread * q_attenuation * 
                   high_freq_filter * radiation_factor * unit_conversion)
    
    # ===== 添加随机波动 =====
    if random_seed is not None:
        # 标准差：高频域约0.3-0.5, 低频域约0.2-0.3
        if frequency < 0.5:
            sigma = 0.25
        elif frequency < 2.0:
            sigma = 0.35
        else:
            sigma = 0.4
        
        # 应用随机波动：Sa_random = Sa * exp(σ * ε)
        acceleration = acceleration * math_lib.exp(sigma * random_seed)
    
    return max(acceleration, 1e-6)


# ============================================================================
# 一致危险谱和条件均值谱计算模块
# ============================================================================

class SeismicHazardAnalyzer:
    """
    地震危险性分析器 - 计算一致危险谱和条件均值谱
    """
    
    def __init__(self, simulation_results: Dict, region_bounds: Dict):
        self.simulation_results = simulation_results
        self.region_bounds = region_bounds
        self.math_lib = SimpleMath()
        
        # 定义周期范围 (s) - 增加分辨率到33个周期
        self.periods = [
            0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09,
            0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90,
            1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.50, 10.00
        ]
        # 对应的频率 (Hz)
        self.frequencies = [1.0 / p for p in self.periods]
        
        # 存储分析结果
        self.hazard_curves = None
        self.uhs_50yr_20p = None
        self.cms = None
        
    def calculate_hazard_curves(self) -> Dict[float, List[Tuple[float, float]]]:
        """
        计算各周期的危险性曲线
        
        返回:
        Dict[float, List[Tuple[float, float]]]: 键为周期，值为(谱加速度, 年超越概率)列表
        """
        print("计算各周期的危险性曲线...")
        
        # 获取第一次模拟的数据
        sim = self.simulation_results['simulations'][0]
        n_events = len(sim['occurrence_times'])
        
        # 计算台站位置（研究区域中心）
        station_lon = (self.region_bounds['lon_min'] + self.region_bounds['lon_max']) / 2
        station_lat = (self.region_bounds['lat_min'] + self.region_bounds['lat_max']) / 2
        
        # 存储每个周期的谱加速度值
        period_sa_values = {period: [] for period in self.periods}
        
        # 对每个事件计算反应谱
        for event_idx in range(n_events):
            mag = sim['magnitudes'][event_idx]
            lon = sim['longitudes'][event_idx]
            lat = sim['latitudes'][event_idx]
            
            # 计算震源到台站的距离
            distance = self._calculate_distance((lon, lat), (station_lon, station_lat))
            
            # 计算每个周期的谱加速度
            for period, freq in zip(self.periods, self.frequencies):
                # 添加随机波动（高斯分布）
                epsilon = random.gauss(0, 1)  # 标准正态分布
                sa = improved_ground_motion_model(freq, mag, distance, random_seed=epsilon)
                period_sa_values[period].append(sa)
        
        # 构建危险性曲线
        hazard_curves = {}
        
        for period, sa_values in period_sa_values.items():
            if not sa_values:
                continue
                
            # 对谱加速度排序
            sorted_sa = sorted(sa_values, reverse=True)
            
            # 计算年超越概率
            annual_rates = []
            n_total = len(sorted_sa)
            simulation_years = self.simulation_results['time_period']
            
            for i, sa in enumerate(sorted_sa):
                # 年超越概率 = (排名 + 1) / (总事件数 + 1) / 模拟年数
                # 使用平均年超越概率公式
                annual_rate = (i + 1) / (n_total + 1) / simulation_years
                annual_rates.append((sa, annual_rate))
            
            hazard_curves[period] = annual_rates
        
        self.hazard_curves = hazard_curves
        return hazard_curves
    
    def calculate_uniform_hazard_spectrum(self, target_annual_probability: float = 0.0044) -> Dict[float, float]:
        """
        计算一致危险谱
        
        参数:
        target_annual_probability (float): 目标年超越概率
                                         50年20%超越概率对应的年超越概率 = 1 - (1-0.2)^(1/50) ≈ 0.0044
        
        返回:
        Dict[float, float]: 一致危险谱 {周期: 谱加速度}
        """
        print(f"计算一致危险谱 (年超越概率 = {target_annual_probability:.6f})...")
        
        if self.hazard_curves is None:
            self.calculate_hazard_curves()
        
        uhs = {}
        
        for period, hazard_curve in self.hazard_curves.items():
            # 找到最接近目标概率的谱加速度
            target_sa = None
            min_diff = float('inf')
            
            for sa, prob in hazard_curve:
                diff = abs(prob - target_annual_probability)
                if diff < min_diff:
                    min_diff = diff
                    target_sa = sa
            
            if target_sa is not None:
                uhs[period] = target_sa
        
        self.uhs_50yr_20p = uhs
        return uhs
    
    def calculate_conditional_mean_spectrum(self, control_period: float = 1.0) -> Dict[float, float]:
        """
        计算条件均值谱
        
        参数:
        control_period (float): 控制周期 (s)，通常选择结构基本周期或UHS峰值周期
        
        返回:
        Dict[float, float]: 条件均值谱 {周期: 谱加速度}
        """
        print(f"计算条件均值谱 (控制周期 = {control_period}s)...")
        
        if self.uhs_50yr_20p is None:
            self.calculate_uniform_hazard_spectrum()
        
        # 获取控制周期的UHS值
        if control_period not in self.uhs_50yr_20p:
            # 找到最接近的控制周期
            closest_period = min(self.uhs_50yr_20p.keys(), 
                               key=lambda p: abs(p - control_period))
            control_period = closest_period
            print(f"使用最接近的控制周期: {control_period}s")
        
        uhs_control = self.uhs_50yr_20p[control_period]
        
        # 计算控制周期的epsilon值
        # 这里使用简化的地震动预测方程
        epsilon_control = self._calculate_epsilon(control_period, uhs_control)
        
        # 计算条件均值谱
        cms = {}
        
        for period in self.periods:
            if period == control_period:
                cms[period] = uhs_control
            else:
                # 根据Baker(2011)的CMS公式
                # CMS(T) = exp(μ_lnSa(T) + ρ(T, T*) * σ_lnSa(T) * ε(T*))
                
                # 简化的相关系数估计
                rho = self._estimate_correlation(period, control_period)
                
                # 计算条件均值
                mu_ln_sa = self._calculate_mean_ln_sa(period)
                sigma_ln_sa = self._calculate_std_ln_sa(period)
                
                conditional_mean_ln_sa = mu_ln_sa + rho * sigma_ln_sa * epsilon_control
                cms_sa = self.math_lib.exp(conditional_mean_ln_sa)
                
                cms[period] = cms_sa
        
        self.cms = cms
        return cms
    
    def _calculate_distance(self, event_location: Tuple, station_location: Tuple) -> float:
        """计算震源到台站的距离（大圆距离）"""
        lat1 = math.radians(event_location[1])
        lon1 = math.radians(event_location[0])
        lat2 = math.radians(station_location[1])
        lon2 = math.radians(station_location[0])
        
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return 6371.0 * c  # 地球半径 (km)
    
    def _calculate_epsilon(self, period: float, target_sa: float) -> float:
        """
        计算epsilon值 (标准正态变量)
        
        ε = [ln(Sa_target) - μ_lnSa] / σ_lnSa
        """
        mu_ln_sa = self._calculate_mean_ln_sa(period)
        sigma_ln_sa = self._calculate_std_ln_sa(period)
        
        if sigma_ln_sa == 0:
            return 0
        
        epsilon = (self.math_lib.log(target_sa) - mu_ln_sa) / sigma_ln_sa
        return epsilon
    
    def _calculate_mean_ln_sa(self, period: float) -> float:
        """
        计算ln(Sa)的均值（基于模拟数据）
        
        使用改进的方法：从实际模拟数据计算而不是经验值
        """
        # 获取该周期的所有谱加速度值
        if self.hazard_curves is None or period not in self.hazard_curves:
            return 0.5  # 默认值
        
        hazard_curve = self.hazard_curves[period]
        if not hazard_curve:
            return 0.5
        
        # 计算中位数对应的sa值（从危险性曲线中提取）
        # 年超越概率约0.5时的中位数
        sa_values = [sa for sa, prob in hazard_curve]
        
        if sa_values:
            # 计算中位值的对数
            median_sa = sa_values[len(sa_values) // 2]
            return self.math_lib.log(max(median_sa, 1e-6))
        
        return 0.5
    
    def _calculate_std_ln_sa(self, period: float) -> float:
        """
        计算ln(Sa)的标准差（基于模拟数据）
        
        改进方法：从实际模拟数据中计算，考虑周期相关性
        """
        # 获取该周期的所有谱加速度值
        if self.hazard_curves is None or period not in self.hazard_curves:
            return 0.6  # 默认值
        
        hazard_curve = self.hazard_curves[period]
        if len(hazard_curve) < 2:
            return 0.6
        
        # 计算ln(Sa)的样本标准差
        sa_values = [sa for sa, prob in hazard_curve]
        ln_sa_values = [self.math_lib.log(max(sa, 1e-6)) for sa in sa_values]
        
        # 计算平均值
        mean_ln_sa = sum(ln_sa_values) / len(ln_sa_values)
        
        # 计算标准差
        variance = sum((x - mean_ln_sa) ** 2 for x in ln_sa_values) / len(ln_sa_values)
        std_ln_sa = math.sqrt(variance) if variance > 0 else 0.6
        
        # 对长周期应用衰减因子（长周期通常有更小的变异性）
        if period > 2.0:
            std_ln_sa = std_ln_sa * (0.8 + 0.2 * math.exp(-period / 5))
        
        return max(std_ln_sa, 0.3)  # 最小标准差0.3
    
    def _estimate_correlation(self, period1: float, period2: float) -> float:
        """
        估计两个周期之间的相关系数
        
        使用Baker和Cornell(2006)提出的经验公式改进版本
        能更精确地捕捉地震动在不同周期间的相关性
        """
        math_lib = SimpleMath()
        
        if period1 == period2:
            return 1.0
        
        # 计算周期比的对数（取绝对值）
        T_min = min(period1, period2)
        T_max = max(period1, period2)
        log_ratio = math_lib.log(T_max / T_min)
        
        # Baker and Cornell (2006) 改进的经验公式
        # ρ(T1,T2) = 1 - cos(π/2 - 0.366*ln(T_max/T_min))
        # 简化形式，考虑更多物理因素
        
        if log_ratio < 0.05:
            # 周期非常接近，相关性极高
            return 0.99
        elif log_ratio < 0.2:
            # 非常接近的周期
            return 0.98 - 0.01 * (log_ratio / 0.2)
        elif log_ratio < 0.5:
            # 接近的周期
            return 0.95 - 0.05 * ((log_ratio - 0.2) / 0.3)
        elif log_ratio < 1.0:
            # 中等距离的周期
            return 0.85 - 0.15 * ((log_ratio - 0.5) / 0.5)
        elif log_ratio < 1.5:
            # 相距较远的周期
            return 0.70 - 0.15 * ((log_ratio - 1.0) / 0.5)
        elif log_ratio < 2.0:
            # 相距远的周期
            return 0.55 - 0.15 * ((log_ratio - 1.5) / 0.5)
        else:
            # 非常远的周期，相关性很低
            return max(0.3 - 0.1 * (log_ratio - 2.0), 0.1)
    
    def export_hazard_results(self, output_dir: str = "."):
        """导出危险性分析结果"""
        if self.uhs_50yr_20p is None or self.cms is None:
            print("请先计算一致危险谱和条件均值谱")
            return
        
        # 导出一致危险谱
        uhs_filename = "一致危险谱_50年20%.csv"
        try:
            with open(uhs_filename, 'w', encoding='utf-8') as f:
                f.write("# 一致危险谱 (50年超越概率20%)\n")
                f.write("# 周期(s), 谱加速度(cm/s^2)\n")
                for period in sorted(self.uhs_50yr_20p.keys()):
                    f.write(f"{period:.3f}, {self.uhs_50yr_20p[period]:.6e}\n")
            print(f"✓ 一致危险谱已导出到: {uhs_filename}")
        except IOError as e:
            print(f"✗ 导出失败: {str(e)}")
        
        # 导出条件均值谱
        cms_filename = "条件均值谱.csv"
        try:
            with open(cms_filename, 'w', encoding='utf-8') as f:
                f.write("# 条件均值谱\n")
                f.write("# 周期(s), 谱加速度(cm/s^2)\n")
                for period in sorted(self.cms.keys()):
                    f.write(f"{period:.3f}, {self.cms[period]:.6e}\n")
            print(f"✓ 条件均值谱已导出到: {cms_filename}")
        except IOError as e:
            print(f"✗ 导出失败: {str(e)}")
        
        # 导出危险性曲线
        hazard_filename = "危险性曲线.csv"
        try:
            with open(hazard_filename, 'w', encoding='utf-8') as f:
                f.write("# 各周期危险性曲线\n")
                f.write("周期(s), 谱加速度(cm/s^2), 年超越概率\n")
                for period, curve in self.hazard_curves.items():
                    for sa, prob in curve:
                        f.write(f"{period:.3f}, {sa:.6e}, {prob:.6e}\n")
            print(f"✓ 危险性曲线已导出到: {hazard_filename}")
        except IOError as e:
            print(f"✗ 导出失败: {str(e)}")


# ============================================================================
# CSV 数据导出函数
# ============================================================================

def export_ground_motion_data(results: Dict, region_bounds: Dict, output_dir: str = "."):
    """
    导出地震动加速度谱数据到 CSV 文件
    """
    
    # 生成频率数组：0.1 到 10 Hz，间隔 0.1 Hz
    frequencies = [round(0.1 + i * 0.1, 2) for i in range(100)]
    
    # 导出地震事件和地震动谱
    csv_filename = "地震动加速度谱.csv"
    
    try:
        with open(csv_filename, 'w', encoding='utf-8') as f:
            # 写入头部信息
            f.write("# 地震模拟系统 - 地震动加速度谱数据\n")
            f.write("# 模拟参数\n")
            f.write(f"# 总模拟次数: {results['n_simulations']}\n")
            f.write(f"# 时间跨度: {results['time_period']} 年\n")
            f.write(f"# 区域范围: 经度 [{region_bounds['lon_min']}, {region_bounds['lon_max']}], 纬度 [{region_bounds['lat_min']}, {region_bounds['lat_max']}]\n")
            f.write("\n")
            
            # 处理第一次模拟的数据
            if results['simulations']:
                sim = results['simulations'][0]
                
                # 对每个事件写入数据
                for event_idx in range(len(sim['occurrence_times'])):
                    mag = sim['magnitudes'][event_idx]
                    lon = sim['longitudes'][event_idx]
                    lat = sim['latitudes'][event_idx]
                    depth = sim['depths'][event_idx]
                    time = sim['occurrence_times'][event_idx]
                    
                    # 计算到台站的距离
                    station_lon = (region_bounds['lon_min'] + region_bounds['lon_max']) / 2
                    station_lat = (region_bounds['lat_min'] + region_bounds['lat_max']) / 2
                    
                    lat1_rad = math.radians(lat)
                    lon1_rad = math.radians(lon)
                    lat2_rad = math.radians(station_lat)
                    lon2_rad = math.radians(station_lon)
                    dlat = lat2_rad - lat1_rad
                    dlon = lon2_rad - lon1_rad
                    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
                    c = 2 * math.asin(math.sqrt(a))
                    distance = 6371 * c
                    
                    # 写入事件信息和加速度谱
                    f.write(f"事件 {event_idx + 1}\n")
                    f.write(f"震级(M), 时间(年), 经度(deg), 纬度(deg), 深度(km), 距离(km)\n")
                    f.write(f"{mag:.2f}, {time:.2f}, {lon:.4f}, {lat:.4f}, {depth:.2f}, {distance:.2f}\n")
                    f.write("\n")
                    f.write("频率(Hz), 加速度谱(cm/s^2)\n")
                    
                    for freq in frequencies:
                        # 添加随机波动
                        epsilon = random.gauss(0, 1)
                        acc = improved_ground_motion_model(freq, mag, distance, random_seed=epsilon)
                        f.write(f"{freq:.1f}, {acc:.6e}\n")
                    
                    f.write("\n")
        
        print(f"✓ 加速度谱数据已导出到: {csv_filename}")
    
    except IOError as e:
        print(f"✗ 导出失败: {str(e)}")
    
    # 导出事件汇总表
    csv_summary = "地震事件汇总.csv"
    
    try:
        with open(csv_summary, 'w', encoding='utf-8') as f:
            f.write("事件号, 发生时间(年), 震级(M), 经度(deg), 纬度(deg), 深度(km)\n")
            
            if results['simulations']:
                sim = results['simulations'][0]
                
                for event_idx in range(len(sim['occurrence_times'])):
                    f.write(f"{event_idx + 1}, {sim['occurrence_times'][event_idx]:.2f}, {sim['magnitudes'][event_idx]:.2f}, "
                           f"{sim['longitudes'][event_idx]:.4f}, {sim['latitudes'][event_idx]:.4f}, {sim['depths'][event_idx]:.2f}\n")
        
        print(f"✓ 事件汇总数据已导出到: {csv_summary}")
    
    except IOError as e:
        print(f"✗ 导出失败: {str(e)}")


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("纯 Python 版地震模拟系统 - 包含危险性分析")
    print("（计算50年超越概率20%的一致危险谱和条件均值谱）")
    print("=" * 70 + "\n")
    
    # 1. 定义研究区域
    print("[1] 初始化研究区域...")
    region_bounds = {
        'lon_min': -90.5,
        'lon_max': -89.5,
        'lat_min': 35.0,
        'lat_max': 35.5,
        'depth_min': 5,
        'depth_max': 25
    }
    print(f"    区域范围: 经度 [{region_bounds['lon_min']}, {region_bounds['lon_max']}], "
          f"纬度 [{region_bounds['lat_min']}, {region_bounds['lat_max']}]")
    
    # 2. 创建模拟器
    print("\n[2] 创建地震蒙特卡洛模拟器...")
    simulator = SeismicMonteCarloSimple(region_bounds, time_period=500.0)  # 改进：更长的时间跨度用于更好的统计
    
    # 3. 设置参数
    print("[3] 设置地震统计参数...")
    simulator.set_seismic_parameters(
        annual_rate=0.08,  # 改进：更现实的年均发生率
        b_value=0.95,  # 改进：更符合美国中部地震带特征
        M_min=4.5,  # 改进：提高最小震级（检测阈值）
        M_max=8.2,  # 改进：考虑更大的最大可能地震
        depth_params={'type': 'gamma', 'mean': 13, 'std': 7},  # 改进：使用Gamma分布
        spatial_params={'type': 'uniform'}
    )
    print("    年均发生率: 0.08 次/年")
    print("    b 值: 0.95")
    print("    震级范围: 4.5 - 8.2")
    print("    深度分布: Gamma分布 (μ=13km, σ=7km)")
    
    # 4. 运行模拟
    print("\n[4] 运行蒙特卡洛地震事件模拟...")
    results = simulator.run_monte_carlo_simulation(n_simulations=5, random_seed=42)
    
    # 5. 获取统计
    print("\n[5] 获取统计摘要...")
    summary = simulator.get_summary_statistics()
    
    print("\n" + "=" * 70)
    print("模拟结果统计")
    print("=" * 70)
    print(f"总模拟次数: {summary['total_simulations']}")
    print(f"每次模拟平均事件数: {summary['mean_events_per_simulation']:.2f} +/- {summary['std_events_per_simulation']:.2f}")
    print(f"事件数范围: {summary['min_events_per_simulation']} - {summary['max_events_per_simulation']}")
    print()
    print("震级统计:")
    print(f"  平均: {summary['magnitude_stats']['mean']:.2f}")
    print(f"  标准差: {summary['magnitude_stats']['std']:.2f}")
    print(f"  范围: {summary['magnitude_stats']['min']:.2f} - {summary['magnitude_stats']['max']:.2f}")
    print()
    print("深度统计 (km):")
    print(f"  平均: {summary['depth_stats']['mean']:.2f}")
    print(f"  标准差: {summary['depth_stats']['std']:.2f}")
    print(f"  范围: {summary['depth_stats']['min']:.2f} - {summary['depth_stats']['max']:.2f}")
    
    # 6. 危险性分析
    print("\n" + "=" * 70)
    print("地震危险性分析")
    print("=" * 70)
    
    print("\n[6] 进行地震危险性分析...")
    analyzer = SeismicHazardAnalyzer(results, region_bounds)
    
    # 计算危险性曲线
    hazard_curves = analyzer.calculate_hazard_curves()
    print(f"✓ 已计算 {len(hazard_curves)} 个周期的危险性曲线")
    
    # 计算50年超越概率20%的一致危险谱
    # 年超越概率 = 1 - (1-0.2)^(1/50) = 0.0044
    uhs = analyzer.calculate_uniform_hazard_spectrum(target_annual_probability=0.0044)
    print(f"✓ 已计算一致危险谱 (50年超越概率20%)")
    
    # 计算条件均值谱 (控制周期1.0s)
    cms = analyzer.calculate_conditional_mean_spectrum(control_period=1.0)
    print(f"✓ 已计算条件均值谱")
    
    # 显示结果
    print("\n" + "=" * 70)
    print("一致危险谱 (50年超越概率20%)")
    print("=" * 70)
    print("周期(s)   谱加速度(cm/s^2)")
    print("-" * 30)
    for period in sorted(uhs.keys()):
        print(f"{period:>6.2f}    {uhs[period]:>12.4e}")
    
    print("\n" + "=" * 70)
    print("条件均值谱 (控制周期1.0s)")
    print("=" * 70)
    print("周期(s)   谱加速度(cm/s^2)")
    print("-" * 30)
    for period in sorted(cms.keys()):
        print(f"{period:>6.2f}    {cms[period]:>12.4e}")
    
    # 7. 导出危险性分析结果
    print("\n" + "=" * 70)
    print("导出危险性分析结果")
    print("=" * 70)
    
    analyzer.export_hazard_results()
    
    # 8. 导出地震动数据
    print("\n" + "=" * 70)
    print("导出地震动数据")
    print("=" * 70)
    
    export_ground_motion_data(results, region_bounds)
    
    print("\n" + "=" * 70)
    print("✓ 纯 Python 版地震危险性分析完成！")
    print("=" * 70)
    print("\n说明:")
    print("  1. 一致危险谱(UHS): 各周期具有相同超越概率(50年20%)的反应谱")
    print("  2. 条件均值谱(CMS): 在控制周期(1.0s)谱值给定的条件下，其他周期的条件均值反应谱")
    print("  3. 危险性曲线: 各周期谱加速度与年超越概率的关系曲线")
    print("\n文件输出:")
    print("  - 一致危险谱_50年20%.csv: 一致危险谱数据")
    print("  - 条件均值谱.csv: 条件均值谱数据") 
    print("  - 危险性曲线.csv: 各周期危险性曲线")
    print("  - 地震动加速度谱.csv: 详细的地震动数据")
    print("  - 地震事件汇总.csv: 地震事件列表")
    print("\n注意:")
    print("  这是一个演示版本，实际应用中需要使用更精确的地震动预测方程和相关系数模型")
    print("  建议使用专业的地震危险性分析软件进行实际工程应用")


if __name__ == "__main__":
    main()