#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
纯 Python 版本的完整地震模拟系统
无需 NumPy C 扩展依赖，用于验证代码逻辑
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
        """生成深度分布"""
        depth_params = self.seismic_params['depth_distribution']
        depth_min = self.region_bounds['depth_min']
        depth_max = self.region_bounds['depth_max']
        
        if depth_params['type'] == 'lognormal':
            mean = depth_params.get('mean', 10)
            std = depth_params.get('std', 5)
            depths = []
            for _ in range(n_events):
                # 简化的对数正态分布采样
                u1, u2 = random.random(), random.random()
                z = self.math_lib.sqrt(-2 * self.math_lib.log(u1)) * self.math_lib.cos(2 * self.math_lib.pi() * u2)
                mu = self.math_lib.log(mean ** 2 / self.math_lib.sqrt(std ** 2 + mean ** 2))
                sigma = self.math_lib.sqrt(self.math_lib.log(1 + std ** 2 / mean ** 2))
                depth = self.math_lib.exp(mu + sigma * z)
                depths.append(depth)
        else:
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

def simple_ground_motion_model(frequency: float, magnitude: float, distance: float) -> float:
    """简化的点源模型地震动计算"""
    math_lib = SimpleMath()
    
    # 简化的 AB95 模型
    rho = 2.8
    Vs = 3.5
    M0 = 10 ** (1.5 * magnitude + 9.05)
    
    # 拐角频率
    stress_drop = 100
    f_c = 4.9e6 * Vs * 1000 * (stress_drop / M0) ** (1/3)
    
    # 位移谱
    if frequency < f_c:
        disp_spectrum = M0 / (1 + (frequency / f_c) ** 2)
    else:
        disp_spectrum = M0 / (1 + (frequency / f_c) ** 2) * (10 / frequency) ** 2
    
    # 路径效应
    path_effect = 1.0 / max(distance, 1.0)
    
    # 衰减效应
    kappa = 0.04
    attenuation = math_lib.exp_negative(math.pi * frequency * kappa)
    
    # 加速度谱
    acceleration = (2 * math.pi * frequency) ** 2 * disp_spectrum * path_effect * attenuation
    
    return max(acceleration, 0)


# ============================================================================
# ============================================================================
# CSV 数据导出函数
# ============================================================================

def export_ground_motion_data(results: Dict, region_bounds: Dict, output_dir: str = "."):
    """
    导出地震动加速度谱数据到 CSV 文件
    
    参数:
    results (dict): 模拟结果字典
    region_bounds (dict): 研究区域边界
    output_dir (str): 输出目录
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
                        acc = simple_ground_motion_model(freq, mag, distance)
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
    print("纯 Python 版地震模拟系统 - 演示版本")
    print("（无需 NumPy/SciPy 编译依赖）")
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
    simulator = SeismicMonteCarloSimple(region_bounds, time_period=100.0)
    
    # 3. 设置参数
    print("[3] 设置地震统计参数...")
    simulator.set_seismic_parameters(
        annual_rate=0.05,
        b_value=1.0,
        M_min=4.0,
        M_max=8.0,
        depth_params={'type': 'lognormal', 'mean': 12, 'std': 6},
        spatial_params={'type': 'uniform'}
    )
    print("    年均发生率: 0.05 次/年")
    print("    b 值: 1.0")
    print("    震级范围: 4.0 - 8.0")
    
    # 4. 运行模拟
    print("\n[4] 运行蒙特卡洛地震事件模拟...")
    results = simulator.run_monte_carlo_simulation(n_simulations=3, random_seed=42)
    
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
    
    # 6. 地震动模拟示例
    print("\n" + "=" * 70)
    print("地震动模型示例（预设工况）")
    print("=" * 70)
    
    test_cases = [
        (0.1, 5.0, 10.0, "低频, M5.0, 距离 10km"),
        (1.0, 6.0, 20.0, "中频, M6.0, 距离 20km"),
        (10.0, 7.0, 50.0, "高频, M7.0, 距离 50km"),
    ]
    
    print("\n加速度谱计算示例 (cm/s^2):")
    for freq, mag, dist, desc in test_cases:
        acc = simple_ground_motion_model(freq, mag, dist)
        print(f"  {desc}: {acc:.6e} cm/s^2")
    
    # 6.5 为实际模拟的地震事件计算地震动谱
    print("\n" + "=" * 70)
    print("实际地震事件的加速度谱")
    print("=" * 70)
    
    if results['simulations']:
        sim = results['simulations'][0]
        
        if len(sim['occurrence_times']) > 0:
            print(f"\n第 1 次模拟共有 {len(sim['occurrence_times'])} 个事件，计算频率范围 0.1-10 Hz 的加速度谱：\n")
            
            # 定义频率数组：0.1 到 10 Hz，间隔 0.1 Hz
            frequencies = [round(0.1 + i * 0.1, 2) for i in range(100)]
            
            # 对每个事件计算谱
            for event_idx in range(min(3, len(sim['occurrence_times']))):  # 显示前 3 个事件
                mag = sim['magnitudes'][event_idx]
                lon = sim['longitudes'][event_idx]
                lat = sim['latitudes'][event_idx]
                depth = sim['depths'][event_idx]
                time = sim['occurrence_times'][event_idx]
                
                # 假设台站位置为研究区域中心
                station_lon = (region_bounds['lon_min'] + region_bounds['lon_max']) / 2
                station_lat = (region_bounds['lat_min'] + region_bounds['lat_max']) / 2
                
                # 计算震源到台站的距离（简化的大圆距离）
                lat1_rad = math.radians(lat)
                lon1_rad = math.radians(lon)
                lat2_rad = math.radians(station_lat)
                lon2_rad = math.radians(station_lon)
                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad
                a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
                c = 2 * math.asin(math.sqrt(a))
                distance = 6371 * c  # 地球半径，单位 km
                
                print(f"事件 {event_idx + 1}：M{mag:.2f}，发生时间 {time:.1f} 年，深度 {depth:.1f} km，距离 {distance:.1f} km")
                print(f"  频率(Hz)   加速度谱(cm/s^2)")
                print(f"  " + "-" * 40)
                
                for freq in frequencies:
                    acc = simple_ground_motion_model(freq, mag, distance)
                    print(f"  {freq:>6.1f}     {acc:>12.4e}")
                print()
    
    # 7. 输出详细事件列表
    print("\n" + "=" * 70)
    print("第 1 次模拟事件列表（前 10 个事件）")
    print("=" * 70)
    
    if results['simulations']:
        sim = results['simulations'][0]
        n_events = min(10, len(sim['occurrence_times']))
        
        # 表头和分隔线
        print("\n" + "事件号  时间(年)    震级    经度      纬度      深度(km)")
        print("-" * 55)
        
        # 数据行
        for i in range(n_events):
            print(f"{i+1:<6} {sim['occurrence_times'][i]:>10.2f} {sim['magnitudes'][i]:>7.2f} {sim['longitudes'][i]:>8.2f} {sim['latitudes'][i]:>8.2f} {sim['depths'][i]:>9.2f}")
        
        if len(sim['occurrence_times']) > 10:
            print(f"... 共 {len(sim['occurrence_times'])} 个事件")
    
    # 8. 导出数据到 CSV 文件
    print("\n" + "=" * 70)
    print("导出数据到 CSV 文件")
    print("=" * 70)
    
    export_ground_motion_data(results, region_bounds)
    
    # 8.5 绘制加速度谱图形
    print("\n" + "=" * 70)
    print("✓ 纯 Python 版演示完成！")
    print("=" * 70)
    print("\n说明:")
    print("  这是一个不依赖 NumPy C 扩展的纯 Python 版本。")
    print("  可以验证代码逻辑是否正确，地震模型是否工作正常。")
    print("\n生成的文件:")
    print("  - 地震动加速度谱.csv: 加速度谱数据")
    print("  - 地震事件汇总.csv: 地震事件列表")
    print("\n可视化步骤:")
    print("  1. 本脚本生成 CSV 数据文件")
    print("  2. 使用独立工具进行可视化: python 可视化加速度谱.py")
    print("  3. 生成的图形支持两种格式:")
    print("     - PNG 格式 (matplotlib 模式，高精度)")
    print("     - SVG 格式 (纯 Python 模式，无依赖)")
    print()


if __name__ == "__main__":
    main()
