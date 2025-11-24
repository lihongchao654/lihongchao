# ============================================================================
# 处理 NumPy/科学库导入问题的容错机制
# ============================================================================
import sys
import warnings
warnings.filterwarnings('ignore')

# 尝试导入 NumPy，若失败则给出诊断信息
try:
    import numpy as np
except ImportError as e:
    print("=" * 70)
    print("错误：无法导入 NumPy")
    print("=" * 70)
    print(f"详细错误信息：{str(e)}")
    print("")
    print("这通常由以下原因引起（按常见度排序）：")
    print("  1. 缺少 Windows 运行时库（Microsoft Visual C++ Redistributable）")
    print("  2. Python 版本与 NumPy 二进制版本不兼容")
    print("  3. 环境变量或 PATH 配置问题")
    print("")
    print("建议的解决方案：")
    print("  方案 A：安装 Microsoft Visual C++ 2015-2022 Redistributable (x64)")
    print("         下载链接：https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist")
    print("  方案 B：使用 Conda 创建隔离环境（更稳妥）")
    print("         命令：conda create -n seismic python=3.11 -y")
    print("              conda activate seismic")
    print("              conda install numpy pandas scipy matplotlib -y")
    print("=" * 70)
    sys.exit(1)

try:
    import pandas as pd
except ImportError as e:
    print(f"错误：无法导入 Pandas - {str(e)}")
    sys.exit(1)

try:
    from scipy import stats, integrate
except ImportError as e:
    print(f"错误：无法导入 SciPy - {str(e)}")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"错误：无法导入 Matplotlib - {str(e)}")
    sys.exit(1)

from typing import Dict, List, Tuple, Optional, Union
import math

class CompleteSeismicSimulation:
    """
    完整的地震模拟系统
    整合了蒙特卡洛地震事件模拟和地震动模拟
    """
    
    def __init__(self, region_bounds: Dict, time_period: float = 1000.0):
        """
        初始化完整地震模拟系统
        
        参数:
        region_bounds (dict): 研究区域边界
        time_period (float): 模拟的时间段（年）
        """
        self.region_bounds = region_bounds
        self.time_period = time_period
        self.mc_simulator = SeismicMonteCarlo(region_bounds, time_period)
        
        # 存储所有模拟结果
        self.complete_results = None
        
    def set_seismic_parameters(self, annual_rate: float, b_value: float, 
                             M_min: float, M_max: float, 
                             depth_params: Dict, spatial_params: Dict):
        """
        设置地震统计参数
        """
        self.mc_simulator.set_seismic_parameters(
            annual_rate, b_value, M_min, M_max, depth_params, spatial_params
        )
    
    def run_complete_simulation(self, n_simulations: int = 1, 
                               station_locations: List[Tuple] = None,
                               frequencies: np.ndarray = None,
                               random_seed: Optional[int] = None) -> Dict:
        """
        运行完整的地震模拟（事件模拟 + 地震动模拟）
        
        参数:
        n_simulations (int): 模拟次数
        station_locations (list): 台站位置列表 [(lon, lat, elevation), ...]
        frequencies (np.array): 频率数组
        random_seed (int): 随机种子
        
        返回:
        dict: 完整模拟结果
        """
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 设置默认参数
        if frequencies is None:
            frequencies = np.logspace(-2, 2, 200)  # 0.01 到 100 Hz
            
        if station_locations is None:
            # 默认在研究区域中心设置一个台站
            center_lon = (self.region_bounds['lon_min'] + self.region_bounds['lon_max']) / 2
            center_lat = (self.region_bounds['lat_min'] + self.region_bounds['lat_max']) / 2
            station_locations = [(center_lon, center_lat, 0)]
        
        print("开始蒙特卡洛地震事件模拟...")
        # 1. 运行蒙特卡洛地震事件模拟
        mc_results = self.mc_simulator.run_monte_carlo_simulation(
            n_simulations, random_seed
        )
        
        print("开始地震动模拟...")
        # 2. 对每个模拟中的每个地震事件进行地震动模拟
        complete_simulations = []
        
        for sim_idx, mc_sim in enumerate(mc_results['simulations']):
            print(f"处理模拟 #{sim_idx + 1}/{len(mc_results['simulations'])}")
            
            simulation_events = []
            
            for event_idx in range(len(mc_sim['occurrence_times'])):
                event_data = {
                    'occurrence_time': mc_sim['occurrence_times'][event_idx],
                    'magnitude': mc_sim['magnitudes'][event_idx],
                    'longitude': mc_sim['longitudes'][event_idx],
                    'latitude': mc_sim['latitudes'][event_idx],
                    'depth': mc_sim['depths'][event_idx],
                    'station_results': {}
                }
                
                # 对每个台站进行地震动模拟
                for station_idx, station_loc in enumerate(station_locations):
                    # 计算震源到台站的距离
                    distance = self._calculate_distance(
                        (mc_sim['longitudes'][event_idx], mc_sim['latitudes'][event_idx]),
                        (station_loc[0], station_loc[1])
                    )
                    
                    # 根据震级选择模型
                    if mc_sim['magnitudes'][event_idx] < 7.0:
                        # 使用点源模型
                        ground_motion_result = self._run_point_source_model(
                            frequencies, mc_sim['magnitudes'][event_idx], distance
                        )
                        model_used = "point_source"
                    else:
                        # 使用混合震源模型
                        ground_motion_result = self._run_hybrid_source_model(
                            frequencies, mc_sim['magnitudes'][event_idx], 
                            (mc_sim['longitudes'][event_idx], mc_sim['latitudes'][event_idx]),
                            station_loc
                        )
                        model_used = "hybrid_source"
                    
                    event_data['station_results'][f'station_{station_idx}'] = {
                        'ground_motion': ground_motion_result,
                        'model_used': model_used,
                        'distance': distance,
                        'station_location': station_loc
                    }
                
                simulation_events.append(event_data)
            
            complete_simulations.append(simulation_events)
        
        # 存储完整结果
        self.complete_results = {
            'monte_carlo_results': mc_results,
            'complete_simulations': complete_simulations,
            'station_locations': station_locations,
            'frequencies': frequencies,
            'simulation_parameters': {
                'n_simulations': n_simulations,
                'time_period': self.time_period,
                'region_bounds': self.region_bounds
            }
        }
        
        print("完整模拟完成!")
        return self.complete_results
    
    def _calculate_distance(self, event_location: Tuple, station_location: Tuple) -> float:
        """
        计算震源到台站的距离（简化的大圆距离）
        
        参数:
        event_location (tuple): 震源位置 (lon, lat)
        station_location (tuple): 台站位置 (lon, lat)
        
        返回:
        float: 距离 (km)
        """
        # 将经纬度转换为弧度
        lat1 = math.radians(event_location[1])
        lon1 = math.radians(event_location[0])
        lat2 = math.radians(station_location[1])
        lon2 = math.radians(station_location[0])
        
        # 大圆距离公式
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # 地球半径 (km)
        r = 6371.0
        
        return c * r
    
    def _run_point_source_model(self, frequencies: np.ndarray, Mw: float, 
                              distance: float) -> Dict:
        """
        运行点源模型
        
        参数:
        frequencies (np.array): 频率数组
        Mw (float): 矩震级
        distance (float): 震源距 (km)
        
        返回:
        dict: 点源模型结果
        """
        return ab95_bc_point_source_model(frequencies, Mw, distance=distance)
    
    def _run_hybrid_source_model(self, frequencies: np.ndarray, Mw: float,
                               event_location: Tuple, station_location: Tuple) -> Dict:
        """
        运行混合震源模型
        
        参数:
        frequencies (np.array): 频率数组
        Mw (float): 矩震级
        event_location (tuple): 事件位置 (lon, lat)
        station_location (tuple): 台站位置 (lon, lat, elevation)
        
        返回:
        dict: 混合源模型结果
        """
        # 根据震级估算断层尺寸 (Wells & Coppersmith, 1994 经验关系)
        fault_length, fault_width = self._estimate_fault_dimensions(Mw)
        subfault_size = min(fault_length, fault_width) / 10  # 子断层大小
        
        # 创建混合源模型
        fault_params = {
            'fault_length': fault_length,
            'fault_width': fault_width,
            'subfault_size': subfault_size,
            'Mw': Mw,
            'hypocenter': (0.5, 0.5)  # 简化的相对位置
        }
        
        hybrid_model = HybridSourceModel(**fault_params)
        
        # 计算混合源谱
        result = hybrid_model.compute_hybrid_source_spectrum(frequencies, station_location)
        
        return result
    
    def _estimate_fault_dimensions(self, Mw: float) -> Tuple[float, float]:
        """
        根据震级估算断层尺寸 (Wells & Coppersmith, 1994)
        
        参数:
        Mw (float): 矩震级
        
        返回:
        tuple: (断层长度, 断层宽度) (km)
        """
        # 走滑断层的关系式
        if Mw < 5.0:
            length = 10 ** (0.69 * Mw - 3.22)
            width = 10 ** (0.32 * Mw - 1.01)
        else:
            length = 10 ** (-2.57 + 0.62 * Mw)
            width = 10 ** (-1.14 + 0.35 * Mw)
        
        return max(length, 5.0), max(width, 3.0)  # 确保最小尺寸
    
    def get_detailed_summary(self) -> Dict:
        """
        获取详细的模拟统计摘要
        """
        if self.complete_results is None:
            raise ValueError("请先运行完整模拟")
        
        mc_summary = self.mc_simulator.get_summary_statistics()
        
        # 统计模型使用情况
        model_usage = {'point_source': 0, 'hybrid_source': 0}
        ground_motion_stats = []
        
        for sim in self.complete_results['complete_simulations']:
            for event in sim:
                for station_key, station_result in event['station_results'].items():
                    model_usage[station_result['model_used']] += 1
                    
                    gm_data = station_result['ground_motion']
                    if 'acceleration_spectrum' in gm_data:
                        max_acc = np.max(np.abs(gm_data['acceleration_spectrum']))
                        ground_motion_stats.append({
                            'magnitude': event['magnitude'],
                            'distance': station_result['distance'],
                            'max_acceleration': max_acc,
                            'model': station_result['model_used']
                        })
        
        ground_motion_df = pd.DataFrame(ground_motion_stats)
        
        summary = {
            'monte_carlo_summary': mc_summary,
            'model_usage': model_usage,
            'total_ground_motion_simulations': len(ground_motion_stats),
            'ground_motion_statistics': {
                'mean_max_acceleration': np.mean(ground_motion_df['max_acceleration']) if len(ground_motion_df) > 0 else 0,
                'max_max_acceleration': np.max(ground_motion_df['max_acceleration']) if len(ground_motion_df) > 0 else 0,
                'min_max_acceleration': np.min(ground_motion_df['max_acceleration']) if len(ground_motion_df) > 0 else 0,
            }
        }
        
        return summary
    
    def plot_comprehensive_results(self, simulation_idx: int = 0, 
                                 station_idx: int = 0,
                                 save_path: Optional[str] = None):
        """
        绘制综合模拟结果
        
        参数:
        simulation_idx (int): 模拟索引
        station_idx (int): 台站索引
        save_path (str): 图像保存路径
        """
        if self.complete_results is None:
            raise ValueError("请先运行完整模拟")
        
        mc_sim = self.complete_results['monte_carlo_results']['simulations'][simulation_idx]
        complete_sim = self.complete_results['complete_simulations'][simulation_idx]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. 空间分布图
        if len(mc_sim['longitudes']) > 0:
            colors = ['blue' if m < 7.0 else 'red' for m in mc_sim['magnitudes']]
            sizes = [50 + (m - 4) * 20 for m in mc_sim['magnitudes']]
            
            scatter = axes[0, 0].scatter(mc_sim['longitudes'], mc_sim['latitudes'], 
                                       c=colors, s=sizes, alpha=0.7)
            axes[0, 0].set_xlabel('经度')
            axes[0, 0].set_ylabel('纬度')
            axes[0, 0].set_title('地震空间分布\n(蓝色:点源模型, 红色:混合源模型)')
            
            # 添加台站位置
            station_loc = self.complete_results['station_locations'][station_idx]
            axes[0, 0].plot(station_loc[0], station_loc[1], 'k^', markersize=15, label='台站')
            axes[0, 0].legend()
        
        # 2. 震级分布直方图
        if len(mc_sim['magnitudes']) > 0:
            axes[0, 1].hist(mc_sim['magnitudes'], bins=20, alpha=0.7, edgecolor='black')
            axes[0, 1].axvline(x=7.0, color='red', linestyle='--', label='模型分界 (M7.0)')
            axes[0, 1].set_xlabel('震级')
            axes[0, 1].set_ylabel('频数')
            axes[0, 1].set_title('震级分布')
            axes[0, 1].legend()
        
        # 3. 模型使用统计
        model_usage = {'point_source': 0, 'hybrid_source': 0}
        for event in complete_sim:
            for station_result in event['station_results'].values():
                model_usage[station_result['model_used']] += 1
        
        axes[0, 2].bar(model_usage.keys(), model_usage.values(), color=['blue', 'red'])
        axes[0, 2].set_ylabel('使用次数')
        axes[0, 2].set_title('震源模型使用统计')
        
        # 4. 示例地震动谱
        if len(complete_sim) > 0:
            # 选择一个事件展示地震动谱
            example_event = complete_sim[0]
            station_key = f'station_{station_idx}'
            
            if station_key in example_event['station_results']:
                gm_result = example_event['station_results'][station_key]['ground_motion']
                frequencies = self.complete_results['frequencies']
                
                if 'acceleration_spectrum' in gm_result:
                    axes[1, 0].loglog(frequencies, np.abs(gm_result['acceleration_spectrum']))
                    axes[1, 0].set_xlabel('频率 (Hz)')
                    axes[1, 0].set_ylabel('加速度谱')
                    axes[1, 0].set_title(f'示例地震动谱 (M{example_event["magnitude"]:.1f})')
                
                if 'hybrid_spectrum' in gm_result:
                    axes[1, 0].loglog(frequencies, np.abs(gm_result['hybrid_spectrum']))
                    axes[1, 0].set_xlabel('频率 (Hz)')
                    axes[1, 0].set_ylabel('加速度谱')
                    axes[1, 0].set_title(f'示例地震动谱 (M{example_event["magnitude"]:.1f})')
        
        # 5. 最大加速度与震级关系
        magnitudes = []
        max_accelerations = []
        
        for event in complete_sim:
            station_key = f'station_{station_idx}'
            if station_key in event['station_results']:
                gm_result = event['station_results'][station_key]['ground_motion']
                if 'acceleration_spectrum' in gm_result:
                    max_acc = np.max(np.abs(gm_result['acceleration_spectrum']))
                    magnitudes.append(event['magnitude'])
                    max_accelerations.append(max_acc)
                elif 'hybrid_spectrum' in gm_result:
                    max_acc = np.max(np.abs(gm_result['hybrid_spectrum']))
                    magnitudes.append(event['magnitude'])
                    max_accelerations.append(max_acc)
        
        if len(magnitudes) > 0:
            axes[1, 1].scatter(magnitudes, max_accelerations, alpha=0.7)
            axes[1, 1].set_xlabel('震级')
            axes[1, 1].set_ylabel('最大加速度')
            axes[1, 1].set_title('震级-最大加速度关系')
            axes[1, 1].set_yscale('log')
        
        # 6. 时间序列图
        if len(mc_sim['occurrence_times']) > 0:
            colors = ['blue' if m < 7.0 else 'red' for m in mc_sim['magnitudes']]
            axes[1, 2].scatter(mc_sim['occurrence_times'], mc_sim['magnitudes'], c=colors, alpha=0.7)
            axes[1, 2].set_xlabel('时间 (年)')
            axes[1, 2].set_ylabel('震级')
            axes[1, 2].set_title('地震发生时间序列')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_complete_results(self, simulation_idx: int = 0) -> pd.DataFrame:
        """
        导出完整模拟结果
        
        参数:
        simulation_idx (int): 模拟索引
        
        返回:
        pd.DataFrame: 包含完整模拟结果的DataFrame
        """
        if self.complete_results is None:
            raise ValueError("请先运行完整模拟")
        
        complete_sim = self.complete_results['complete_simulations'][simulation_idx]
        
        results_data = []
        for event in complete_sim:
            for station_key, station_result in event['station_results'].items():
                gm_data = station_result['ground_motion']
                
                # 提取关键地震动参数
                if 'acceleration_spectrum' in gm_data:
                    acc_spectrum = gm_data['acceleration_spectrum']
                    max_acc = np.max(np.abs(acc_spectrum))
                    mean_acc = np.mean(np.abs(acc_spectrum))
                elif 'hybrid_spectrum' in gm_data:
                    acc_spectrum = gm_data['hybrid_spectrum']
                    max_acc = np.max(np.abs(acc_spectrum))
                    mean_acc = np.mean(np.abs(acc_spectrum))
                else:
                    max_acc = mean_acc = 0
                
                result_row = {
                    'simulation_id': simulation_idx,
                    'occurrence_time': event['occurrence_time'],
                    'magnitude': event['magnitude'],
                    'longitude': event['longitude'],
                    'latitude': event['latitude'],
                    'depth': event['depth'],
                    'station': station_key,
                    'model_used': station_result['model_used'],
                    'distance_km': station_result['distance'],
                    'max_acceleration': max_acc,
                    'mean_acceleration': mean_acc
                }
                
                results_data.append(result_row)
        
        return pd.DataFrame(results_data)


# =============================================================================
# 以下是从前面代码中复制的类定义，确保完整性
# =============================================================================

class SeismicMonteCarlo:
    """
    基于平稳泊松模型的地震蒙特卡洛模拟器
    """
    
    def __init__(self, region_bounds: Dict, time_period: float = 1000.0):
        self.region_bounds = region_bounds
        self.time_period = time_period
        
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
    
    def gutenberg_richter_magnitude(self, n_events: int) -> np.ndarray:
        b_value = self.seismic_params['b_value']
        M_min = self.seismic_params['M_min']
        M_max = self.seismic_params['M_max']
        
        beta = b_value * np.log(10)
        C = 1 - np.exp(-beta * (M_max - M_min))
        
        u = np.random.uniform(0, 1, n_events)
        magnitudes = M_min - (1/beta) * np.log(1 - u * C)
        
        return magnitudes
    
    def generate_depth_distribution(self, n_events: int) -> np.ndarray:
        depth_params = self.seismic_params['depth_distribution']
        depth_min = self.region_bounds['depth_min']
        depth_max = self.region_bounds['depth_max']
        
        if depth_params['type'] == 'lognormal':
            mean = depth_params.get('mean', 10)
            std = depth_params.get('std', 5)
            mu = np.log(mean**2 / np.sqrt(std**2 + mean**2))
            sigma = np.sqrt(np.log(1 + (std**2 / mean**2)))
            depths = np.random.lognormal(mu, sigma, n_events)
        elif depth_params['type'] == 'normal':
            mean = depth_params.get('mean', 10)
            std = depth_params.get('std', 5)
            depths = np.random.normal(mean, std, n_events)
        elif depth_params['type'] == 'uniform':
            depths = np.random.uniform(depth_min, depth_max, n_events)
        else:
            depths = np.random.uniform(depth_min, depth_max, n_events)
        
        depths = np.clip(depths, depth_min, depth_max)
        return depths
    
    def generate_spatial_distribution(self, n_events: int) -> Tuple[np.ndarray, np.ndarray]:
        spatial_params = self.seismic_params['spatial_distribution']
        lon_min = self.region_bounds['lon_min']
        lon_max = self.region_bounds['lon_max']
        lat_min = self.region_bounds['lat_min']
        lat_max = self.region_bounds['lat_max']
        
        if spatial_params['type'] == 'uniform':
            longitudes = np.random.uniform(lon_min, lon_max, n_events)
            latitudes = np.random.uniform(lat_min, lat_max, n_events)
        elif spatial_params['type'] == 'normal':
            center_lon = spatial_params.get('center_lon', (lon_min + lon_max) / 2)
            center_lat = spatial_params.get('center_lat', (lat_min + lat_max) / 2)
            std_lon = spatial_params.get('std_lon', (lon_max - lon_min) / 6)
            std_lat = spatial_params.get('std_lat', (lat_max - lat_min) / 6)
            longitudes = np.random.normal(center_lon, std_lon, n_events)
            latitudes = np.random.normal(center_lat, std_lat, n_events)
            longitudes = np.clip(longitudes, lon_min, lon_max)
            latitudes = np.clip(latitudes, lat_min, lat_max)
        else:
            longitudes = np.random.uniform(lon_min, lon_max, n_events)
            latitudes = np.random.uniform(lat_min, lat_max, n_events)
        
        return longitudes, latitudes
    
    def generate_occurrence_times(self, n_events: int) -> np.ndarray:
        annual_rate = self.seismic_params['annual_rate']
        
        if annual_rate <= 0:
            raise ValueError("年均发生率必须大于0")
        
        inter_event_times = np.random.exponential(1/annual_rate, n_events)
        occurrence_times = np.cumsum(inter_event_times)
        valid_indices = occurrence_times <= self.time_period
        occurrence_times = occurrence_times[valid_indices]
        
        return occurrence_times
    
    def run_monte_carlo_simulation(self, n_simulations: int = 1, 
                                 random_seed: Optional[int] = None) -> Dict:
        if random_seed is not None:
            np.random.seed(random_seed)
        
        all_simulations = []
        
        for sim_idx in range(n_simulations):
            annual_rate = self.seismic_params['annual_rate']
            expected_events = annual_rate * self.time_period
            n_events = np.random.poisson(expected_events)
            
            if n_events == 0:
                simulation = {
                    'occurrence_times': np.array([]),
                    'magnitudes': np.array([]),
                    'longitudes': np.array([]),
                    'latitudes': np.array([]),
                    'depths': np.array([])
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
        if self.simulation_results is None:
            raise ValueError("请先运行蒙特卡洛模拟")
        
        all_magnitudes = []
        all_depths = []
        event_counts = []
        
        for sim in self.simulation_results['simulations']:
            event_counts.append(len(sim['occurrence_times']))
            all_magnitudes.extend(sim['magnitudes'])
            all_depths.extend(sim['depths'])
        
        all_magnitudes = np.array(all_magnitudes)
        all_depths = np.array(all_depths)
        
        summary = {
            'total_simulations': len(self.simulation_results['simulations']),
            'mean_events_per_simulation': np.mean(event_counts),
            'std_events_per_simulation': np.std(event_counts),
            'min_events_per_simulation': np.min(event_counts),
            'max_events_per_simulation': np.max(event_counts),
            'magnitude_stats': {
                'mean': np.mean(all_magnitudes) if len(all_magnitudes) > 0 else 0,
                'std': np.std(all_magnitudes) if len(all_magnitudes) > 0 else 0,
                'min': np.min(all_magnitudes) if len(all_magnitudes) > 0 else 0,
                'max': np.max(all_magnitudes) if len(all_magnitudes) > 0 else 0,
            },
            'depth_stats': {
                'mean': np.mean(all_depths) if len(all_depths) > 0 else 0,
                'std': np.std(all_depths) if len(all_depths) > 0 else 0,
                'min': np.min(all_depths) if len(all_depths) > 0 else 0,
                'max': np.max(all_depths) if len(all_depths) > 0 else 0,
            }
        }
        
        return summary

def ab95_bc_point_source_model(frequencies, Mw, stress_drop=100, kappa=0.04, 
                             geometric_spreading=1.0, Q_frequency_exponent=0.8,
                             distance=None):
    """
    实现AB95_BC双拐角频率点源模型
    """
    # 模型常数
    rho = 2.8
    Vs = 3.5
    beta = Vs * 1000
    
    # 地震矩计算
    M0 = 10 ** (1.5 * Mw + 9.05)
    
    # 拐角频率计算
    f_c = 4.9e6 * beta * (stress_drop / M0) ** (1 / 3)
    f_max = 10
    
    # 双拐角频率震源谱模型
    epsilon = 0.2
    f_c2 = f_c + epsilon
    
    # 计算位移谱
    displacement_spectrum = np.ones_like(frequencies)
    
    for i, f in enumerate(frequencies):
        if f < f_c:
            displacement_spectrum[i] = M0 / (1 + (f / f_c) ** 2)
        elif f < f_c2:
            displacement_spectrum[i] = M0 / (1 + (f / f_c) ** 2) / (1 + (f / f_c2) ** 2)
        else:
            displacement_spectrum[i] = M0 / (1 + (f / f_c) ** 2) / (1 + (f / f_c2) ** 2) * (f_max / f) ** 2
    
    # 路径效应
    if distance is not None:
        geometric_spreading = 1.0 / distance
    
    if callable(geometric_spreading):
        path_effect = geometric_spreading(frequencies)
    else:
        path_effect = geometric_spreading
    
    # 衰减效应
    Q0 = 1000
    f_ref = 1.0
    Q = Q0 * (frequencies / f_ref) ** Q_frequency_exponent
    if distance is not None:
        attenuation = np.exp(-np.pi * frequencies * distance / (Q * Vs))
    else:
        attenuation = np.exp(-np.pi * frequencies * kappa)
    
    # 场地效应
    site_effect = np.exp(-np.pi * frequencies * kappa)
    
    # 合成加速度谱
    acceleration_spectrum = (2 * math.pi * frequencies) ** 2 * displacement_spectrum * path_effect * attenuation * site_effect
    
    return {
        'frequencies': frequencies,
        'displacement_spectrum': displacement_spectrum,
        'acceleration_spectrum': acceleration_spectrum,
        'f_corner1': f_c,
        'f_corner2': f_c2,
        'seismic_moment': M0
    }

class HybridSourceModel:
    """
    实现增加标准化因子的混合震源模型
    """
    
    def __init__(self, fault_length, fault_width, subfault_size, Mw, hypocenter, 
                 crustal_params=None, stochastic_params=None):
        self.fault_length = fault_length
        self.fault_width = fault_width
        self.subfault_size = subfault_size
        self.Mw = Mw
        self.hypocenter = hypocenter
        self.seismic_moment = 10 ** (1.5 * self.Mw + 9.05)
        
        self.crustal_params = crustal_params or {
            'density': 2.8,
            'shear_wave_velocity': 3.5,
            'stress_drop': 100,
            'kappa': 0.04,
            'q0': 1000,
            'q_exponent': 0.8
        }
        
        self.stochastic_params = stochastic_params or {
            'correlation_length': 2.0,
            'hurst_parameter': 0.75,
            'asperity_fraction': 0.2
        }
        
        self.n_subfaults_length = int(fault_length / subfault_size)
        self.n_subfaults_width = int(fault_width / subfault_size)
        self.total_subfaults = self.n_subfaults_length * self.n_subfaults_width
        self.subfault_moment = self.seismic_moment / self.total_subfaults
        
    def _normalization_factor(self, frequencies, slip_distribution):
        total_moment = np.sum(slip_distribution) * self.crustal_params['density'] * \
                      (self.crustal_params['shear_wave_velocity'] * 1000) ** 3
        theoretical_moment = self.seismic_moment
        base_factor = np.sqrt(theoretical_moment / total_moment)
        
        f_norm = 1.0
        frequency_correction = np.ones_like(frequencies)
        
        low_freq_idx = frequencies < 0.1
        high_freq_idx = frequencies > 10
        frequency_correction[low_freq_idx] = 1.1
        frequency_correction[high_freq_idx] = 0.9
        
        normalization = base_factor * frequency_correction
        return normalization
    
    def asperity_model(self):
        slip_distribution = np.zeros((self.n_subfaults_length, self.n_subfaults_width))
        rupture_time = np.zeros((self.n_subfaults_length, self.n_subfaults_width))
        
        hypocenter_x, hypocenter_y = self.hypocenter
        main_asperity_center = (
            int(hypocenter_x * self.n_subfaults_length),
            int(hypocenter_y * self.n_subfaults_width)
        )
        
        asperity_size_length = int(self.n_subfaults_length * 0.3)
        asperity_size_width = int(self.n_subfaults_width * 0.3)
        
        for i in range(self.n_subfaults_length):
            for j in range(self.n_subfaults_width):
                distance_to_asperity = np.sqrt(
                    ((i - main_asperity_center[0]) / asperity_size_length) ** 2 +
                    ((j - main_asperity_center[1]) / asperity_size_width) ** 2
                )
                
                if distance_to_asperity <= 1.0:
                    slip_distribution[i, j] = 2.0 * self.stochastic_params['asperity_fraction']
                else:
                    slip_distribution[i, j] = 1.0 - self.stochastic_params['asperity_fraction']
                
                rupture_time[i, j] = distance_to_asperity * self.subfault_size / \
                                   self.crustal_params['shear_wave_velocity']
        
        slip_distribution = slip_distribution * self.seismic_moment / np.sum(slip_distribution)
        return slip_distribution, rupture_time
    
    def stochastic_source_component(self, frequencies, slip_distribution):
        np.random.seed(42)
        random_phase = np.random.uniform(0, 2 * np.pi, size=slip_distribution.shape)
        random_amplitude = np.random.normal(1.0, 0.3, size=slip_distribution.shape)
        
        stochastic_spectrum = np.zeros(len(frequencies), dtype=complex)
        
        for idx, f in enumerate(frequencies):
            for i in range(self.n_subfaults_length):
                for j in range(self.n_subfaults_width):
                    subfault_contrib = (slip_distribution[i, j] * 
                                      random_amplitude[i, j] * 
                                      np.exp(1j * random_phase[i, j]) *
                                      np.exp(-2j * np.pi * f * 
                                           (i + j) * self.subfault_size / 
                                           self.crustal_params['shear_wave_velocity']))
                    
                    stochastic_spectrum[idx] += subfault_contrib
        
        return stochastic_spectrum
    
    def compute_hybrid_source_spectrum(self, frequencies, station_location):
        slip_distribution, rupture_time = self.asperity_model()
        stochastic_component = self.stochastic_source_component(frequencies, slip_distribution)
        normalization = self._normalization_factor(frequencies, slip_distribution)
        path_effect = self._compute_path_effect(frequencies, station_location)
        site_effect = self._compute_site_effect(frequencies)
        
        hybrid_spectrum = (stochastic_component * normalization * 
                         path_effect * site_effect)
        
        return {
            'frequencies': frequencies,
            'hybrid_spectrum': hybrid_spectrum,
            'slip_distribution': slip_distribution,
            'rupture_time': rupture_time,
            'normalization_factor': normalization,
            'stochastic_component': stochastic_component
        }
    
    def _compute_path_effect(self, frequencies, station_location):
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(station_location, self.hypocenter)))
        geometric_spreading = 1.0 / distance
        q_factor = self.crustal_params['q0'] * (frequencies / 1.0) ** self.crustal_params['q_exponent']
        attenuation = np.exp(-np.pi * frequencies * distance / (q_factor * self.crustal_params['shear_wave_velocity']))
        return geometric_spreading * attenuation
    
    def _compute_site_effect(self, frequencies):
        kappa = self.crustal_params['kappa']
        return np.exp(-np.pi * kappa * frequencies)


# =============================================================================
# 主程序示例
# =============================================================================

if __name__ == "__main__":
    print("开始完整地震模拟系统...")
    
    # 1. 定义研究区域（以孟菲斯地区为例）
    region_bounds = {
        'lon_min': -90.5,
        'lon_max': -89.5,
        'lat_min': 35.0,
        'lat_max': 35.5,
        'depth_min': 5,
        'depth_max': 25
    }
    
    # 2. 创建完整模拟系统
    complete_simulator = CompleteSeismicSimulation(region_bounds, time_period=100.0)
    
    # 3. 设置地震统计参数
    seismic_params = {
        'annual_rate': 0.05,
        'b_value': 1.0,
        'M_min': 4.0,
        'M_max': 8.0,
        'depth_distribution': {
            'type': 'lognormal',
            'mean': 12,
            'std': 6
        },
        'spatial_distribution': {
            'type': 'uniform'
        }
    }
    
    complete_simulator.set_seismic_parameters(
        annual_rate=seismic_params['annual_rate'],
        b_value=seismic_params['b_value'],
        M_min=seismic_params['M_min'],
        M_max=seismic_params['M_max'],
        depth_params=seismic_params['depth_distribution'],
        spatial_params=seismic_params['spatial_distribution']
    )
    
    # 4. 设置台站位置
    station_locations = [
        (-90.0, 35.2, 0),  # 主台站
        
    ]
    
    # 5. 运行完整模拟
    complete_results = complete_simulator.run_complete_simulation(
        n_simulations=1,
        station_locations=station_locations,
        random_seed=42
    )
    
    # 6. 获取详细统计
    summary = complete_simulator.get_detailed_summary()
    print("\n=== 模拟统计摘要 ===")
    print(f"总模拟次数: {summary['monte_carlo_summary']['total_simulations']}")
    print(f"平均每次模拟事件数: {summary['monte_carlo_summary']['mean_events_per_simulation']:.2f}")
    print(f"模型使用情况 - 点源模型: {summary['model_usage']['point_source']}, 混合源模型: {summary['model_usage']['hybrid_source']}")
    print(f"平均最大加速度: {summary['ground_motion_statistics']['mean_max_acceleration']:.6f}")
    print(f"最大加速度范围: {summary['ground_motion_statistics']['min_max_acceleration']:.6f} - {summary['ground_motion_statistics']['max_max_acceleration']:.6f}")
    
    # 7. 绘制综合结果
    print("\n生成结果图表...")
    complete_simulator.plot_comprehensive_results(
        simulation_idx=0, 
        station_idx=0,
        save_path='complete_seismic_simulation.png'
    )
    
    # 8. 导出结果
    results_df = complete_simulator.export_complete_results(simulation_idx=0)
    print(f"\n导出的结果数据行数: {len(results_df)}")
    if len(results_df) > 0:
        print("前5行结果:")
        print(results_df.head())
        
        # 保存到CSV文件
        results_df.to_csv('seismic_simulation_results.csv', index=False)
        print("结果已保存到 seismic_simulation_results.csv")
    
    print("\n完整地震模拟完成!")