"""
Simplified demonstration of earthquake prediction code (without NumPy dependency)
This version uses built-in Python only to show the code logic and structure.
"""

import random
from collections import defaultdict

print("="*80)
print("主震预测 - 简化演示版 (Earthquake Prediction - Simplified Demo)")
print("="*80)

# Step 1: Parameters
print("\n[Step 1] 参数设置")
annual_occurrence_rate = 2.0  # avg earthquakes/year
study_years = 50
num_simulations = 10  # Reduced for demo
periods = [0.01, 0.2, 0.5, 1.0, 2.0, 3.0]  # Response periods in seconds

print(f"  Annual occurrence rate: {annual_occurrence_rate} earthquakes/year")
print(f"  Study period: {study_years} years")
print(f"  Number of simulations: {num_simulations}")
print(f"  Response periods (s): {periods}")

# Step 2: Simulate earthquakes
print("\n[Step 2] 蒙特卡洛地震模拟")

def simulate_earthquake_event():
    """Simulate a single earthquake event"""
    return {
        'lon': random.uniform(110, 112),
        'lat': random.uniform(34, 36),
        'depth': random.uniform(5, 20),
        'mag': random.uniform(5.0, 7.0),
        'mechanism': random.choice(['strike_slip', 'normal', 'reverse'])
    }

# Step 3: Collect response spectra
print("  开始模拟...")
all_sa = []  # Store SA values for all simulations

for sim in range(num_simulations):
    if (sim + 1) % max(1, num_simulations // 5) == 0:
        print(f"    Progress: {sim + 1}/{num_simulations} ({100 * (sim + 1) / num_simulations:.1f}%)")
    
    # Generate random number of earthquakes in this year
    num_events = random.randint(0, 4)
    
    sa_max = [0.0] * len(periods)  # Max SA across all events in this simulation
    
    for event_idx in range(num_events):
        event = simulate_earthquake_event()
        
        # Simple GMPE spectrum (Ground Motion Prediction Equation)
        # SA(T) = scaling_mag * scaling_depth * scaling_distance * attenuation(T)
        mag_scale = max(0.1, (event['mag'] - 5.0) / 2.0)
        depth_scale = max(0.1, 1.0 - event['depth'] / 100.0)
        distance = ((event['lon'] - 111.0)**2 + (event['lat'] - 35.0)**2)**0.5
        distance_scale = 1.0 / (1.0 + distance)
        
        # Compute spectrum for each period
        for i, T in enumerate(periods):
            period_scale = 1.0 / (1.0 + T)
            sa = mag_scale * depth_scale * distance_scale * period_scale
            sa_max[i] = max(sa_max[i], sa)
    
    all_sa.append(sa_max)

print(f"  模拟完成，共处理 {num_simulations} 次实现")

# Step 4: Calculate UHS (Uniform Hazard Spectrum)
print("\n[Step 3] 计算 UHS (Uniform Hazard Spectrum)")
pe_target = 0.10  # 10% exceedance probability
sa_uhs = []

for i, T in enumerate(periods):
    sa_values = sorted([sim_sa[i] for sim_sa in all_sa], reverse=True)
    index = min(int(pe_target * num_simulations), len(sa_values) - 1)
    sa_uhs.append(sa_values[index])

print(f"  超越概率: {pe_target*100:.0f}%")
print(f"  UHS 计算完成")

# Step 5: Calculate CMS (Conditional Mean Spectrum)
print("\n[Step 4] 计算 CMS (Conditional Mean Spectrum)")
target_period_idx = 2  # T = 0.5s
target_sa = sa_uhs[target_period_idx]
tolerance = 0.05 * target_sa if target_sa > 0 else 0.01

# Filter simulations with similar SA at target period
matching_sims = [
    sim_sa for sim_sa in all_sa 
    if abs(sim_sa[target_period_idx] - target_sa) <= tolerance
]

if matching_sims:
    cms = [sum(sim_sa[i] for sim_sa in matching_sims) / len(matching_sims) 
           for i in range(len(periods))]
else:
    # Fallback: use all simulations if no match
    cms = [sum(sim_sa[i] for sim_sa in all_sa) / len(all_sa) 
           for i in range(len(periods))]

print(f"  控制周期: T = {periods[target_period_idx]:.2f}s")
print(f"  容差: {tolerance:.6f}")
print(f"  匹配模拟数: {len(matching_sims)}/{len(all_sa)}")

# Step 6: Output results
print("\n" + "="*80)
print("输出结果 (Results)")
print("="*80)

print(f"\n均匀危险谱 (UHS) @ 10% 超越概率 in {study_years} years:")
print("-" * 80)
for i, T in enumerate(periods):
    print(f"  T = {T:6.2f} s: SA = {sa_uhs[i]:10.6f} g")

print(f"\n条件平均谱 (CMS) @ T = {periods[target_period_idx]:.2f}s:")
print("-" * 80)
for i, T in enumerate(periods):
    print(f"  T = {T:6.2f} s: SA = {cms[i]:10.6f} g")

print("\n" + "="*80)
print("演示完成！(Demo completed!)")
print("="*80)
print("\n说明:")
print("  - 此版本为演示版本，使用随机生成的地震数据")
print("  - 实际应用需使用真实的地震目录和 GMPE 公式")
print("  - 完整版本需要 NumPy/Pandas 支持 (see 主震预测.py)")
print("="*80)
