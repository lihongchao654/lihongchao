#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的加速度谱可视化工具
从 CSV 数据文件读取加速度谱数据，生成美观的可视化图形

支持两种模式：
1. matplotlib 模式 - 如果 matplotlib 可用
2. SVG 纯 Python 模式 - 独立生成 SVG 图形
"""

import os
import re
import sys
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 尝试导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 使用非交互式后端
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
    print("✓ matplotlib 已加载")
except ImportError as e:
    HAS_MATPLOTLIB = False
    print(f"⚠ matplotlib 不可用, 将使用 SVG 纯 Python 模式")


class SVGSpectrumPlotter:
    """纯 Python SVG 加速度谱绘图器"""
    
    def __init__(self, width=1000, height=700, margin=60):
        """初始化 SVG 绘图器"""
        self.width = width
        self.height = height
        self.margin = margin
        self.plot_width = width - 2 * margin
        self.plot_height = height - 2 * margin
    
    def log10_scale(self, value: float) -> float:
        """转换为对数尺度"""
        if value <= 0:
            return 0
        return math.log10(value)
    
    def data_to_svg(self, freq: float, accel: float, 
                    freq_min: float, freq_max: float,
                    accel_min: float, accel_max: float) -> Tuple[float, float]:
        """将数据坐标转换为 SVG 坐标"""
        # 对数尺度转换
        freq_log = self.log10_scale(freq)
        accel_log = self.log10_scale(accel)
        freq_min_log = self.log10_scale(freq_min)
        freq_max_log = self.log10_scale(freq_max)
        accel_min_log = self.log10_scale(accel_min)
        accel_max_log = self.log10_scale(accel_max)
        
        # 归一化到 [0, 1]
        x_norm = (freq_log - freq_min_log) / (freq_max_log - freq_min_log)
        y_norm = (accel_log - accel_min_log) / (accel_max_log - accel_min_log)
        
        # 转换为 SVG 坐标
        x = self.margin + x_norm * self.plot_width
        y = self.height - self.margin - y_norm * self.plot_height
        
        return x, y
    
    def create_single_spectrum_svg(self, event_id: int, event_data: Dict,
                                   output_file: str) -> bool:
        """生成单个事件的 SVG 加速度谱"""
        spectrum = event_data['spectrum']
        info = event_data['info']
        
        frequencies = [s[0] for s in spectrum]
        accelerations = [s[1] for s in spectrum]
        
        freq_min = min(frequencies)
        freq_max = max(frequencies)
        accel_min = min(accelerations)
        accel_max = max(accelerations)
        
        # 开始 SVG
        svg = ['<?xml version="1.0" encoding="UTF-8"?>']
        svg.append(f'<svg width="{self.width}" height="{self.height}" '
                   f'xmlns="http://www.w3.org/2000/svg">')
        svg.append('<style>')
        svg.append('text { font-family: Arial, sans-serif; }')
        svg.append('.title { font-size: 16px; font-weight: bold; }')
        svg.append('.label { font-size: 12px; }')
        svg.append('.tick { font-size: 10px; }')
        svg.append('.grid { stroke: #e0e0e0; stroke-width: 0.5; }')
        svg.append('.curve { stroke: #0066cc; stroke-width: 2; fill: none; }')
        svg.append('.point { stroke: #003366; stroke-width: 1; fill: #0066cc; }')
        svg.append('</style>')
        
        # 背景
        svg.append(f'<rect width="{self.width}" height="{self.height}" fill="white"/>')
        
        # 绘制网格和坐标轴
        svg.append(f'<line x1="{self.margin}" y1="{self.height-self.margin}" '
                   f'x2="{self.width-self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        svg.append(f'<line x1="{self.margin}" y1="{self.margin}" '
                   f'x2="{self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        
        # 频率轴刻度（对数）
        for i in range(int(math.ceil(math.log10(freq_min))), 
                      int(math.floor(math.log10(freq_max))) + 1):
            freq_tick = 10 ** i
            if freq_min <= freq_tick <= freq_max:
                x, _ = self.data_to_svg(freq_tick, accel_min, freq_min, freq_max, 
                                       accel_min, accel_max)
                svg.append(f'<line x1="{x}" y1="{self.height-self.margin}" '
                          f'x2="{x}" y2="{self.height-self.margin+5}" stroke="black"/>')
                svg.append(f'<text x="{x}" y="{self.height-self.margin+20}" '
                          f'text-anchor="middle" class="tick">{freq_tick:.1f}</text>')
        
        # 加速度轴刻度（对数）
        for i in range(int(math.ceil(math.log10(accel_min))), 
                      int(math.floor(math.log10(accel_max))) + 1):
            accel_tick = 10 ** i
            if accel_min <= accel_tick <= accel_max:
                _, y = self.data_to_svg(freq_min, accel_tick, freq_min, freq_max,
                                       accel_min, accel_max)
                svg.append(f'<line x1="{self.margin-5}" y1="{y}" '
                          f'x2="{self.margin}" y2="{y}" stroke="black"/>')
                svg.append(f'<text x="{self.margin-10}" y="{y+4}" '
                          f'text-anchor="end" class="tick">10^{i}</text>')
        
        # 绘制曲线
        path = []
        for freq, accel in zip(frequencies, accelerations):
            x, y = self.data_to_svg(freq, accel, freq_min, freq_max, 
                                   accel_min, accel_max)
            if not path:
                path.append(f"M{x},{y}")
            else:
                path.append(f"L{x},{y}")
        
        svg.append(f'<path d="{" ".join(path)}" class="curve"/>')
        
        # 绘制数据点
        for freq, accel in zip(frequencies, accelerations):
            x, y = self.data_to_svg(freq, accel, freq_min, freq_max,
                                   accel_min, accel_max)
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" class="point"/>')
        
        # 标签
        svg.append(f'<text x="{self.width//2}" y="25" text-anchor="middle" class="title">'
                   f'加速度谱 - 事件 {event_id}</text>')
        svg.append(f'<text x="{self.width//2}" y="45" text-anchor="middle" class="label">'
                   f'M{info["magnitude"]:.2f} | 深度 {info["depth_km"]:.1f} km | '
                   f'距离 {info["distance_km"]:.1f} km</text>')
        
        # X 轴标签
        svg.append(f'<text x="{self.width//2}" y="{self.height-10}" '
                   f'text-anchor="middle" class="label">频率 (Hz)</text>')
        
        # Y 轴标签
        svg.append(f'<text x="20" y="{self.height//2}" '
                   f'text-anchor="middle" class="label" transform="rotate(-90 20 {self.height//2})">'
                   f'加速度谱 (cm/s2)</text>')
        
        # 信息框
        info_y = 70
        info_box = [
            f"震级: M{info['magnitude']:.2f}",
            f"发生时间: {info['time_year']:.1f} 年",
            f"深度: {info['depth_km']:.1f} km",
            f"距离: {info['distance_km']:.1f} km",
        ]
        
        svg.append(f'<rect x="{self.width-220}" y="{info_y-5}" width="210" height="100" '
                   f'fill="wheat" opacity="0.8" rx="5"/>')
        for line_idx, line in enumerate(info_box):
            svg.append(f'<text x="{self.width-210}" y="{info_y+line_idx*22}" '
                       f'class="label">{line}</text>')
        
        svg.append('</svg>')
        
        # 写入文件
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(svg))
            return True
        except Exception as e:
            print(f"❌ 写入失败: {e}")
            return False


class AccelerationSpectrumVisualizer:
    """加速度谱可视化类"""
    
    def __init__(self, csv_file: str = "地震动加速度谱.csv", output_dir: str = "."):
        """
        初始化可视化器
        
        Args:
            csv_file: 输入 CSV 文件路径
            output_dir: 输出图形目录
        """
        self.csv_file = csv_file
        self.output_dir = output_dir
        self.events = {}
        
        # 创建输出目录
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def parse_csv(self) -> bool:
        """解析 CSV 文件，提取事件和加速度谱数据"""
        if not os.path.exists(self.csv_file):
            print(f"❌ 错误：找不到文件 {self.csv_file}")
            return False
        
        try:
            with open(self.csv_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 分割事件块
            event_blocks = re.split(r'事件\s+(\d+)\n', content)[1:]
            
            for i in range(0, len(event_blocks), 2):
                event_id = int(event_blocks[i])
                event_data = event_blocks[i + 1].strip()
                
                # 解析事件信息行
                lines = event_data.split('\n')
                info_line = None
                spectrum_start = -1
                
                for idx, line in enumerate(lines):
                    if '震级' in line and '时间' in line:
                        info_line = idx + 1
                    if '频率' in line and '加速度谱' in line:
                        spectrum_start = idx + 1
                        break
                
                if info_line is None or spectrum_start == -1:
                    print(f"⚠ 警告：无法解析事件 {event_id}")
                    continue
                
                # 解析事件参数
                params = lines[info_line].split(',')
                event_info = {
                    'magnitude': float(params[0].strip()),
                    'time_year': float(params[1].strip()),
                    'longitude': float(params[2].strip()),
                    'latitude': float(params[3].strip()),
                    'depth_km': float(params[4].strip()),
                    'distance_km': float(params[5].strip()),
                }
                
                # 解析加速度谱
                spectrum = []
                for idx in range(spectrum_start, len(lines)):
                    line = lines[idx].strip()
                    if not line or '事件' in line:
                        break
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            freq = float(parts[0].strip())
                            accel = float(parts[1].strip())
                            spectrum.append((freq, accel))
                        except ValueError:
                            continue
                
                self.events[event_id] = {
                    'info': event_info,
                    'spectrum': spectrum
                }
                print(f"✓ 加载事件 {event_id}: M{event_info['magnitude']:.2f}, "
                      f"{len(spectrum)} 个频率点")
            
            print(f"\n✓ 成功解析 {len(self.events)} 个事件\n")
            return True
            
        except Exception as e:
            print(f"❌ 解析错误: {e}")
            return False
    
    def plot_single_spectrum_matplotlib(self, event_id: int) -> bool:
        """使用 matplotlib 绘制单个事件的加速度谱"""
        if event_id not in self.events:
            print(f"❌ 事件 {event_id} 不存在")
            return False
        
        event_data = self.events[event_id]
        info = event_data['info']
        spectrum = event_data['spectrum']
        
        frequencies = [s[0] for s in spectrum]
        accelerations = [s[1] for s in spectrum]
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        
        # 绘制对数-对数曲线
        ax.loglog(frequencies, accelerations, 'b-', linewidth=2.5, label='加速度谱')
        ax.scatter(frequencies, accelerations, s=20, c='darkblue', alpha=0.6, zorder=5)
        
        # 设置网格
        ax.grid(True, which='both', linestyle='--', alpha=0.6, linewidth=0.7)
        ax.set_axisbelow(True)
        
        # 标签和标题
        ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('加速度谱 (cm/s²)', fontsize=12, fontweight='bold')
        title = f"加速度谱 - 事件 {event_id}\n" \
                f"M{info['magnitude']:.2f} | 深度 {info['depth_km']:.1f} km | " \
                f"距离 {info['distance_km']:.1f} km"
        ax.set_title(title, fontsize=13, fontweight='bold', pad=15)
        
        # 添加图例框
        info_text = f"震级: M{info['magnitude']:.2f}\n" \
                   f"发生时间: {info['time_year']:.1f} 年\n" \
                   f"深度: {info['depth_km']:.1f} km\n" \
                   f"距离: {info['distance_km']:.1f} km\n" \
                   f"位置: ({info['longitude']:.2f}°, {info['latitude']:.2f}°)"
        ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        ax.legend(loc='lower right', fontsize=11)
        
        # 保存图形
        output_file = os.path.join(self.output_dir, f"加速度谱_事件{event_id}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_single_spectrum_svg(self, event_id: int) -> bool:
        """使用纯 Python SVG 绘制单个事件的加速度谱"""
        if event_id not in self.events:
            print(f"❌ 事件 {event_id} 不存在")
            return False
        
        event_data = self.events[event_id]
        plotter = SVGSpectrumPlotter()
        output_file = os.path.join(self.output_dir, f"加速度谱_事件{event_id}.svg")
        
        if plotter.create_single_spectrum_svg(event_id, event_data, output_file):
            print(f"✓ 保存: {output_file}")
            return True
        return False
    
    def plot_single_spectrum(self, event_id: int) -> bool:
        """选择合适的方法绘制单个事件"""
        if HAS_MATPLOTLIB:
            return self.plot_single_spectrum_matplotlib(event_id)
        else:
            return self.plot_single_spectrum_svg(event_id)
    
    def visualize_all(self) -> bool:
        """生成所有可视化图形"""
        print("="*70)
        print("开始生成加速度谱可视化图形...")
        print("="*70 + "\n")
        
        if HAS_MATPLOTLIB:
            print("✓ 使用 matplotlib 高精度绘图模式\n")
        else:
            print("✓ 使用 SVG 纯 Python 绘图模式（不依赖 NumPy）\n")
        
        # 单个事件图
        print("[1] 生成单个事件加速度谱...")
        for event_id in sorted(self.events.keys()):
            self.plot_single_spectrum(event_id)
        
        print("\n" + "="*70)
        print("✓ 加速度谱可视化生成完成！")
        print("="*70)
        return True


def main():
    """主函数"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "地震动加速度谱.csv")
    
    # 创建可视化器
    visualizer = AccelerationSpectrumVisualizer(
        csv_file=csv_file,
        output_dir=script_dir
    )
    
    # 解析 CSV
    print("\n正在读取加速度谱数据...\n")
    if not visualizer.parse_csv():
        print("\n❌ 数据读取失败")
        return 1
    
    # 生成所有可视化
    if not visualizer.visualize_all():
        print("\n❌ 可视化生成失败")
        return 1
    
    print("\n生成的图形文件：")
    if HAS_MATPLOTLIB:
        print("  • 加速度谱_事件N.png - 单个事件的加速度谱（PNG 格式，高精度）")
    else:
        print("  • 加速度谱_事件N.svg - 单个事件的加速度谱（SVG 矢量格式）")
    print("\n提示：SVG 文件可在任何浏览器或图形编辑软件中打开")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
