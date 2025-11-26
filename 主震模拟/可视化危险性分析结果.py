#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地震危险性分析结果可视化工具
将一致危险谱、条件均值谱和危险性曲线进行可视化
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
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
    print("✓ matplotlib 已加载\n")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib 不可用, 将使用 SVG 纯 Python 模式\n")


class SVGHazardPlotter:
    """纯 Python SVG 危险性分析绘图器"""
    
    def __init__(self, width=1000, height=700, margin=60):
        self.width = width
        self.height = height
        self.margin = margin
        self.plot_width = width - 2 * margin
        self.plot_height = height - 2 * margin
    
    def log10_scale(self, value: float) -> float:
        if value <= 0:
            return 0
        return math.log10(value)
    
    def data_to_svg(self, x: float, y: float, 
                    x_min: float, x_max: float,
                    y_min: float, y_max: float,
                    log_x=True, log_y=True) -> Tuple[float, float]:
        """将数据坐标转换为 SVG 坐标"""
        if log_x:
            x_log = self.log10_scale(x)
            x_min_log = self.log10_scale(x_min)
            x_max_log = self.log10_scale(x_max)
            x_norm = (x_log - x_min_log) / (x_max_log - x_min_log)
        else:
            x_norm = (x - x_min) / (x_max - x_min)
        
        if log_y:
            y_log = self.log10_scale(y)
            y_min_log = self.log10_scale(y_min)
            y_max_log = self.log10_scale(y_max)
            y_norm = (y_log - y_min_log) / (y_max_log - y_min_log)
        else:
            y_norm = (y - y_min) / (y_max - y_min)
        
        x_svg = self.margin + x_norm * self.plot_width
        y_svg = self.height - self.margin - y_norm * self.plot_height
        
        return x_svg, y_svg
    
    def create_uhs_svg(self, periods: List[float], sa: List[float], 
                       output_file: str) -> bool:
        """生成一致危险谱 SVG"""
        if not periods or not sa:
            return False
        
        period_min = min(periods)
        period_max = max(periods)
        sa_min = min(sa) * 0.1
        sa_max = max(sa) * 10
        
        svg = ['<?xml version="1.0" encoding="UTF-8"?>']
        svg.append(f'<svg width="{self.width}" height="{self.height}" '
                   f'xmlns="http://www.w3.org/2000/svg">')
        svg.append('<style>')
        svg.append('text { font-family: Arial, sans-serif; }')
        svg.append('.title { font-size: 16px; font-weight: bold; }')
        svg.append('.label { font-size: 12px; }')
        svg.append('.tick { font-size: 10px; }')
        svg.append('.curve { stroke: #d62728; stroke-width: 2.5; fill: none; }')
        svg.append('.point { stroke: #8b0000; stroke-width: 1; fill: #d62728; }')
        svg.append('</style>')
        
        svg.append(f'<rect width="{self.width}" height="{self.height}" fill="white"/>')
        
        # 坐标轴
        svg.append(f'<line x1="{self.margin}" y1="{self.height-self.margin}" '
                   f'x2="{self.width-self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        svg.append(f'<line x1="{self.margin}" y1="{self.margin}" '
                   f'x2="{self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        
        # 周期轴刻度
        for period in [0.01, 0.1, 1.0, 10.0]:
            if period_min <= period <= period_max:
                x, _ = self.data_to_svg(period, sa_min, period_min, period_max,
                                       sa_min, sa_max, log_x=True, log_y=True)
                svg.append(f'<line x1="{x}" y1="{self.height-self.margin}" '
                          f'x2="{x}" y2="{self.height-self.margin+5}" stroke="black"/>')
                svg.append(f'<text x="{x}" y="{self.height-self.margin+20}" '
                          f'text-anchor="middle" class="tick">{period:.2f}</text>')
        
        # 谱加速度轴刻度
        for exp in range(int(math.log10(sa_min)), int(math.log10(sa_max)) + 1):
            sa_tick = 10 ** exp
            _, y = self.data_to_svg(period_min, sa_tick, period_min, period_max,
                                   sa_min, sa_max, log_x=True, log_y=True)
            svg.append(f'<line x1="{self.margin-5}" y1="{y}" '
                      f'x2="{self.margin}" y2="{y}" stroke="black"/>')
            svg.append(f'<text x="{self.margin-10}" y="{y+4}" '
                      f'text-anchor="end" class="tick">10^{exp}</text>')
        
        # 绘制曲线
        path = []
        for period, acc in zip(periods, sa):
            x, y = self.data_to_svg(period, acc, period_min, period_max,
                                   sa_min, sa_max, log_x=True, log_y=True)
            if not path:
                path.append(f"M{x},{y}")
            else:
                path.append(f"L{x},{y}")
        
        svg.append(f'<path d="{" ".join(path)}" class="curve"/>')
        
        # 数据点
        for period, acc in zip(periods, sa):
            x, y = self.data_to_svg(period, acc, period_min, period_max,
                                   sa_min, sa_max, log_x=True, log_y=True)
            svg.append(f'<circle cx="{x}" cy="{y}" r="3" class="point"/>')
        
        # 标题和标签
        svg.append(f'<text x="{self.width//2}" y="25" text-anchor="middle" class="title">'
                   f'一致危险谱 (Uniform Hazard Spectrum)</text>')
        svg.append(f'<text x="{self.width//2}" y="45" text-anchor="middle" class="label">'
                   f'50年超越概率 20%</text>')
        
        svg.append(f'<text x="{self.width//2}" y="{self.height-10}" '
                   f'text-anchor="middle" class="label">周期 (s)</text>')
        
        svg.append(f'<text x="20" y="{self.height//2}" '
                   f'text-anchor="middle" class="label" transform="rotate(-90 20 {self.height//2})">'
                   f'谱加速度 (cm/s²)</text>')
        
        svg.append('</svg>')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(svg))
            return True
        except Exception as e:
            print(f"❌ 写入失败: {e}")
            return False


class HazardAnalysisVisualizer:
    """危险性分析结果可视化类"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def parse_csv(self, filename: str) -> Tuple[List[float], List[float]]:
        """解析 CSV 文件"""
        periods = []
        values = []
        
        if not os.path.exists(filename):
            return periods, values
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            period = float(parts[0].strip())
                            value = float(parts[1].strip())
                            periods.append(period)
                            values.append(value)
                        except ValueError:
                            continue
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
        
        return periods, values
    
    def plot_uhs_matplotlib(self) -> bool:
        """使用 matplotlib 绘制一致危险谱"""
        periods, sa = self.parse_csv('一致危险谱_50年20%.csv')
        if not periods:
            return False
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        
        ax.loglog(periods, sa, 'r-', linewidth=2.5, label='UHS (50yr, 20%)', marker='o', markersize=6)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('一致危险谱 (Uniform Hazard Spectrum)\n50年超越概率 20%', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=11)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        output_file = os.path.join(self.output_dir, '一致危险谱.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_cms_matplotlib(self) -> bool:
        """使用 matplotlib 绘制条件均值谱"""
        periods, sa = self.parse_csv('条件均值谱.csv')
        if not periods:
            return False
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        
        ax.loglog(periods, sa, 'g-', linewidth=2.5, label='CMS (T=1.0s)', marker='s', markersize=6)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('条件均值谱 (Conditional Mean Spectrum)\n控制周期 T=1.0s', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=11)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        output_file = os.path.join(self.output_dir, '条件均值谱.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_combined_matplotlib(self) -> bool:
        """绘制 UHS 和 CMS 对比"""
        uhs_periods, uhs_sa = self.parse_csv('一致危险谱_50年20%.csv')
        cms_periods, cms_sa = self.parse_csv('条件均值谱.csv')
        
        if not uhs_periods or not cms_periods:
            return False
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        
        ax.loglog(uhs_periods, uhs_sa, 'r-', linewidth=2.5, label='UHS (50yr, 20%)', marker='o', markersize=5)
        ax.loglog(cms_periods, cms_sa, 'g--', linewidth=2.5, label='CMS (T=1.0s)', marker='s', markersize=5)
        
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_axisbelow(True)
        
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('地震危险性分析结果对比\n一致危险谱 vs 条件均值谱', 
                    fontsize=13, fontweight='bold', pad=15)
        
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.set_facecolor('#f8f9fa')
        fig.patch.set_facecolor('white')
        
        output_file = os.path.join(self.output_dir, 'UHS_vs_CMS.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_uhs_svg(self) -> bool:
        """使用纯 Python SVG 绘制一致危险谱"""
        periods, sa = self.parse_csv('一致危险谱_50年20%.csv')
        if not periods:
            return False
        
        plotter = SVGHazardPlotter()
        output_file = os.path.join(self.output_dir, '一致危险谱.svg')
        
        if plotter.create_uhs_svg(periods, sa, output_file):
            print(f"✓ 保存: {output_file}")
            return True
        return False
    
    def visualize_all(self) -> bool:
        """生成所有可视化"""
        print("="*70)
        print("开始生成地震危险性分析结果可视化...")
        print("="*70 + "\n")
        
        if HAS_MATPLOTLIB:
            print("✓ 使用 matplotlib 高精度绘图模式\n")
            print("[1] 绘制一致危险谱...")
            self.plot_uhs_matplotlib()
            
            print("\n[2] 绘制条件均值谱...")
            self.plot_cms_matplotlib()
            
            print("\n[3] 绘制 UHS vs CMS 对比...")
            self.plot_combined_matplotlib()
        else:
            print("✓ 使用 SVG 纯 Python 绘图模式\n")
            print("[1] 绘制一致危险谱...")
            self.plot_uhs_svg()
        
        print("\n" + "="*70)
        print("✓ 地震危险性分析结果可视化完成！")
        print("="*70)
        return True


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    visualizer = HazardAnalysisVisualizer(output_dir=script_dir)
    
    print("\n正在读取危险性分析数据...\n")
    
    if not visualizer.visualize_all():
        print("\n❌ 可视化生成失败")
        return 1
    
    print("\n生成的图形文件：")
    if HAS_MATPLOTLIB:
        print("  • 一致危险谱.png - UHS 危险性分析（PNG 格式）")
        print("  • 条件均值谱.png - CMS 危险性分析（PNG 格式）")
        print("  • UHS_vs_CMS.png - 对比分析（PNG 格式）")
    else:
        print("  • 一致危险谱.svg - UHS 危险性分析（SVG 矢量格式）")
    
    print("\n说明：")
    print("  • UHS (Uniform Hazard Spectrum): 一致危险谱")
    print("  • CMS (Conditional Mean Spectrum): 条件均值谱")
    print("  • 这些图形可用于地震工程设计和危险性评估")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
