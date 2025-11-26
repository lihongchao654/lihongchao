#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合地震分析可视化工具
集成加速度谱、一致危险谱和条件均值谱的完整可视化
支持 matplotlib (PNG) 和纯 Python SVG 两种输出格式
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
    print("✓ matplotlib 已加载")
except ImportError:
    HAS_MATPLOTLIB = False
    print("⚠ matplotlib 不可用，将使用 SVG 纯 Python 模式")


class SVGPlotter:
    """纯 Python SVG 绘图器"""
    
    def __init__(self, width=1000, height=700, margin=60):
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
    
    def create_spectrum_svg(self, x_data: List[float], y_data: List[float],
                           x_label: str, y_label: str, title: str,
                           output_file: str, color: str = "#0066cc") -> bool:
        """创建通用谱图的 SVG"""
        if not x_data or not y_data:
            return False
        
        x_min = min(x_data)
        x_max = max(x_data)
        y_min = min(y_data) * 0.1
        y_max = max(y_data) * 10
        
        svg = ['<?xml version="1.0" encoding="UTF-8"?>']
        svg.append(f'<svg width="{self.width}" height="{self.height}" '
                   f'xmlns="http://www.w3.org/2000/svg">')
        svg.append('<style>')
        svg.append('text { font-family: Arial, sans-serif; }')
        svg.append('.title { font-size: 16px; font-weight: bold; }')
        svg.append('.label { font-size: 12px; }')
        svg.append('.tick { font-size: 10px; }')
        svg.append(f'.curve {{ stroke: {color}; stroke-width: 2.5; fill: none; }}')
        svg.append(f'.point {{ stroke: {color}; stroke-width: 1; fill: {color}; opacity: 0.7; }}')
        svg.append('</style>')
        
        svg.append(f'<rect width="{self.width}" height="{self.height}" fill="white"/>')
        
        # 坐标轴
        svg.append(f'<line x1="{self.margin}" y1="{self.height-self.margin}" '
                   f'x2="{self.width-self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        svg.append(f'<line x1="{self.margin}" y1="{self.margin}" '
                   f'x2="{self.margin}" y2="{self.height-self.margin}" '
                   f'stroke="black" stroke-width="2"/>')
        
        # X 轴刻度
        x_ticks = self._get_log_ticks(x_min, x_max)
        for tick in x_ticks:
            x_svg, _ = self.data_to_svg(tick, y_min, x_min, x_max, 
                                       y_min, y_max, log_x=True, log_y=True)
            svg.append(f'<line x1="{x_svg}" y1="{self.height-self.margin}" '
                      f'x2="{x_svg}" y2="{self.height-self.margin+5}" stroke="black"/>')
            svg.append(f'<text x="{x_svg}" y="{self.height-self.margin+20}" '
                      f'text-anchor="middle" class="tick">{tick:.2f}</text>')
        
        # Y 轴刻度
        y_ticks = self._get_log_ticks(y_min, y_max)
        for tick in y_ticks:
            _, y_svg = self.data_to_svg(x_min, tick, x_min, x_max,
                                       y_min, y_max, log_x=True, log_y=True)
            svg.append(f'<line x1="{self.margin-5}" y1="{y_svg}" '
                      f'x2="{self.margin}" y2="{y_svg}" stroke="black"/>')
            if tick < 1e6:
                svg.append(f'<text x="{self.margin-10}" y="{y_svg+4}" '
                          f'text-anchor="end" class="tick">{tick:.0e}</text>')
        
        # 绘制曲线
        path = []
        for x, y in zip(x_data, y_data):
            x_svg, y_svg = self.data_to_svg(x, y, x_min, x_max,
                                           y_min, y_max, log_x=True, log_y=True)
            if not path:
                path.append(f"M{x_svg},{y_svg}")
            else:
                path.append(f"L{x_svg},{y_svg}")
        
        svg.append(f'<path d="{" ".join(path)}" class="curve"/>')
        
        # 数据点
        for x, y in zip(x_data, y_data):
            x_svg, y_svg = self.data_to_svg(x, y, x_min, x_max,
                                           y_min, y_max, log_x=True, log_y=True)
            svg.append(f'<circle cx="{x_svg}" cy="{y_svg}" r="3" class="point"/>')
        
        # 标题和标签
        svg.append(f'<text x="{self.width//2}" y="25" text-anchor="middle" class="title">'
                   f'{title}</text>')
        svg.append(f'<text x="{self.width//2}" y="{self.height-10}" '
                   f'text-anchor="middle" class="label">{x_label}</text>')
        svg.append(f'<text x="20" y="{self.height//2}" '
                   f'text-anchor="middle" class="label" transform="rotate(-90 20 {self.height//2})">'
                   f'{y_label}</text>')
        
        svg.append('</svg>')
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(svg))
            return True
        except Exception as e:
            print(f"❌ 写入失败: {e}")
            return False
    
    def _get_log_ticks(self, v_min: float, v_max: float) -> List[float]:
        """获取对数刻度"""
        ticks = []
        min_exp = int(math.floor(math.log10(v_min)))
        max_exp = int(math.ceil(math.log10(v_max)))
        for exp in range(min_exp, max_exp + 1):
            ticks.append(10 ** exp)
        return ticks


class ComprehensiveVisualizer:
    """综合地震分析可视化类"""
    
    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    def parse_csv(self, filename: str, col1: int = 0, col2: int = 1) -> Tuple[List[float], List[float]]:
        """解析 CSV 文件"""
        data1, data2 = [], []
        
        if not os.path.exists(filename):
            return data1, data2
        
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) > max(col1, col2):
                        try:
                            data1.append(float(parts[col1].strip()))
                            data2.append(float(parts[col2].strip()))
                        except ValueError:
                            continue
        except Exception as e:
            print(f"❌ 读取 {filename} 失败: {e}")
        
        return data1, data2
    
    # ========== matplotlib 可视化方法 ==========
    
    def plot_acceleration_spectrum_mpl(self) -> bool:
        """matplotlib 加速度谱"""
        freqs, accels = self.parse_csv('地震动加速度谱.csv')
        if not freqs:
            return False
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.loglog(freqs, accels, 'b-', linewidth=2.5, marker='o', markersize=3, alpha=0.8, label='加速度谱')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_xlabel('频率 (Hz)', fontsize=12, fontweight='bold')
        ax.set_ylabel('加速度谱 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('地震加速度反应谱', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.set_facecolor('#f8f9fa')
        
        output_file = os.path.join(self.output_dir, '加速度反应谱.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_uhs_mpl(self) -> bool:
        """matplotlib 一致危险谱"""
        periods, sa = self.parse_csv('一致危险谱_50年20%.csv')
        if not periods:
            return False
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.loglog(periods, sa, 'r-', linewidth=2.5, marker='o', markersize=6, label='UHS (50yr, 20%)')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('一致危险谱\n50年超越概率 20%', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.set_facecolor('#f8f9fa')
        
        output_file = os.path.join(self.output_dir, '一致危险谱_PNG.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_cms_mpl(self) -> bool:
        """matplotlib 条件均值谱"""
        periods, sa = self.parse_csv('条件均值谱.csv')
        if not periods:
            return False
        
        fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
        ax.loglog(periods, sa, 'g-', linewidth=2.5, marker='s', markersize=6, label='CMS (T=1.0s)')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('条件均值谱\n控制周期 T=1.0s', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11)
        ax.set_facecolor('#f8f9fa')
        
        output_file = os.path.join(self.output_dir, '条件均值谱_PNG.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: {output_file}")
        return True
    
    def plot_combined_mpl(self) -> bool:
        """matplotlib UHS vs CMS 对比"""
        uhs_periods, uhs_sa = self.parse_csv('一致危险谱_50年20%.csv')
        cms_periods, cms_sa = self.parse_csv('条件均值谱.csv')
        
        if not uhs_periods or not cms_periods:
            return False
        
        fig, ax = plt.subplots(figsize=(12, 8), dpi=150)
        ax.loglog(uhs_periods, uhs_sa, 'r-', linewidth=2.5, marker='o', markersize=5, label='UHS (50yr, 20%)')
        ax.loglog(cms_periods, cms_sa, 'g--', linewidth=2.5, marker='s', markersize=5, label='CMS (T=1.0s)')
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_xlabel('周期 (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('谱加速度 (cm/s²)', fontsize=12, fontweight='bold')
        ax.set_title('危险性分析对比: UHS vs CMS', fontsize=13, fontweight='bold', pad=15)
        ax.legend(loc='best', fontsize=11, framealpha=0.95)
        ax.set_facecolor('#f8f9fa')
        
        output_file = os.path.join(self.output_dir, 'UHS_vs_CMS_PNG.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ 保存: {output_file}")
        return True
    
    # ========== SVG 可视化方法 ==========
    
    def plot_acceleration_spectrum_svg(self) -> bool:
        """SVG 加速度谱"""
        freqs, accels = self.parse_csv('地震动加速度谱.csv')
        if not freqs:
            return False
        
        plotter = SVGPlotter()
        output_file = os.path.join(self.output_dir, '加速度反应谱.svg')
        result = plotter.create_spectrum_svg(freqs, accels, '频率 (Hz)', '加速度谱 (cm/s²)',
                                            '地震加速度反应谱', output_file, color='#0066cc')
        if result:
            print(f"✓ 保存: {output_file}")
        return result
    
    def plot_uhs_svg(self) -> bool:
        """SVG 一致危险谱"""
        periods, sa = self.parse_csv('一致危险谱_50年20%.csv')
        if not periods:
            return False
        
        plotter = SVGPlotter()
        output_file = os.path.join(self.output_dir, '一致危险谱_SVG.svg')
        result = plotter.create_spectrum_svg(periods, sa, '周期 (s)', '谱加速度 (cm/s²)',
                                            '一致危险谱 (50年超越概率20%)', output_file, color='#d62728')
        if result:
            print(f"✓ 保存: {output_file}")
        return result
    
    def plot_cms_svg(self) -> bool:
        """SVG 条件均值谱"""
        periods, sa = self.parse_csv('条件均值谱.csv')
        if not periods:
            return False
        
        plotter = SVGPlotter()
        output_file = os.path.join(self.output_dir, '条件均值谱_SVG.svg')
        result = plotter.create_spectrum_svg(periods, sa, '周期 (s)', '谱加速度 (cm/s²)',
                                            '条件均值谱 (控制周期T=1.0s)', output_file, color='#2ca02c')
        if result:
            print(f"✓ 保存: {output_file}")
        return result
    
    def visualize_all(self) -> bool:
        """生成所有可视化"""
        print("\n" + "="*70)
        print("开始综合地震分析可视化...")
        print("="*70 + "\n")
        
        if HAS_MATPLOTLIB:
            print("✓ 使用 matplotlib 高精度绘图模式 (PNG)\n")
            
            print("[1] 生成加速度反应谱...")
            self.plot_acceleration_spectrum_mpl()
            
            print("\n[2] 生成一致危险谱...")
            self.plot_uhs_mpl()
            
            print("\n[3] 生成条件均值谱...")
            self.plot_cms_mpl()
            
            print("\n[4] 生成 UHS vs CMS 对比图...")
            self.plot_combined_mpl()
        else:
            print("✓ 使用 SVG 纯 Python 绘图模式\n")
            
            print("[1] 生成加速度反应谱...")
            self.plot_acceleration_spectrum_svg()
            
            print("\n[2] 生成一致危险谱...")
            self.plot_uhs_svg()
            
            print("\n[3] 生成条件均值谱...")
            self.plot_cms_svg()
        
        print("\n" + "="*70)
        print("✓ 综合可视化生成完成！")
        print("="*70 + "\n")
        return True


def main():
    """主函数"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    visualizer = ComprehensiveVisualizer(output_dir=script_dir)
    
    if not visualizer.visualize_all():
        print("\n❌ 可视化生成失败")
        return 1
    
    print("📊 生成的可视化文件：")
    print("\n加速度谱可视化：")
    print("  • 加速度反应谱（频域分析）")
    
    print("\n危险性分析可视化：")
    print("  • 一致危险谱 (UHS) - 50年超越概率20%")
    print("  • 条件均值谱 (CMS) - 控制周期T=1.0s")
    print("  • UHS vs CMS 对比分析")
    
    if HAS_MATPLOTLIB:
        print("\n✓ 输出格式: PNG (高精度栅格图形)")
    else:
        print("\n✓ 输出格式: SVG (矢量图形)")
    
    print("\n💡 应用场景：")
    print("  1. 地震工程设计 - 选择合适的设计谱")
    print("  2. 危险性评估 - 评估地震风险")
    print("  3. 结构分析 - 进行动力响应分析")
    print("  4. 学术研究 - 地震学和工程地震学研究")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
