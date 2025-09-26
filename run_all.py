#!/usr/bin/env python3
"""
MCM Project - 一键运行所有问题
高速列车轴承智能故障诊断项目完整执行脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def print_header(title, problem_num):
    """打印问题标题"""
    print("\n" + "="*70)
    print(f"🎯 问题{problem_num}: {title}")
    print("="*70)

def run_python_script(script_path, problem_name):
    """运行Python脚本"""
    print(f"\n🚀 正在执行 {problem_name}...")
    print(f"📍 脚本路径: {script_path}")

    try:
        # 获取脚本目录
        script_dir = os.path.dirname(script_path)
        script_name = os.path.basename(script_path)

        # 切换到脚本目录执行
        original_dir = os.getcwd()
        os.chdir(script_dir)

        # 执行脚本
        start_time = time.time()
        result = subprocess.run([sys.executable, script_name],
                              capture_output=True, text=True,
                              encoding='utf-8', errors='replace')

        end_time = time.time()
        execution_time = end_time - start_time

        # 返回原目录
        os.chdir(original_dir)

        if result.returncode == 0:
            print(f"✅ {problem_name} 执行成功! 用时: {execution_time:.1f}s")
            if result.stdout:
                print("📄 输出信息:")
                print(result.stdout)
        else:
            print(f"❌ {problem_name} 执行失败!")
            if result.stderr:
                print("🚨 错误信息:")
                print(result.stderr)
            return False

        return True

    except Exception as e:
        print(f"❌ 执行 {problem_name} 时发生异常: {e}")
        return False

def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")

    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        return False

    # 检查必要的包
    required_packages = ['torch', 'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"❌ 缺少以下包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False

    print("✅ 环境检查通过")
    return True

def create_output_dirs():
    """创建输出目录"""
    print("📁 创建输出目录...")

    dirs = [
        'figs/问题1',
        'figs/问题2',
        'figs/问题3',
        'figs/问题4',
        'data/processed/问题1',
        'data/processed/问题2',
        'data/processed/问题3',
        'data/processed/问题4'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

    print("✅ 输出目录创建完成")

def main():
    """主执行流程"""
    print("🎯 MCM Project - 高速列车轴承智能故障诊断")
    print("🚀 一键运行所有问题解决方案")
    print("="*70)

    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，程序退出")
        return

    # 创建输出目录
    create_output_dirs()

    # 记录开始时间
    total_start_time = time.time()

    # 执行各个问题
    problems = [
        {
            'num': '1',
            'title': '数据预处理与特征提取分析',
            'script': 'src/问题1/1.py'
        },
        {
            'num': '2',
            'title': '域适应方法对比分析',
            'script': 'src/问题2/2.py'
        },
        {
            'num': '3',
            'title': 'CDAN模型实现与完整可视化分析',
            'script': 'src/问题3/3.py'
        },
        {
            'num': '4',
            'title': '模型可解释性分析与评估',
            'script': 'src/问题4/4.py'
        }
    ]

    successful_runs = 0
    failed_runs = []

    for problem in problems:
        print_header(problem['title'], problem['num'])

        if os.path.exists(problem['script']):
            success = run_python_script(problem['script'], f"问题{problem['num']}")
            if success:
                successful_runs += 1
            else:
                failed_runs.append(problem['num'])
        else:
            print(f"❌ 脚本文件不存在: {problem['script']}")
            failed_runs.append(problem['num'])

    # 总结报告
    total_end_time = time.time()
    total_execution_time = total_end_time - total_start_time

    print("\n" + "="*70)
    print("📊 执行总结报告")
    print("="*70)
    print(f"⏱️  总执行时间: {total_execution_time:.1f}s")
    print(f"✅ 成功执行: {successful_runs}/4 个问题")

    if failed_runs:
        print(f"❌ 执行失败: 问题 {', '.join(failed_runs)}")

    print(f"\n📁 输出文件位置:")
    print(f"   📊 可视化图表: figs/问题*/")
    print(f"   📄 处理结果: data/processed/问题*/")

    if successful_runs == 4:
        print("\n🎉 所有问题执行完成! 项目运行成功!")
    else:
        print(f"\n⚠️  {4-successful_runs} 个问题执行失败，请检查错误信息")

    print("\n📖 详细使用说明请参考 README.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  用户中断执行")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ 执行过程中发生意外错误: {e}")
        sys.exit(1)