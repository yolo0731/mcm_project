#!/usr/bin/env python3
"""
MCM Project Setup Script
高速列车轴承智能故障诊断项目安装脚本
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def create_directories():
    """创建必要的目录"""
    dirs = [
        'data/raw/源域数据集',
        'data/raw/目标域数据集',
        'data/processed',
        'figs/问题1',
        'figs/问题2',
        'figs/问题3',
        'figs/问题4',
        'models',
        'logs'
    ]

    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {dir_path}")

def install_requirements():
    """安装依赖包"""
    print("📦 Installing requirements...")
    os.system("pip install -r requirements.txt")

def check_pretrained_models():
    """检查预训练模型文件"""
    print("🤖 Checking pre-trained models...")
    model_files = [
        'models/pretrained/source_pretrained_model.pth',
        'models/pretrained/dann_model.pth'
    ]

    all_present = True
    for model_file in model_files:
        if Path(model_file).exists():
            size = Path(model_file).stat().st_size / (1024*1024)  # MB
            print(f"✅ Found {model_file} ({size:.1f}MB)")
        else:
            print(f"❌ Missing {model_file}")
            all_present = False

    return all_present

def run_basic_test():
    """运行基础测试"""
    print("🧪 Running basic tests...")
    try:
        import torch
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        import sklearn
        print("✅ All core dependencies imported successfully")

        # Check pre-trained models
        models_ok = check_pretrained_models()
        if models_ok:
            print("✅ Pre-trained models ready")
        else:
            print("⚠️  Some pre-trained models missing (will use simulation)")

        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """主安装流程"""
    print("🚀 MCM Project Setup")
    print("=" * 50)

    # 检查Python版本
    check_python_version()

    # 创建目录结构
    create_directories()

    # 安装依赖
    install_requirements()

    # 运行测试
    if run_basic_test():
        print("\n✅ Setup completed successfully!")
        print("\n📋 Next steps:")
        print("1. Place your data files in data/raw/ directories")
        print("2. Run: cd src/问题1 && python 1.py")
        print("3. Follow the README.md for detailed instructions")
    else:
        print("\n❌ Setup failed. Please check error messages above.")

if __name__ == "__main__":
    main()