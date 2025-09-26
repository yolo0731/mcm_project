# MCM Project - 高速列车轴承智能故障诊断

本项目是基于深度学习和域适应技术的高速列车轴承智能故障诊断系统，包含四个问题的完整解决方案。

## 项目概述

本项目实现了基于CDAN（Conditional Domain Adversarial Networks）的轴承故障诊断系统，具有以下特性：

- 🎯 **问题1**: 数据预处理与特征提取分析
- 🔄 **问题2**: 域适应方法对比分析
- 🤖 **问题3**: CDAN模型实现与完整可视化分析
- 🔍 **问题4**: 模型可解释性分析与评估

## 项目结构

```
mcm_project/
├── src/                    # 源代码
│   ├── 问题1/              # 数据预处理与特征分析
│   ├── 问题2/              # 域适应方法对比
│   ├── 问题3/              # CDAN模型主实现
│   └── 问题4/              # 可解释性分析
├── data/                   # 数据目录
│   ├── raw/               # 原始数据
│   │   ├── 源域数据集/     # 源域轴承数据
│   │   └── 目标域数据集/   # 目标域轴承数据
│   └── processed/         # 处理后的数据
├── models/                # 预训练模型
│   ├── pretrained/        # 训练好的模型文件
│   │   ├── source_pretrained_model.pth  # 源域预训练模型 (2.6MB)
│   │   └── dann_model.pth              # DANN域适应模型 (2.8MB)
│   └── README.md          # 模型说明文档
├── figs/                  # 生成的图表
├── docs/                  # 文档
├── questions/            # 问题描述文档
└── README.md             # 本文件
```

## 环境要求

### Python 版本
- Python 3.8+

### 核心依赖
```bash
pip install torch torchvision torchaudio
pip install numpy pandas matplotlib seaborn
pip install scikit-learn scipy
pip install pathlib argparse
```

### 完整依赖安装
```bash
pip install -r requirements.txt
```

## 数据准备

### 1. 数据结构
项目需要以下数据文件：

**源域数据集** (在 `data/raw/源域数据集/`):
- 48kHz_DE_data/ 目录包含各种故障类型的.mat文件
- 包括：Normal, Ball Fault (B), Inner Ring Fault (IR), Outer Ring Fault (OR)

**目标域数据集** (在 `data/raw/目标域数据集/`):
- A.mat 到 P.mat 共16个测试文件

### 2. 特征文件
预处理后的特征文件位于 `data/processed/`:
- `94feature.csv` - 源域特征数据
- `16feature.csv` - 目标域特征数据

### 3. 预训练模型 ⭐
**本项目包含已训练完成的模型文件**，可以直接使用：
- `models/pretrained/source_pretrained_model.pth` - 源域预训练模型 (2.6MB)
- `models/pretrained/dann_model.pth` - DANN域适应模型 (2.8MB)

**优势**：
- ✅ 无需重新训练，节省时间
- ✅ 直接运行推理和可视化分析
- ✅ 模型性能已验证（目标域准确率85%+）

> 📖 详细模型信息请参考：[MODELS.md](MODELS.md)

## 快速开始

### 方法1：一键自动运行（推荐）
```bash
# 1. 克隆项目（包含预训练模型）
git clone https://github.com/your-username/mcm_project.git
cd mcm_project

# 2. 自动安装和设置
python setup.py

# 3. 一键运行所有问题（使用预训练模型）
python run_all.py
```

> 🎯 **优势**：项目自带预训练模型，克隆后即可直接运行，无需等待训练过程！

### 方法2：手动逐步运行
```bash
# 1. 克隆项目
git clone https://github.com/your-username/mcm_project.git
cd mcm_project

# 2. 安装依赖
pip install -r requirements.txt
```

### 3. 运行各个问题

#### 问题1：数据预处理与特征分析
```bash
cd src/问题1
python 1.py
```
**输出**：
- 生成特征提取和数据分析图表
- 保存在 `figs/问题1/` 目录

#### 问题2：域适应方法对比
```bash
cd src/问题2
python 2.py
```
**输出**：
- 生成域适应方法对比图表
- 保存在 `figs/问题2/` 目录

#### 问题3：CDAN模型与完整可视化
```bash
cd src/问题3
python 3.py
```
**输出**：
- 生成22个完整的可视化图表
- 包括训练曲线、t-SNE图、可靠性分析等
- 保存在 `figs/问题3/` 目录

#### 问题4：可解释性分析
```bash
cd src/问题4
python 4.py
```
**输出**：
- 生成可解释性分析报告
- 保存分析结果和图表
- 输出在 `data/processed/问题4/` 和 `figs/问题4/`

## 详细复现步骤

### 步骤1：环境设置
```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\\Scripts\\activate  # Windows

# 安装依赖
pip install torch numpy pandas matplotlib seaborn scikit-learn scipy pathlib
```

### 步骤2：数据验证
```bash
# 检查数据文件是否存在
ls data/raw/源域数据集/
ls data/raw/目标域数据集/
ls data/processed/
```

### 步骤3：按顺序运行
```bash
# 1. 运行问题1（数据预处理）
cd src/问题1
python 1.py
cd ../..

# 2. 运行问题2（域适应对比）
cd src/问题2
python 2.py
cd ../..

# 3. 运行问题3（主要CDAN实现）
cd src/问题3
python 3.py
cd ../..

# 4. 运行问题4（可解释性分析）
cd src/问题4
python 4.py
cd ../..
```

### 步骤4：查看结果
```bash
# 查看生成的图表
ls figs/问题1/
ls figs/问题2/
ls figs/问题3/
ls figs/问题4/

# 查看处理结果
ls data/processed/问题*/
```

## 主要功能模块

### 问题1 - 数据预处理与特征分析
- **文件**: `src/问题1/1.py`
- **功能**:
  - 原始振动信号预处理
  - 时域和频域特征提取
  - 故障特征可视化分析
  - 数据质量评估

### 问题2 - 域适应方法对比
- **文件**: `src/问题2/2.py`
- **功能**:
  - 不同域适应方法对比
  - 性能评估与可视化
  - 最优方法选择

### 问题3 - CDAN模型实现（核心）
- **文件**: `src/问题3/3.py`
- **功能**:
  - 完整的CDAN模型实现
  - 22个专业可视化图表
  - 包含训练曲线、t-SNE分析、可靠性评估等
  - 生成论文级别的图表

**支持文件**:
- `train_source.py` - 源域预训练
- `train_dann.py` - DANN训练
- `inference.py` - 推理预测
- `tsne_analysis.py` - t-SNE可视化
- `reporting.py` - 报告生成

### 问题4 - 可解释性分析
- **文件**: `src/问题4/4.py`
- **功能**:
  - SHAP特征重要性分析
  - 模型决策可视化
  - 物理机理验证
  - 综合可解释性评估

## 预训练模型使用 🤖

### 模型文件说明
本项目提供两个预训练模型：

1. **源域预训练模型** (`source_pretrained_model.pth`, 2.6MB)
   - 在源域数据上训练的基础分类模型
   - 验证准确率: ~85%

2. **DANN域适应模型** (`dann_model.pth`, 2.8MB)
   - 经过域适应训练的完整模型
   - 目标域准确率: ~85%
   - 可直接用于目标域预测

### 直接使用预训练模型
```python
import torch
from src.问题3.models import DANNModel

# 加载DANN模型
model = DANNModel()
model.load_state_dict(torch.load('models/pretrained/dann_model.pth', map_location='cpu'))
model.eval()

# 进行预测
with torch.no_grad():
    predictions = model(input_data)
```

### 无需训练的快速体验
```bash
# 直接运行推理（使用预训练模型）
cd src/问题3
python inference.py

# 生成所有可视化（基于预训练模型结果）
python 3.py
```

## 输出说明

### 可视化输出
- **问题1**: 8-12个数据分析图表
- **问题2**: 6-8个域适应对比图表
- **问题3**: 22个完整分析图表（包括PNG和PDF格式）
- **问题4**: 可解释性分析图表和报告

### 数据输出
- 特征数据CSV文件
- 模型预测结果
- 评估指标报告
- JSON格式分析结果

## 故障排除

### 常见问题

1. **ImportError: No module named 'xxx'**
   ```bash
   pip install xxx
   ```

2. **CUDA相关错误**
   ```bash
   # CPU模式运行
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **内存不足**
   - 减小batch_size参数
   - 使用较小的数据集进行测试

4. **matplotlib后端问题**
   - 项目已设置非交互式后端，正常情况下不会出现显示问题

### 数据文件缺失
如果某些数据文件缺失，程序会：
- 自动生成模拟数据
- 显示警告信息
- 继续执行后续分析

## 项目特性

- ✅ **完全自动化**: 一键运行所有分析
- 🤖 **预训练模型**: 包含完整训练好的DANN模型，无需重新训练
- 📊 **专业可视化**: 22+高质量图表
- 🔧 **容错处理**: 数据缺失时自动使用模拟数据
- 📖 **详细日志**: 完整的运行进度显示
- 🎯 **模块化设计**: 每个问题独立运行
- 🖼️ **论文级输出**: 适合直接用于学术报告
- ⚡ **快速部署**: 克隆即用，3分钟内可看到结果

## 技术栈

- **深度学习框架**: PyTorch
- **数据处理**: NumPy, Pandas
- **可视化**: Matplotlib, Seaborn
- **机器学习**: Scikit-learn
- **信号处理**: SciPy

## 贡献

欢迎提交Issue和Pull Request来改进项目。

## 许可证

本项目采用MIT许可证。

## 联系方式

如有问题，请通过以下方式联系：
- 提交GitHub Issue
- 邮箱: your-email@example.com

---

**注意**: 本项目为学术研究项目，仅用于教育和研究目的。