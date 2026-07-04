# ScratchDetect - 工业划痕缺陷检测系统

基于 YOLO11 + PyQt5 的工业工件表面划痕缺陷检测系统，支持 TensorRT / ONNX Runtime 加速推理。

## 功能特性

- **多源检测** — 支持图片、视频、摄像头实时检测
- **多引擎推理** — TensorRT（NVIDIA GPU 加速）/ ONNX Runtime（CPU/GPU 通用）
- **实时预览** — Qt5 界面实时显示检测结果与标注
- **距离计算** — 内置像素-物理距离换算，支持实际尺寸测量
- **聚集检测** — 支持缺陷聚集区域判定与统计
- **检测标准配置** — 可自定义缺陷判定阈值与分类标准
- **数据导出** — 检测结果 CSV 导出，支持批量处理
- **主题美化** — 现代化 QSS 样式表，界面清爽易用

## 技术栈

| 组件 | 技术 |
|------|------|
| GUI 框架 | PyQt5 |
| 检测模型 | YOLO11 (Ultralytics) |
| GPU 加速 | TensorRT 10 + CUDA 12 |
| 通用推理 | ONNX Runtime |
| 图像处理 | OpenCV + NumPy |
| 打包工具 | PyInstaller |

## 环境要求

- **操作系统**: Windows 10/11 (64-bit)
- **Python**: 3.10+
- **GPU** (可选): NVIDIA GPU，CUDA 12.x + cuDNN 9.x
- **TensorRT** (可选): 10.x（用于 GPU 加速推理）

## 安装

### 1. 克隆仓库

```bash
git clone https://github.com/<your-username>/ScratchDetect.git
cd ScratchDetect
```

### 2. 创建虚拟环境

```bash
conda create -n scratch python=3.11
conda activate scratch
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

如果需要 TensorRT 加速（需 NVIDIA GPU）:

```bash
pip install tensorrt>=10.0
```

### 4. 准备模型权重

将训练好的 YOLO 模型权重文件放入 `weights/` 目录：

```
weights/
├── best.pt    # PyTorch 权重（可选）
├── best.onnx  # ONNX 模型
└── best.trt   # TensorRT 引擎（首次运行自动生成）
```

> 权重文件不随仓库分发，请自行训练或从其他途径获取。详见 [weights/README.md](weights/README.md)。

## 使用说明

### 启动程序

```bash
python ui.py
```

### PyInstaller 打包

```bash
build.bat
```

打包产物位于 `dist/ScratchDetect/` 目录。

## 项目结构

```
ScratchDetect/
├── ui.py                 # 主界面（Qt5 前端）
├── detect_qt5.py         # 检测后端（TensorRT/ONNX 推理引擎）
├── distance_calc.py      # 距离计算模块
├── utils.py              # 辅助工具函数
├── theme.qss             # UI 主题样式表
├── logo.ico              # 应用图标
├── pyi_runtime_dll_paths.py  # PyInstaller 运行时 DLL 路径
├── build.bat             # PyInstaller 构建脚本
├── requirements.txt      # Python 依赖
├── weights/              # 模型权重目录（不随仓库分发）
│   └── README.md
├── data/                 # 数据目录
│   └── source_image/
└── LICENSE               # MIT 许可证
```

## 截图

<!-- TODO: 添加程序运行截图 -->

## 许可证

本项目基于 [MIT License](LICENSE) 开源。

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 检测模型
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) — GUI 框架
- [TensorRT](https://developer.nvidia.com/tensorrt) — GPU 推理加速
- [ONNX Runtime](https://onnxruntime.ai/) — 跨平台推理引擎
