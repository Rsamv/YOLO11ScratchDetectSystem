# YOLO11ScratchDetectSystem

> 基于 YOLO11 的工业工件表面划痕缺陷检测系统 — 开箱即用的 PyQt5 桌面端 + TensorRT / ONNX Runtime 加速推理

![Python](https://img.shields.io/badge/python-3.10+-green)
![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)
![License](https://img.shields.io/badge/license-MIT-yellow)
![YOLO](https://img.shields.io/badge/YOLO-v11-blue)

## 解决什么问题？

工业产线上，人工目视检测划痕效率低、漏检率高、标准不统一。传统机器视觉方案又需要大量定制开发。

YOLO11ScratchDetectSystem 的目标：**拍一张图进来，系统自动告诉你有没有划痕、在哪里、多严重。**

- **不会写代码也能用** — PyQt5 图形界面，点点鼠标就行
- **检测速度够快** — TensorRT 加速下单张图片推理毫秒级
- **结果可量化** — 缺陷数量、位置坐标、置信度、物理距离，全部输出
- **标准可配置** — 铝合金零件的判定阈值、聚集判定规则，随时调

## 核心功能

### 🔍 多源检测

| 输入源 | 说明 |
|--------|------|
| **单张图片** | 拖入或选择文件，即检即出结果 |
| **视频文件** | 支持 mp4/avi 等常见格式，逐帧检测 |
| **摄像头** | 实时检测，支持外接 USB 摄像头 |

### ⚡ 双引擎推理

| 引擎 | 适用场景 | 依赖 |
|------|---------|------|
| **TensorRT** | NVIDIA GPU 加速，速度最快 | CUDA 12 + cuDNN 9 + TensorRT 10 |
| **ONNX Runtime** | CPU / GPU 通用，部署门槛低 | 仅需 `onnxruntime` |

首次运行时自动将 ONNX 模型转换为 TensorRT 引擎，无需手动操作。

### 📐 距离计算

内置像素→物理距离换算模块，支持标定后直接输出缺陷的实际尺寸（mm）。

### 📊 聚集检测

自动识别缺陷聚集区域，区分「散布型」和「集中型」缺陷，辅助判定工件是否合格。

### 🎨 现代化界面

- QSS 主题样式表，清爽不辣眼
- 检测结果实时标注在预览图上
- 缺陷列表 + 统计面板一目了然

### 💾 结果导出

检测结果支持 CSV 导出，包含：缺陷类别、置信度、坐标、尺寸、时间戳，方便后续分析。

## 截图

> TODO: 添加程序运行截图

## 快速开始

### 环境要求

- Windows 10/11 (64-bit)
- Python 3.10+
- NVIDIA GPU（可选，TensorRT 加速需要）

### 安装

```bash
# 克隆仓库
git clone https://github.com/Rsamv/YOLO11ScratchDetectSystem.git
cd YOLO11ScratchDetectSystem

# 创建虚拟环境
conda create -n scratch python=3.11
conda activate scratch

# 安装依赖
pip install -r requirements.txt

# （可选）TensorRT 加速
pip install tensorrt>=10.0
```

### 准备模型权重

将训练好的 YOLO 模型权重放入 `weights/` 目录：

```
weights/
├── best.pt    # PyTorch 权重（可选）
├── best.onnx  # ONNX 模型
└── best.trt   # TensorRT 引擎（首次运行自动生成）
```

> 权重文件不随仓库分发，请自行训练或从其他途径获取。详见 [weights/README.md](weights/README.md)。

### 运行

```bash
python ui.py
```

### 打包为 EXE

```bash
build.bat
```

打包产物在 `dist/ScratchDetect/` 目录。（PyInstaller 输出名保持 ScratchDetect）

## 项目结构

```
YOLO11ScratchDetectSystem/
├── ui.py                    # 主界面（Qt5 前端，6000+ 行）
├── detect_qt5.py            # 检测后端（TensorRT/ONNX 推理引擎）
├── distance_calc.py         # 距离计算模块
├── utils.py                 # 辅助工具函数
├── theme.qss                # UI 主题样式表
├── logo.ico                 # 应用图标
├── build.bat                # PyInstaller 构建脚本
├── pyi_runtime_dll_paths.py # 运行时 DLL 路径
├── requirements.txt         # Python 依赖
├── weights/                 # 模型权重（不随仓库分发）
├── data/                    # 数据目录
└── LICENSE                  # MIT 许可证
```

## 技术栈

| 层级 | 技术 |
|------|------|
| GUI 框架 | PyQt5 |
| 检测模型 | YOLO11 (Ultralytics) |
| GPU 加速 | TensorRT 10 + CUDA 12 |
| 通用推理 | ONNX Runtime |
| 图像处理 | OpenCV + NumPy |
| 打包 | PyInstaller |

## 开发路线

- [x] V1 — 基础图片检测 + PyQt5 界面
- [x] V2 — 视频/摄像头实时检测
- [x] V3 — TensorRT 加速推理
- [x] V4 — 距离计算 + 聚集检测
- [x] V5 — 检测标准配置 + 数据导出
- [x] V6 — 界面美化 + 多源检测整合
- [ ] V7 — 模型热更新 + 批量检测模式

## License

MIT — [Rsamv](https://github.com/Rsamv)

## 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) — 检测模型
- [PyQt5](https://www.riverbankcomputing.com/software/pyqt/) — GUI 框架
- [TensorRT](https://developer.nvidia.com/tensorrt) — GPU 推理加速
- [ONNX Runtime](https://onnxruntime.ai/) — 跨平台推理引擎
