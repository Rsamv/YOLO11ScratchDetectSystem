# 模型权重文件

此目录用于存放 YOLO 检测模型的权重文件。

## 需要的文件

| 文件名 | 说明 |
|--------|------|
| `best.pt` | PyTorch 原始训练权重 |
| `best.onnx` | ONNX 导出模型 |
| `best.trt` | TensorRT 引擎文件（运行时自动生成） |

## 获取方式

1. **自行训练**: 使用 Ultralytics YOLO 框架在自定义数据集上训练
2. **导出 ONNX**: `yolo export model=best.pt format=onnx`
3. **生成 TensorRT**: 程序首次运行时会自动将 ONNX 转换为 TensorRT 引擎

> **注意**: 权重文件不随仓库分发，请自行训练或从其他途径获取后放置于此目录。
