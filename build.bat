@echo off
chcp 65001 >nul
echo ============================================
echo   ScratchDetect PyInstaller build script
echo ============================================
echo.

set PYTHON=D:\CONDA\envs\py\python.exe
set PIP=D:\CONDA\envs\py\Scripts\pip.exe
set CONDA=D:\CONDA\envs\py

%PYTHON% --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python environment not found: %CONDA%
    pause
    exit /b 1
)

%PYTHON% -m PyInstaller --version >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not found, installing...
    %PIP% install pyinstaller
)

echo [INFO] Building ScratchDetect...
echo.

%PYTHON% -m PyInstaller ^
    --onedir ^
    --windowed ^
    --icon=logo.ico ^
    --name "ScratchDetect" ^
    --add-data "weights;weights" ^
    --add-data "data;data" ^
    --add-data "theme.qss;." ^
    --add-data "logo.ico;." ^
    --add-binary "%CONDA%\Library\bin\ffi.dll;." ^
    --add-binary "%CONDA%\Library\bin\ffi-7.dll;." ^
    --add-binary "%CONDA%\Library\bin\ffi-8.dll;." ^
    --add-binary "%CONDA%\Library\bin\liblzma.dll;." ^
    --add-binary "%CONDA%\Library\bin\libbz2.dll;." ^
    --add-binary "%CONDA%\Library\bin\libexpat.dll;." ^
    --add-binary "%CONDA%\Library\bin\libcrypto-3-x64.dll;." ^
    --add-binary "%CONDA%\Library\bin\libssl-3-x64.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\nvidia\cuda_runtime\bin\cudart64_12.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cublas64_12.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cublasLt64_12.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cufft64_11.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\curand64_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cusparse64_12.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\nvJitLink_120_0.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\nvrtc64_120_0.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\nvrtc-builtins64_124.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_adv64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_cnn64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_engines_precompiled64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_engines_runtime_compiled64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_graph64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_heuristic64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\torch\lib\cudnn_ops64_9.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_plugin_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvonnxparser_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_ptx_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm75_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm80_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm86_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm89_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm90_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm100_10.dll;." ^
    --add-binary "%CONDA%\Lib\site-packages\tensorrt_libs\nvinfer_builder_resource_sm120_10.dll;." ^
    --paths "%CONDA%\Library\bin" ^
    --paths "%CONDA%\Lib\site-packages\onnxruntime\capi" ^
    --paths "%CONDA%\Lib\site-packages\torch\lib" ^
    --paths "%CONDA%\Lib\site-packages\tensorrt_libs" ^
    --runtime-hook pyi_runtime_dll_paths.py ^
    --collect-binaries onnxruntime ^
    --collect-submodules onnxruntime.capi ^
    --copy-metadata onnxruntime-gpu ^
    --hidden-import cv2 ^
    --hidden-import numpy ^
    --hidden-import PyQt5 ^
    --hidden-import tensorrt ^
    --hidden-import onnxruntime ^
    --hidden-import onnxruntime.capi.onnxruntime_pybind11_state ^
    --exclude-module matplotlib ^
    --exclude-module IPython ^
    --exclude-module jupyter ^
    --exclude-module pandas ^
    --exclude-module scipy ^
    --exclude-module ultralytics ^
    --exclude-module torch ^
    --noconfirm ^
    ui.py

echo.
echo ============================================
if exist "dist\ScratchDetect\ScratchDetect.exe" (
    echo   Build completed.
    echo   Output directory: dist\ScratchDetect
) else (
    echo   Build may have failed. Check the PyInstaller output above.
)
echo ============================================
pause
