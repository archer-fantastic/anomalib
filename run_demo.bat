@echo off
REM 涂布缺陷检测 Demo 启动脚本
REM 使用前请确保已执行 convert_labelme_to_mask.py 生成 masks 目录

set PYTHONPATH=z:\14-调试数据\lxm\Projects\Anomalib\src
cd /d z:\14-调试数据\lxm\Projects\Anomalib

echo ========================================
echo   Anomalib 涂布缺陷检测 Demo (PatchCore)
echo ========================================
echo.
echo 数据目录: Z:\14-调试数据\lxm\Dataset\Anomalib\涂布
echo 配置: train_tubu.py
echo.

python train_tubu.py

echo.
pause
