# 树莓派拉取并运行 CPU-Stress Demo

本文档说明如何在树莓派或类似小型 Linux 设备上，从 GitHub 拉取本项目并运行 OpenCV CPU 压力测试 demo。

## 1. 准备系统依赖

先更新软件源并安装 Git、Python 虚拟环境工具：

```fish
sudo apt update
sudo apt install -y git python3 python3-venv python3-pip
```

如果你的系统已经有 Python 3.11，可以确认一下：

```fish
python3.11 --version
```

如果没有 Python 3.11，也可以先使用系统默认的 `python3` 运行。

## 2. 从 GitHub 拉取项目

进入你想保存项目的目录，例如家目录：

```fish
cd ~
```

克隆仓库：

```fish
git clone https://github.com/chegnhua68/CPU-Stress.git
```

进入项目目录：

```fish
cd CPU-Stress
```

确认项目文件：

```fish
ls
ls pictures
```

正常情况下应能看到：

```text
cpu_stress_opencv.py
requirements.txt
check_env.py
pictures/
```

## 3. 创建 Python 虚拟环境

如果树莓派上有 Python 3.11：

```fish
python3.11 -m venv .venv
```

如果没有 Python 3.11：

```fish
python3 -m venv .venv
```

激活虚拟环境：

```fish
source .venv/bin/activate.fish
```

升级 pip 并安装依赖：

```fish
python -m pip install -U pip
python -m pip install -r requirements.txt
```

如果需要生成函数级 profile 报告，额外安装：

```fish
python -m pip install -r requirements-dev.txt
```

## 4. 检查 OpenCV 环境

运行环境检查：

```fish
python check_env.py
```

如果看到类似下面的输出，说明环境基本可用：

```text
NumPy: ...
OpenCV: ...
OpenCV smoke test: OK
Environment looks ready.
```

## 5. 放置测试图片

默认图片目录是项目根目录下的 `pictures/`：

```text
CPU-Stress/
  pictures/
    DSC_5101.JPG
```

Linux 文件名区分大小写，请保持目录名为小写：

```text
pictures
```

不要写成：

```text
Pictures
picture
```

支持的图片格式包括：

```text
jpg, jpeg, png, bmp, tif, tiff, webp
```

## 6. 运行基础测试

每张图片按原始分辨率测试 1 次：

```fish
python cpu_stress_opencv.py -n 1
```

每张图片测试 3 次：

```fish
python cpu_stress_opencv.py -n 3
```

指定单张图片：

```fish
python cpu_stress_opencv.py --source-image pictures/DSC_5101.JPG -n 1
```

## 7. 输出处理后的图片

如果不仅想看性能结果，还想保存 OpenCV 处理后的图片，使用：

```fish
python cpu_stress_opencv.py -n 1 --save-stages --preview
```

如果还想生成 pyinstrument HTML profile：

```fish
python cpu_stress_opencv.py -n 1 --format jpg --save-stages --preview --profile
```

输出目录默认为：

```text
benchmark_output/
```

其中：

```text
benchmark_output/report.json       机器信息和汇总报告
benchmark_output/samples.csv       每次测试的详细耗时
benchmark_output/*_preview.jpg     预览图
benchmark_output/processed/*.jpg   处理后的图片
benchmark_output/profile.html      pyinstrument 函数级 profile 报告
```

`processed/` 中会包含：

```text
gray.jpg         灰度图
binary.jpg       二值化图
morphology.jpg   形态学处理结果
contours.jpg     轮廓叠加图
components.jpg   连通域伪彩色图
```

报告里的关键时间字段：

```text
input_read_ms             原图读入时间
benchmark_write_ms        基准图片写盘时间
benchmark_read_ms         基准图片读回时间
core_compute_ms           灰度、二值化、形态学、轮廓、连通域、质心计算总时间
cpu_cores                 系统检测到的 CPU 核心/逻辑处理器数量
opencv_threads            OpenCV 当前使用的线程数
stage_compute_ms          灰度图、二值图、轮廓图、连通域伪彩图等结果图生成时间
stage_write_ms            处理结果图写盘时间
output_write_ms           处理结果图和预览图输出时间
processed_pipeline_ms     核心计算 + 处理图生成 + 处理图输出总时间
centroid_ms               连通域质心计算总时间
centroid_avg_us           每个质心平均计算时间，单位微秒
centroid_rate_per_second  每秒可计算质心数量
total_ms                  本轮完整耗时
```

## 8. 推荐树莓派测试命令

树莓派性能较弱，建议先跑 1 次确认没有内存或散热问题：

```fish
python cpu_stress_opencv.py -n 1 --format jpg --save-stages --preview
```

如果运行稳定，再跑多轮：

```fish
python cpu_stress_opencv.py -n 3 --format jpg --save-stages --preview
```

如果只想看性能，不保存处理图：

```fish
python cpu_stress_opencv.py -n 3 --format jpg
```

如果要同时保存处理图和 profile：

```fish
python cpu_stress_opencv.py -n 1 --format jpg --save-stages --preview --profile
```

## 9. 查看测试结果

查看终端输出中的 Summary：

```text
Summary
resolution avg_total_ms avg_write_ms avg_read_ms ...
```

查看 JSON 报告：

```fish
cat benchmark_output/report.json
```

查看 CSV 报告：

```fish
cat benchmark_output/samples.csv
```

如果使用桌面版 Raspberry Pi OS，可以直接打开 `benchmark_output/processed/` 查看处理后的图片。

## 10. 同步后续更新

以后如果电脑端更新了代码并 push 到 GitHub，树莓派上进入项目目录执行：

```fish
cd ~/CPU-Stress
git pull
```

如果依赖有变化，再执行：

```fish
source .venv/bin/activate.fish
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

## 11. 常见问题

### pip 安装 OpenCV 很慢

可以尝试使用国内镜像：

```fish
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 内存不足或运行很慢

先换一张分辨率较低的图片，或者只跑 1 次：

```fish
python cpu_stress_opencv.py -n 1 --format jpg
```

### 找不到图片

确认图片在项目根目录的 `pictures/` 中：

```fish
pwd
ls pictures
```

也可以显式指定图片：

```fish
python cpu_stress_opencv.py --source-image pictures/DSC_5101.JPG -n 1
```
