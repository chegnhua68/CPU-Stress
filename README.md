# Linux CPU/OpenCV 压力测试

这是一个面向树莓派、香橙派、工控小主机等小体型 Linux 设备的 CPU/OpenCV 压力测试工具。它会优先读取项目 `pictures/` 目录中的图片，并按图片原始分辨率测试 OpenCV 在图片读写、二值化、形态学、轮廓提取和连通域分析上的表现；如果没有图片，也可以自动回退到合成渲染场景，效果接近一个轻量版的 Cinebench 图像场景压测。

## 测试内容

- 读取 `pictures/` 中的 PNG/JPG/BMP/TIFF/WebP 图片
- 分析任意分辨率的原图，不缩放、不裁剪
- 在没有输入图片时渲染合成测试图片
- 测试 `cv2.imwrite` 写图片性能
- 测试 `cv2.imread` 读图片性能
- 测试灰度转换、Gaussian blur、Otsu 二值化
- 测试形态学开闭运算
- 测试 `cv2.findContours` 形状/轮廓提取
- 测试 `cv2.connectedComponentsWithStats` 连通域分析
- 输出 CSV 和 JSON 报告，便于后续对比不同机器

## 安装

建议在 Linux 测试机上使用 Python 3.11 虚拟环境：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

安装后可先做一次环境检查：

```bash
python3.11 check_env.py
```

如果是在 Windows 上准备环境，且 Python 3.11 由 Scoop 安装在 E 盘，可先确认路径：

```powershell
E:\Software\Scoop\Apps\apps\python311\current\python.exe --version
E:\Software\Scoop\Apps\apps\python311\current\python.exe check_env.py
```

实际压力测试建议在目标 Linux 设备上运行，这样结果才反映设备自身 CPU、内存和存储 I/O 表现。

如果当前 Windows 机器只有 Scoop 的新版 Python，例如 Python 3.14，而 NumPy/OpenCV wheel 不稳定，请不要用它作为最终测试环境；在 Linux 目标机上使用 Python 3.11 虚拟环境更可靠。

在你当前这台 Windows 机器上，也可以直接用 Scoop 的 Python 3.11 做功能验证。默认会读取 `pictures/` 中的原图分辨率：

```powershell
E:\Software\Scoop\Apps\apps\python311\current\python.exe cpu_stress_opencv.py -n 3
```

## 快速运行

默认会扫描项目根目录下的 `pictures/` 文件夹。当前仓库中可放置例如：

```text
pictures/linux_penguin_max_upscaled_8192x8192.png
```

在树莓派等 Linux 设备上，保持目录名为小写 `pictures` 即可；Linux 文件名区分大小写，所以不要写成 `Pictures` 或 `picture`。

默认扫描 `pictures/`，每张图片按原始分辨率跑 3 次：

```bash
python3.11 cpu_stress_opencv.py
```

如果 `pictures/` 中没有图片，会自动回退到合成图片模式。也可以主动指定合成图片分辨率：

```bash
python3.11 cpu_stress_opencv.py --synthetic -r 720p 1080p 2k 4k -n 3
```

合成图片自定义分辨率，例如 1600x900：

```bash
python3.11 cpu_stress_opencv.py --synthetic -r 1600x900 -n 5
```

保存预览图和每轮生成的原图：

```bash
python3.11 cpu_stress_opencv.py --preview --keep-images
```

保存处理后的图片，包括灰度图、二值化图、形态学结果、轮廓叠加图和连通域伪彩色图：

```bash
python3.11 cpu_stress_opencv.py -n 1 --save-stages --preview
```

限制 OpenCV 线程数，便于做单线程或固定线程对比：

```bash
python3.11 cpu_stress_opencv.py -n 5 --threads 1
```

指定其他图片目录：

```bash
python3.11 cpu_stress_opencv.py --picture-dir pictures -n 3
```

指定单张图片：

```bash
python3.11 cpu_stress_opencv.py --source-image pictures/linux_penguin_max_upscaled_8192x8192.png -n 3
```

强制使用合成渲染场景，不读取 `pictures/`：

```bash
python3.11 cpu_stress_opencv.py --synthetic -r 720p 1080p -n 3
```

## 输出结果

默认输出到 `benchmark_output/`：

- `samples.csv`：每一次迭代的详细耗时
- `report.json`：机器信息、汇总结果、详细样本
- `preview_*.jpg`：使用 `--preview` 时生成的预览图
- `processed/*.jpg`：使用 `--save-stages` 时生成的处理结果图

终端中会显示类似：

```text
Summary
resolution avg_total_ms avg_write_ms avg_read_ms avg_threshold_ms ...
720p      185.123      42.100       16.300      9.800
1080p     402.881      91.442       34.120      22.713
```

其中 `opencv_megapixels_per_second` 越高，说明 OpenCV 纯图像处理阶段吞吐越好；`avg_write_ms` 和 `avg_read_ms` 则更受存储和文件格式影响。

## 参数

```text
-r, --resolutions    合成图片分辨率，可选 720p 1080p 2k 4k 或 WIDTHxHEIGHT
-n, --iterations     每个分辨率重复次数
--shapes             合成模式下 1080p 的基础图形数量，其他分辨率按像素比例缩放
--picture-dir        图片目录，默认 pictures；相对路径会从项目根目录解析
--source-image       指定单张输入图片
--synthetic          强制使用合成图片，不扫描 pictures
--output-dir         输出目录
--format             图片格式：png、jpg、bmp
--threads            OpenCV 线程数，0 表示使用默认值
--seed               随机种子，便于复现实验
--keep-images        保留每轮生成的图片
--preview            每个分辨率保存一张预览图
--save-stages        保存灰度、二值化、形态学、轮廓、连通域处理结果图
--json               指定 JSON 报告路径
--csv                指定 CSV 报告路径
```

## 嵌入式 Linux 测试建议

- 把测试图片放进项目的 `pictures/` 目录，树莓派上 clone 后可直接识别
- 先跑 `720p` 和 `1080p`，确认内存、散热和依赖没有问题后再跑 `2k`、`4k`
- 使用相同 `--seed`、`--iterations`、`--format` 对比不同机器
- 想更关注 CPU 图像算法性能，可使用 `--format bmp` 降低压缩编码影响
- 想更贴近日常图片场景，可使用默认 `png` 或 `jpg`
- 长时间压测时可以配合 `htop`、`vcgencmd measure_temp`、`sensors` 观察温度和降频
