# MiniMax-H3 FL2VA 海光八卡适配

队伍：**究竟对不队**。仓库：<https://github.com/shiqigeng1/songshan-minimax-h3>。提交标签：`prelim-20261111`。

本项目提供 InfiniLM / InfiniCore 固定源码、海光 MHA 兼容补丁、九齿独立算子实验、构建运行脚本、报告及实验结果。
**当前原生生成存在棋盘纹异常，对参考视频平均 SSIM 为 8.997%；尺寸和帧数一致，生成内容一致性尚未解决。**
已知限制和原始记录见 [实测结果](docs/RESULTS.md)，独立算子 3.732× 不属于整模型加速结果。

## 文件结构

| 目录 | 内容 |
| --- | --- |
| `upstream/InfiniCore/` | 固定版本完整源码及第三方子模块内容 |
| `upstream/InfiniLM/` | 固定版本源码、`examples/minimax_h3.py` 入口和原始比较程序 |
| `patches/` | 海光 MHA 构建与注册适配补丁 |
| `scripts/` | Python 环境、Core/LM 构建、FL2VA 运行、比较与算子实验入口 |
| `tools/xmake` | 展开的 Linux x86-64 xmake 3.0.5 工具 |
| `dependencies/xmake-repo/` | 固定版本构建依赖定义及来源声明 |
| `requirements/` | 生成、比较和九齿环境的依赖版本 |
| `h3_ops/` | 九齿门控残差、Softmax、历史 SiLU 代码和数据 |
| `diagnostics/` | 底层算子、DiT 与条件编码定位脚本 |
| `reports/` | 指定模板填写的正式报告 |
| `evidence/` | 已有视频、预览、原始回传和结构化结果 |

源码和脚本直接展开，无须解压项目包。模型权重和平台系统软件由主办方环境提供。
没有个人 AFS 下载地址；安装目录由 `H3_WORK_HOME` 指定，可选用平台分配给使用者的持久目录。

## 1. 平台前提

- Linux x86-64、Python 3.11、DTK 26.04、海光 Torch 2.9.0 / HIP 6.3.26093 / C++ ABI 1。
- 完整生成使用八张 BW / gfx936 DCU，已测每卡约 64 GiB。
- gcc、g++、cmake、make、patch、git、flock，以及 Python 开发头文件。系统工具由平台维护。
- 首次安装依赖需要联网 CPU 工作区；离线模式需要使用者提供 wheel 目录和预备好的 xmake 依赖缓存。仓库不包含这些系统软件或模型权重。

## 2. 下载源码并准备 Python 环境

在联网 CPU 终端执行：

```bash
git clone https://github.com/shiqigeng1/songshan-minimax-h3.git
cd songshan-minimax-h3
git checkout prelim-20261111
export H3_WORK_HOME="${IFAB_AFS_HOME:-$HOME}/songshan-h3"
bash scripts/prepare_env.sh runtime
```

`H3_WORK_HOME` 可以设置为任意有写权限的安装目录。脚本继承平台海光 Torch，建立 Python venv，安装其余运行依赖。
`torchvision` 使用配套 CPU 包处理图片，`accelerate` 以 `--no-deps` 安装，避免替换海光 Torch。
环境文件写入 `$H3_WORK_HOME/env_gfx936.sh`；源码仍从本仓库读取。

首次构建的 Boost / pybind11 依赖可先在有网络的工作区缓存：

```bash
bash scripts/build_native.sh deps
```

## 3. 编译 InfiniCore 和 InfiniLM

在带兼容 DTK 的计算终端进入同一仓库，并重新设置相同的 `H3_WORK_HOME`：

```bash
export H3_WORK_HOME="${IFAB_AFS_HOME:-$HOME}/songshan-h3"
BUILD_JOBS=32 bash scripts/build_native.sh core
BUILD_JOBS=32 bash scripts/build_native.sh lm
```

`core` 自动复制源码到本地临时空间、应用 MHA 补丁、编译并安装四个基础库与 Python 扩展。
`lm` 复用安装的 Core，再构建 `_infinilm`。安装库、Python 运行文件和日志保留在 `H3_WORK_HOME`。
编译并行度按实际 CPU 配额设置，共享 CPU 登录节点应降低 `BUILD_JOBS`。
新整理的独立构建脚本已做本地语法与路径检查，尚未完成干净比赛环境的重建验证。

## 4. 准备统一首帧

报告保留了模板中的原始输入 PNG，可以提取原始字节：

```bash
python3 scripts/extract_first_frame.py --output "$H3_WORK_HOME/inputs/first-frame.png"
export H3_FIRST_FRAME="$H3_WORK_HOME/inputs/first-frame.png"
```

也可将 `H3_FIRST_FRAME` 指向主办方提供的同一张图片。
图像 SHA-256：`b97c3487ca35da0c803fba3787cf2b7f936d3d4595548ae5b7b576c19934be54`。

## 5. 八卡 FL2VA 运行

在八卡终端执行，模型路径按比赛挂载位置设置：

```bash
export H3_WORK_HOME="${IFAB_AFS_HOME:-$HOME}/songshan-h3"
export H3_FIRST_FRAME="$H3_WORK_HOME/inputs/first-frame.png"
export H3_MODEL_PATH=/public/home/aclsylqidf/shared/MiniMax-H3/diffusers
bash scripts/run_fl2va.sh \
  --model-path "$H3_MODEL_PATH" --first-frame "$H3_FIRST_FRAME" \
  --text 'A dog falls asleep' --gpus 0,1,2,3,4,5,6,7 \
  --num-frames 124 --steps 50 --seed 42 \
  --output "$H3_WORK_HOME/outputs/fl2va.mp4"
```

脚本将参数转交原始 `minimax_h3.py fl2va`。视频 VAE 解码选择 MATH SDPA，其他生成参数保持原始入口语义。
日志为 `$H3_WORK_HOME/logs/fl2va-run-*.log`。完整调用结束后返回，视频输出必须显式指定。

## 6. PyTorch 参考路径

参考路径的 FlashAttention 依赖需先在联网 CPU 工作区安装一次：

```bash
bash scripts/prepare_env.sh flash
```

随后在八卡终端重复第 5 步命令，加 `--legacy`，输出改为 `reference.mp4`。
新终端加载环境时会自动设置 `FA_SO_PATH` 为已安装 Flash 动态库的完整路径。
该 vendor wheel 约 629 MiB，可通过 `H3_FLASH_WHEEL` 指定本地安装包。

## 7. CPU 比较已有视频

```bash
bash scripts/prepare_env.sh compare
bash scripts/compare_existing.sh \
  "$H3_WORK_HOME/outputs/reference.mp4" "$H3_WORK_HOME/outputs/fl2va.mp4"
```

比较使用独立 venv，执行未修改的 `minimax_h3_compare.py --tasks fl2va`。
结果、完整输出和输入摘要保存到 `$H3_WORK_HOME/checks/fl2va-compare-*`。

## 8. 九齿独立门控残差实验

指定匹配平台的 Triton wheel，准备独立环境后，在 DCU 终端运行：

```bash
export H3_TRITON_WHEEL=/path/to/triton-3.5.1+das.opt1.dtk2604.torch290-cp311-cp311-manylinux_2_28_x86_64.whl
bash scripts/prepare_env.sh nine
bash scripts/run_operator.sh
```

此实验仅测单卡算子，不会改变正式模型路径。固定形状、语义、计时方法和负结果见 [RESULTS.md](docs/RESULTS.md)。

进一步说明：[构建与运行](docs/BUILD_AND_RUN.md) · [依赖](docs/DEPENDENCIES.md) · [版本来源](docs/SOURCE_PROVENANCE.md) · [报告](reports/README.md)。
