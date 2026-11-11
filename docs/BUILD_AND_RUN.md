# 构建与运行细节

按根目录 README 的 1—7 步完成环境准备、编译、统一首帧提取、生成和比较。
源码由 GitHub 仓库提供，不使用队伍个人存储作为源代码、权重或依赖下载地址。

## 工作目录

`H3_WORK_HOME` 是使用者可写的持久安装目录，默认为 `${IFAB_AFS_HOME:-$HOME}/songshan-h3`。
`IFAB_AFS_HOME` 只表示平台为当前使用者分配的目录。非 InfiniFabric 环境可直接设置其他路径。

| 目录 | 作用 |
| --- | --- |
| `$H3_WORK_HOME/venvs/native` | 继承平台 Torch 的生成 Python 环境 |
| `$H3_WORK_HOME/venvs/compare` | CPU 比较依赖隔离环境 |
| `$H3_WORK_HOME/venvs/nine` | 九齿独立实验环境 |
| `$H3_WORK_HOME/.infini-gfx936` | 原生头文件和共享库 |
| `$H3_WORK_HOME/runtime-gfx936` | 编译后的 Python 包和原始入口副本 |
| `$H3_WORK_HOME/logs` | 安装、构建和推理日志 |
| `$H3_WORK_HOME/outputs`、`checks` | 视频和比较结果 |

Core/LM 源码从 `upstream/` 复制到本地临时目录编译，避免 AFS `.deps/*.o.d` 并发写入问题。
`H3_SCRATCH_ROOT` 可以指定临时空间；未设置时使用 `IFAB_WORKSPACE_ROOT` 或 `/tmp`。
资源释放会清除临时空间，已安装运行文件保留。未完成的阶段可能需要重新编译。

## 编译命令与实际修改

Core 配置：`--hygon-dcu=y --hygon-arch=gfx936 --aten=y --ccl=y`，CUDA 兼容工具链来自 `$DTK_ROOT/cuda/cuda`。
`DTK_ROOT` 默认 `/opt/dtk`，也可指定 `HYGON_CUDA_HOME`。
构建使用 xmake 3.0.5，禁用编译缓存，C++ ABI 从平台 Torch 读取。
Core 安装目标为 infinirt、infiniop、infiniccl、infinicore_cpp_api 和 _infinicore；LM 目标为 _infinilm。
构建脚本在临时副本固定 Boost 1.89.0 和 pybind11 v3.0.1，Core 自动应用 `patches/gfx936-hygon-mha-aten.patch`。
上游快照不直接修改，比较脚本保持原始内容。

## 离线条件

首次依赖获取需要联网，后续可使用准备好的本地目录：

```bash
export H3_WHEEL_DIR=/path/to/individual-linux-cp311-wheels
export H3_OFFLINE=1
bash scripts/prepare_env.sh runtime
```

wheel 目录须包含固定版本运行依赖及其依赖项。Flash 用 `H3_FLASH_WHEEL`，Triton 用 `H3_TRITON_WHEEL` 指定平台 wheel。
xmake 离线构建需要已经准备的包缓存；`H3_PACKAGE_SEARCHDIR` 可指定其搜索目录。
空目录不能完成离线构建，仓库没有包含驱动、系统编译器、全部 Python 包或模型权重。

## 常见问题

- setuptools 的 stdlib/distutils 冲突：脚本设置 `SETUPTOOLS_USE_DISTUTILS=stdlib`。
- pip 无权读取 `/etc/mime.types`：安装入口禁用系统 MIME 文件扫描。
- CPU 终端 Core 导入报告设备枚举失败：完整推理在 DCU 终端运行；构建阶段只核对扩展文件，不在 CPU 导入模型。
- Flash 库找不到：加载环境后检查 `FA_SO_PATH`，它应为 `flash_attn_2_cuda*.so` 的完整路径。
- 原生输出棋盘纹：这是当前已知模型正确性问题；换编码格式或成功解码不代表内容正确。
- 编译时间超出资源时限：分别执行 core、lm，持久目录中的已安装 Core 可供下一轮 LM 构建使用。

当前说明中新增的独立准备/构建/启动脚本已做本地静态检查，未执行干净平台全流程验证；历史八卡实测及其局限以 RESULTS.md 为准。
