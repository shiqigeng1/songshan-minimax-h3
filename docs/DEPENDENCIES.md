# 环境和依赖

仓库提供展开的源码、xmake 工具和包定义；Python 依赖通过 requirements 安装，或从使用者指定的 wheel 目录取得。
没有队伍个人 AFS 下载链接，也不分发项目 ZIP/TAR 包。

| 范围 | 版本/条件 |
| --- | --- |
| 平台 | Linux x86-64、Python 3.11、DTK 26.04 |
| Torch | 平台海光 2.9.0、HIP 6.3.26093、C++ ABI 1 |
| 构建 | xmake 3.0.5、Boost 1.89.0、pybind11 3.0.1、系统 CMake/GCC |
| 生成 | diffusers 0.40.0、transformers 5.14.1、accelerate 1.13.0、av 16.0.1、torchvision 0.24.0+cpu |
| Flash 参考 | flash-attn 2.6.1+das.opt1.dtk2604.torch290 |
| 九齿实验 | ninetoothed 0.26.0、海光 triton 3.5.1+das.opt1.dtk2604.torch290 |
| CPU 比较 | scikit-image 0.22.0、scipy 1.15.3、numpy 1.26.4，其余见 requirements/compare.txt |

Flash 来源为海光软件分发站，已用 wheel SHA-256：
`8546e46bcbfcd346757c67a0ff180ace2a49716d9724d52ee870d04b4c3b41d1`。
Triton wheel 应与平台实际 DTK/Torch/架构匹配，可从平台或海光软件分发站合法获取。
九齿环境独立创建，不在生成环境安装 ntops。

xmake 的 Apache 2.0 声明位于 tools/XMAKE-LICENSE.md；依赖定义的原始声明位于 dependencies/xmake-repo/。
其他组件保留各自声明，详见 THIRD_PARTY_NOTICES.md。模型权重和主办方原始输入不单独打包。
