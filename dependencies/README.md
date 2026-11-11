# 构建依赖定义

xmake-repo/ 只保留 Boost、pybind11 以及其系统工具依赖 Python、CMake 的固定来源定义。
Boost 的 CMake 构建范围限定为 filesystem 与 stacktrace。
这些是展开的包定义源码，依赖实际安装文件不包含在仓库内。
首次联网获取或通过使用者提供的本地缓存安装，路径不绑定个人存储账号。
