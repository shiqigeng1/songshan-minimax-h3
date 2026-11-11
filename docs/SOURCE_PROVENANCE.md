# 版本与源码来源

仓库固定标签：`prelim-20261111`；分支：`main`。

| 项目 | 上游来源 |
| --- | --- |
| InfiniCore | minimax-h3 / f32e2fbb2a9b2ca912b5711f4f3836c403196a74 |
| InfiniLM | minimax_h3 / 65deb91f840ebaef752d028fad2f3217ce0950f4 |
| xmake-repo | 4b60b70e44cf1908dd367d019cf7471eb7362dd8 |

upstream/ 中的固定源码与原始来源归档逐文件核对，完整第三方源码已展开。
海光 MHA 修改单独保存在 patches/gfx936-hygon-mha-aten.patch，构建时应用于临时副本。
Boost 的依赖定义限制构建 filesystem 与 stacktrace；构建副本固定 pybind11 版本。原始比较程序没有修改。

历史运行扩展 SHA-256：`eaf688defefdf6c20bf755c0d32ce1499e0d964d900393adfb41a5a56306c0d0`。
历史桥接 Python SHA-256：`5e5fc07bf0c02925c553603285fd163c44c973eb6c3de71349cf6c7677e3e4e5`。
桥接文件已与本地固定版本核对一致。

历史运行环境全部修改与已加载二进制尚未完成逐项对应核验，不能保证本次固定来源加补丁重建后与历史扩展字节一致。
运行环境曾出现的 at::addcmul 修改没有作为已经验证的源码改动加入本仓库。
真实 DiT 短序列及完整视频记录均保留其原始软件指纹与范围限制。

公开日志中的个人工作目录替换为 `<H3_WORK_HOME>`，其他原始输出和数值保留。
