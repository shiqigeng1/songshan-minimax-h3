# 第三方来源与许可声明

本仓库含不同来源的代码，各组件保留自己的版权和许可，不统一改写为同一种许可证。

- InfiniCore 的 MIT 声明位于 upstream/InfiniCore/LICENSE。
- InfiniLM 的固定来源与提交记录见 docs/SOURCE_PROVENANCE.md；该来源快照未包含顶层 LICENSE，保留原有文件和第三方声明。
- InfiniCore/InfiniLM 的 third_party 代码属于上游依赖，不计为队伍新增贡献。
- xmake 3.0.5 的官方 Apache 2.0 声明位于 tools/XMAKE-LICENSE.md。
- xmake 依赖定义保留 dependencies/xmake-repo/LICENSE.md 与 NOTICE.md，Boost 定义有明确的队伍构建范围调整。
- Python、NineToothed、Triton、FlashAttention 等运行软件的声明随其原始分发包提供，本仓库不重复分发其 wheel。
- 报告保留官方模板的统一图片；工程没有额外的模型权重或输入数据包。

实验记录区分独立算子与实际模型运行，生成失败样例保留其内容和对照结果。
