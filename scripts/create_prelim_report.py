"""Populate the supplied preliminary-report template with verified evidence only."""

from hashlib import sha256
import argparse
import json
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt


ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--template", type=Path, default=ROOT / "【嵩山杯】Minimax初赛阶段报告-队伍名（模板）.docx")
parser.add_argument("--output", type=Path, default=ROOT / "【嵩山杯】Minimax初赛阶段报告-究竟对不队（草稿）.docx")
parser.add_argument("--submission", action="store_true", help="Use formal technical-report wording while retaining measured limitations")
args = parser.parse_args()
TEMPLATE, OUTPUT = args.template.resolve(), args.output.resolve()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def set_text(paragraph, text, size=10.5, mono=False):
    paragraph.text = text
    for run in paragraph.runs:
        run.font.size = Pt(size)
        run.font.name = "Consolas" if mono else "Times New Roman"
        run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
    paragraph.paragraph_format.space_after = Pt(5)


def table_style(table):
    if "Table Grid" in doc.styles:
        table.style = "Table Grid"
    properties = table._tbl.tblPr
    borders = properties.find(qn("w:tblBorders"))
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        properties.insert_element_before(
            borders, "w:shd", "w:tblLayout", "w:tblCellMar", "w:tblLook",
            "w:tblCaption", "w:tblDescription", "w:tblPrChange",
        )
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        border = OxmlElement(f"w:{edge}")
        for key, value in (("val", "single"), ("sz", "4"), ("color", "D9D9D9")):
            border.set(qn(f"w:{key}"), value)
        borders.append(border)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_before = Pt(2)
                paragraph.paragraph_format.space_after = Pt(2)
                for run in paragraph.runs:
                    run.font.size = Pt(9)
                    run._element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "宋体")
        properties = row._tr.get_or_add_trPr()
        properties.append(OxmlElement("w:cantSplit"))
    for cell in table.rows[0].cells:
        shading = OxmlElement("w:shd")
        shading.set(qn("w:val"), "clear")
        shading.set(qn("w:fill"), "F2F2F2")
        cell._tc.get_or_add_tcPr().append(shading)
        for run in cell.paragraphs[0].runs:
            run.bold = True


def add_heading(text):
    paragraph = doc.add_paragraph()
    set_text(paragraph, text, 14)
    paragraph.runs[0].bold = True
    paragraph.paragraph_format.keep_with_next = True
    level = OxmlElement("w:outlineLvl")
    level.set(qn("w:val"), "0")
    paragraph._p.get_or_add_pPr().append(level)
    return paragraph


original = TEMPLATE.read_bytes()
doc = Document(TEMPLATE)
replacements = {
    "【嵩山杯】Minimax初赛阶段报告-队伍名（模板）":
        "【嵩山杯】MiniMax 初赛阶段报告\n究竟对不队（工作草稿）",
    "队伍名：": "队伍名：究竟对不队",
    "仓库地址：": "仓库地址：拟发布至 https://github.com/shiqigeng1/songshan-minimax-h3；当前仅本地准备，仓库尚未创建或上传。",
    "分支：": "分支：拟用 main，最终提交分支待冻结。",
    "commit：": "commit：待冻结最终提交 commit；上游基线版本另见补充说明。",
    "第三方依赖（如有）：":
        "第三方依赖（如有）：平台海光 PyTorch 2.9.0、HIP 6.3.26093、DTK 26.04；"
        "Diffusers 0.40.0、Transformers 5.14.1、Accelerate 1.13.0、PyAV 16.0.1；"
        "海光 FlashAttention 2.6.1+das.opt1.dtk2604.torch290 用于 PyTorch 参考路径；"
        "NineToothed 0.26.0 与海光 Triton 3.5.1 用于独立算子实验。"
        "scikit-image 0.22.0、SciPy 1.15.3 等用于独立 CPU 视频比较环境。"
        "各类依赖及离线归档摘要随代码清单提供，附件发布位置与再分发声明仍待完成。",
    "简述实现了哪些功能":
        "已在比赛八卡 DCU 环境以 InfiniLM FL2VA 入口打通加载、条件处理调用、"
        "迭代去噪、视频/音频解码调用及媒体文件输出。统一首帧为本模板原始 PNG，"
        "提示词为 A dog falls asleep，seed 42，124 帧，steps 参数为 50。"
        "当前安装版本已生成可读取的 768×928、24 fps、H.264/AAC 视频并通过规格检查。"
        "PyTorch 参考视频已生成，原始比较得到当前原生视频平均 SSIM 8.997%。"
        "画面核验发现原生输出为棋盘纹理，未保持小狗内容，因此模型生成正确性尚未完成适配，"
        "不能将规格通过写为初赛相似度通过。",
    "哪些还在继续改进":
        "首先定位并修复原生视频棋盘纹理及低相似度。已完成 22 项单卡基础数值检查和真实 DiT 权重短序列对照，"
        "下一步对照 Qwen3-VL 条件特征，并继续核验正式长序列。待完成九齿门控残差在 C++ 模型路径中的"
        "真实接入、源代码与二进制对应记录及端到端 A/B；待整理公开仓库与最终 commit，"
        "并在干净环境复核离线构建。每卡峰值显存、连续多用例和稳态重复结果亦待补。",
}
for paragraph in doc.paragraphs:
    text = paragraph.text.strip()
    if text in replacements:
        set_text(paragraph, replacements[text], 17 if text.startswith("【嵩山杯】") else 10.5)

reproduction = r'''# 工作草稿：以下是已有 AFS 安装环境的运行命令。
# 从源码离线构建到安装的最终完整指令仍待补齐与重建验证。
export SETUPTOOLS_USE_DISTUTILS=stdlib
source "${IFAB_AFS_HOME:?}/minimax-h3/env_gfx936.sh"
mkdir -p "$WORK_HOME/outputs" "$WORK_HOME/logs"
run="fl2va-submit-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PYTHON" -u "$WORK_HOME/run_fl2va_vae_math.py" \
  "$WORK_HOME/runtime-gfx936/examples/minimax_h3.py" fl2va \
  --model-path /public/home/aclsylqidf/shared/MiniMax-H3/diffusers \
  --first-frame "$IFAB_AFS_HOME/songshan-fl2va-first-frame.png" \
  --text "A dog falls asleep" \
  --gpus 0,1,2,3,4,5,6,7 \
  --num-frames 124 --steps 50 --seed 42 \
  --output "$WORK_HOME/outputs/$run.mp4" \
  2>&1 | tee "$WORK_HOME/logs/$run.log"
# 上述启动器只在视频 VAE 解码范围选用 MATH SDPA。
# 相同参数加 --legacy 生成 PyTorch 参考。
# CPU 比较执行原始 minimax_h3_compare.py --tasks fl2va。
# 离线依赖与全部比较参数见 scripts/compare_existing.sh 和 run_compare_offline.sh。
'''
set_text(doc.tables[0].cell(0, 0).paragraphs[0], reproduction.strip(), 8, mono=True)
doc.tables[0].cell(0, 0).paragraphs[0].paragraph_format.space_after = Pt(0)

result = """原始比较脚本输出（2026-11-11，现有 MP4 比较）：
[result] MiniMax-H3 video comparison
  FL2VA
    Size:         PASS (PyTorch 768x928, InfiniLM 768x928)
    Length:       PASS (PyTorch 124 frames, InfiniLM 124 frames)
    Average SSIM: 8.997%

旧原生视频对相同参考：Size PASS，Length PASS，Average SSIM: 7.326%。
脚本没有给出 SSIM 合格阈值；以上不是初赛质量通过标记。
人工核验：参考视频显示小狗，两个原生视频抽样帧均为棋盘纹。

当前原生失败生成的工程记录（不作为质量合格的性能基线）：
[timing] compute.h3_transformer 410.703 s (49 calls, 8.382 s/call)
[timing] generation.pipeline_total 461.177 s
[timing] total.end_to_end 513.209 s
[spec] 768x928, 124 frames, 24 fps
[codec] h264 | audio: ['aac']
[spec] PASS

AFS 工作目录：$IFAB_AFS_HOME/minimax-h3
视频：outputs/fl2va-candidate-20261111T085622Z.mp4
日志：logs/fl2va-candidate-20261111T085622Z.log
参考：outputs/fl2va-reference-20261111T105437Z.mp4
比较：checks/fl2va-compare-20261111T133733Z-1034319/
"""
set_text(doc.tables[1].cell(0, 0).paragraphs[0], result.strip(), 8.5, mono=True)

add_heading("补充：适配与优化证据")
doc.add_paragraph(
    "平台原生适配使用海光 MHA 构建/注册兼容修改，并通过现有入口适配层在视频 VAE 解码范围选择 MATH SDPA。"
    "神经网络 VAE 解码属于模型计算，报告中不将其列为普通媒体编解码例外。"
    "最终代码审计、许可基础能力范围核对及兼容改动的完整回归材料待归档。"
)
doc.add_paragraph(
    "九齿门控残差表达式为 residual + gate * branch。实现显式保留乘法结果先舍入到 BF16 的语义。"
    "已测输入为构造数据，形状和门控步幅对应统一样例；两组均逐元素一致。"
    "内核源码为 h3_ops/gated_residual_nt.py。真实 C++ 模型接入与整体收益仍待完成。"
)
table = doc.add_table(rows=1, cols=5)
for cell, value in zip(table.rows[0].cells, ["行×列", "门控步幅", "原生 ms", "九齿 ms", "加速比"]):
    cell.text = value
for values in [
    ["512×5376", "(32256,1)", "0.1007", "0.0768", "1.312×"],
    ["27570×5376", "(32256,1)", "4.5841", "1.2285", "3.732×"],
]:
    for cell, value in zip(table.add_row().cells, values):
        cell.text = value
table_style(table)
doc.add_paragraph(
    "上述算子计时包含 Python 调用开销，是单卡同步批量调用的中位数。"
    "3.732× 不代表端到端加速。仅改成 at::addcmul 的实验也不能使用此数据作为九齿模型接入证据。"
)

table = doc.add_table(rows=1, cols=3)
for cell, value in zip(table.rows[0].cells, ["项目", "11月9日原生失败运行", "11月11日原生失败运行"]):
    cell.text = value
for values in [
    ["DiT 耗时", "410.693 s", "410.703 s"],
    ["生成阶段", "460.430 s", "461.177 s"],
    ["端到端", "498.614 s", "513.209 s"],
    ["规格", "768×928 / 124帧 / 24fps", "768×928 / 124帧 / 24fps"],
]:
    for cell, value in zip(table.add_row().cells, values):
        cell.text = value
table_style(table)
doc.add_paragraph(
    "两次数据来自不同会话的单次运行，目前没有建立整模型加速结论。"
    "两个原生视频均有棋盘纹质量问题，以上不能作为保持生成效果前提下的性能收益。"
    "PyTorch 参考单次端到端 742.279 s、DiT 530.621 s；跨后端、跨会话结果不构成优化 A/B。"
    "具体 C++ 修改与运行二进制的对应记录仍需补齐。"
    "整行缩放加 Softmax 的六组九齿实验全部慢于原生；保留其负结果作为选型依据。"
)

add_heading("补充：正确性诊断进度")
doc.add_paragraph(
    "基础探针已完成 22 项单卡检查，覆盖 FP32/BF16 数据桥接、linear、RMSNorm、索引操作和短序列 MHA，"
    "均满足各项诊断容差。它们使用构造数据和显式同步，不代表完整模型通过。"
    "真实且不修改的 DiT 权重在固定长度 104 输入下，对参考的视频相对 RMSE 为 TP2 0.8245%、"
    "TP8 0.9676%，音频为 2.4056%、2.5540%。这是张量误差，不是视频 SSIM。"
    "重复前向仍有变化，正式长度 27570、条件编码和完整迭代仍需排查。"
    "条件编码配对诊断脚本已准备，设备执行结果待回传。"
)
doc.add_paragraph(
    "已加载 _infinilm 扩展 SHA-256：eaf688defefdf6c20bf755c0d32ce1499e0d964d900393adfb41a5a56306c0d0。"
    "本地固定版本源码与 AFS 当前修改版的对应关系尚未冻结，最终提交前须导出实际源码、补丁、"
    "已加载扩展摘要和构建记录并重新核验。"
)

add_heading("补充：复现来源与待补事项")
for text in [
    "上游基线：InfiniCore f32e2fbb2a9b2ca912b5711f4f3836c403196a74；InfiniLM 65deb91f840ebaef752d028fad2f3217ce0950f4。以上为上游来源，尚非队伍最终提交 commit。",
    "固定依赖来源还包括 xmake 3.0.5、Boost 1.89.0、pybind11 3.0.1。离线构建、运行、九齿及比较归档均已有本地版本和 SHA-256 记录，拟在 GitHub Release 单独组织；尚未发布。固定模型权重和出题方数据按赛题要求不重复打包。",
    "待补：修复棋盘纹并完成生成质量验收、导出 AFS 实际源码与最终构建对应关系、发布仓库和最终分支/commit、干净环境离线重建、九齿模型真实接入及 A/B、资源与连续运行记录。成员和联系电话按队伍要求暂留空。",
    "此文档是依据官方模板生成的工作草稿。尚未满足全部待补项，不作为最终提交版，也不填写未经实测的 SSIM、整模型加速比或最终测评结论。",
]:
    doc.add_paragraph(text)

if args.submission:
    formal = {
        "【嵩山杯】MiniMax 初赛阶段报告\n究竟对不队（工作草稿）":
            "【嵩山杯】MiniMax 初赛阶段报告\n究竟对不队",
        "仓库地址：拟发布至 https://github.com/shiqigeng1/songshan-minimax-h3；当前仅本地准备，仓库尚未创建或上传。":
            "仓库地址：https://github.com/shiqigeng1/songshan-minimax-h3",
        "分支：拟用 main，最终提交分支待冻结。": "分支：main",
        "commit：待冻结最终提交 commit；上游基线版本另见补充说明。":
            "commit：以固定版本标签 prelim-20261111 对应的提交为准。",
        "各类依赖及离线归档摘要随代码清单提供，附件发布位置与再分发声明仍待完成。":
            "源码、补丁、依赖说明、运行脚本和实验记录在代码仓库中按目录提供。",
        "不能将规格通过写为初赛相似度通过。":
            "规格检查通过，生成内容一致性仍存在问题。",
        "下一步对照 Qwen3-VL 条件特征，并继续核验正式长序列。":
            "条件编码与正式长序列的一致性是后续排查方向。",
        "待整理公开仓库与最终 commit，": "版本来源见代码仓库，",
        "脚本没有给出 SSIM 合格阈值；以上不是初赛质量通过标记。":
            "对照结论：尺寸和帧数一致，原生输出与参考的内容相似度较低。",
        "当前原生失败生成的工程记录（不作为质量合格的性能基线）":
            "当前原生运行记录（画面存在棋盘纹异常）",
        "11月9日原生失败运行": "11月9日原生运行",
        "11月11日原生失败运行": "11月11日原生运行",
        "3.732× 不代表端到端加速。仅改成 at::addcmul 的实验也不能使用此数据作为九齿模型接入证据。":
            "3.732× 对应独立算子测试，九齿 C++ 模型路径接入及端到端收益尚未验证。",
        "补充：复现来源与待补事项": "补充：版本来源与已知限制",
        "以上为上游来源，尚非队伍最终提交 commit。":
            "以上标识固定上游来源；本作品版本由仓库提交标签标识。",
        "离线构建、运行、九齿及比较归档均已有本地版本和 SHA-256 记录，拟在 GitHub Release 单独组织；尚未发布。":
            "代码以展开的目录结构提供，依赖版本、安装方式和摘要随仓库文档记录。",
        "待补：修复棋盘纹并完成生成质量验收、导出 AFS 实际源码与最终构建对应关系、发布仓库和最终分支/commit、干净环境离线重建、九齿模型真实接入及 A/B、资源与连续运行记录。成员和联系电话按队伍要求暂留空。":
            "已知限制：原生输出存在棋盘纹异常；源码与已运行二进制的完整对应、干净环境重建、九齿模型接入、端到端 A/B、峰值资源及连续多用例验证尚未完成。",
        "此文档是依据官方模板生成的工作草稿。尚未满足全部待补项，不作为最终提交版，也不填写未经实测的 SSIM、整模型加速比或最终测评结论。":
            "适用范围：比赛平台海光 gfx936 单机八卡、公开小狗首帧和 FL2VA 固定配置。实验结果按独立算子、模型诊断和完整视频生成分别报告。",
        "CPU 离线比较": "CPU 视频比较",
    }
    for paragraph in list(doc.paragraphs):
        if "仓库需要为" in paragraph.text and "确认提交" in paragraph.text:
            paragraph._element.getparent().remove(paragraph._element)
    paragraphs = list(doc.paragraphs)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                paragraphs.extend(cell.paragraphs)
    for paragraph in paragraphs:
        old = paragraph.text
        new = old
        for before, after in formal.items():
            new = new.replace(before, after)
        if new != old:
            run = paragraph.runs[0] if paragraph.runs else None
            size = run.font.size.pt if run is not None and run.font.size else 10.5
            bold = run.bold if run is not None else None
            mono = run is not None and run.font.name == "Consolas"
            set_text(paragraph, new, size, mono)
            if paragraph.runs:
                paragraph.runs[0].bold = bold
    for paragraph in doc.paragraphs:
        if paragraph.text.startswith("首先定位并修复原生视频"):
            set_text(paragraph,
                "后续改进集中于原生视频棋盘纹异常、条件编码与长序列计算的一致性，"
                "以及九齿门控残差在 C++ 推理路径中的集成。"
                "源码与二进制对应核验、干净环境构建、每卡峰值显存、连续多用例和稳态重复性能记录尚不完整。")
        elif paragraph.text.startswith("基础探针已完成 22 项"):
            set_text(paragraph, paragraph.text.replace(
                "条件编码配对诊断脚本已准备，设备执行结果待回传。",
                "条件编码配对诊断已有实现，尚未获得设备对照结果。"))
        elif paragraph.text.startswith("已加载 _infinilm 扩展 SHA-256"):
            set_text(paragraph,
                "运行扩展 _infinilm 的 SHA-256 为 "
                "eaf688defefdf6c20bf755c0d32ce1499e0d964d900393adfb41a5a56306c0d0。"
                "该指纹标识本报告中已测运行的软件版本。固定源码快照与运行环境全部修改的对应关系尚未完成核验。")
    formal_commands = r'''# 环境：Linux x86-64、DTK 26.04、海光 Torch 2.9.0、8 张 gfx936
git clone https://github.com/shiqigeng1/songshan-minimax-h3.git
cd songshan-minimax-h3
# 安装目录由使用者指定；AFS 仅作为平台持久目录选项。
export H3_WORK_HOME="${IFAB_AFS_HOME:-$HOME}/songshan-h3"
bash scripts/prepare_env.sh runtime
BUILD_JOBS=32 bash scripts/build_native.sh core
BUILD_JOBS=32 bash scripts/build_native.sh lm
# 首帧由主办方提供，不随工程打包；可用 H3_FIRST_FRAME 指定实际位置。
first_frame="${H3_FIRST_FRAME:-${IFAB_AFS_HOME:-$HOME}/songshan-fl2va-first-frame.png}"
bash scripts/run_fl2va.sh \
  --model-path /public/home/aclsylqidf/shared/MiniMax-H3/diffusers \
  --first-frame "$first_frame" --text "A dog falls asleep" \
  --gpus 0,1,2,3,4,5,6,7 --num-frames 124 --steps 50 --seed 42 \
  --output "$H3_WORK_HOME/outputs/fl2va.mp4"
# 相同参数加 --legacy 生成 PyTorch 参考，输出改为 reference.mp4。
# CPU 比较；依赖与原始程序调用参数见仓库说明。
bash scripts/prepare_env.sh compare
bash scripts/compare_existing.sh \
  "$H3_WORK_HOME/outputs/reference.mp4" "$H3_WORK_HOME/outputs/fl2va.mp4"
# 新整理的独立构建流程尚未在干净比赛环境中验证。
'''
    set_text(doc.tables[0].cell(0, 0).paragraphs[0], formal_commands.strip(), 8, mono=True)
    doc.tables[0].cell(0, 0).paragraphs[0].paragraph_format.space_after = Pt(0)
    all_text = "\n".join(p.text for p in doc.paragraphs)
    assert "工作草稿" not in all_text and "不作为最终提交版" not in all_text
    # Template instructions can be nested in a content control rather than a
    # direct body paragraph. Remove the authoring hint while keeping the fields.
    for element in list(doc.element.iter(qn("w:p"))):
        value = "".join(t.text or "" for t in element.iter(qn("w:t")))
        if "仓库需要为" in value and "分支" in value and "commit" in value:
            element.getparent().remove(element)
        elif value.strip().startswith("统一输入图片"):
            properties = element.find(qn("w:pPr"))
            if properties is None:
                properties = OxmlElement("w:pPr")
                element.insert(0, properties)
            properties.get_or_add_keepNext().val = True
    doc.tables[1].rows[0]._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))

repairs = {"shading_default": 0, "numbering_level_index": 0}
for paragraph in doc.paragraphs:
    if paragraph.text.strip() in {"团队信息", "代码：", "代码:", "完成情况", "复现指令", "结果"}:
        paragraph.paragraph_format.keep_with_next = True
    if len(paragraph.text) < 650:
        paragraph.paragraph_format.keep_together = True
for shading in doc.element.iter(qn("w:shd")):
    if shading.get(qn("w:val")) is None:
        shading.set(qn("w:val"), "clear")
        repairs["shading_default"] += 1
for abstract in doc.part.numbering_part.element.findall(qn("w:abstractNum")):
    for index, level in enumerate(abstract.findall(qn("w:lvl"))):
        if level.get(qn("w:ilvl")) is None:
            level.set(qn("w:ilvl"), str(index))
            repairs["numbering_level_index"] += 1
doc.save(OUTPUT)
assert TEMPLATE.read_bytes() == original, "Template changed"
loaded = Document(OUTPUT)
texts = [p.text for p in loaded.paragraphs]
assert "队伍名：究竟对不队" in texts
assert "成员：" in texts
with ZipFile(TEMPLATE) as source_zip, ZipFile(OUTPUT) as output_zip:
    assert output_zip.testzip() is None
    for name in source_zip.namelist():
        if name.startswith("word/media/"):
            assert source_zip.read(name) == output_zip.read(name), "Template image changed"
manifest = {
    "status": "SUBMISSION_WITH_KNOWN_LIMITATIONS" if args.submission else "WORKING_DRAFT_NOT_READY_TO_SUBMIT",
    "team": "究竟对不队",
    "members": None,
    "phone": None,
    "github_owner": "shiqigeng1",
    "public_repository": "https://github.com/shiqigeng1/songshan-minimax-h3" if args.submission else None,
    "planned_repository": "https://github.com/shiqigeng1/songshan-minimax-h3",
    "repository_created": bool(args.submission),
    "submission_tag": "prelim-20261111" if args.submission else None,
    "reference_ssim_percent": 8.997,
    "earlier_native_ssim_percent": 7.326,
    "model_quality_validated": False,
    "known_failure": "Native videos show checkerboard texture; first-frame content not preserved",
    "template_sha256": sha256(original).hexdigest(),
    "report_sha256": sha256(OUTPUT.read_bytes()).hexdigest(),
    "template_media_preserved": True,
    "template_compatibility_repairs": repairs,
    "visual_render_review": "pending",
}
OUTPUT.with_suffix(".draft.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUTPUT)
print("Submission report generated; template and original media unchanged." if args.submission else
      "Draft generated; template and original media unchanged.")
