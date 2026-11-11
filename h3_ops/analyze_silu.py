"""Generate the application report exclusively from completed raw results."""
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/"results"


def read(name):
    data=json.loads((RESULTS/name).read_text(encoding="utf-8"))
    if data["status"]!="PASS": raise RuntimeError(f"Incomplete result: {name}")
    return data


def main():
    nc=read("silu-native-correctness.json")
    qc=read("silu-nine-correctness.json")
    a,b,jit,plan=(read(f"silu-{name}.json") for name in
                  ("native-a","native-b","nine-jit","nine-plan"))
    maps=[{c["key"]:c for c in data["cases"]} for data in (a,b,jit,plan)]
    if any(set(m)!=set(maps[0]) for m in maps): raise RuntimeError("Case sets differ")
    rows=[]
    for c in a["cases"]:
        ca,cb,cj,cp=(m[c["key"]] for m in maps)
        base=statistics.median(ca["samples_us"]+cb["samples_us"])
        rows.append({"key":c["key"],"rows":c["rows"],"channels":c["channels"],
                     "native_a_us":ca["median_us"],"native_b_us":cb["median_us"],
                     "native_pooled_median_us":base,"nine_jit_us":cj["median_us"],
                     "nine_plan_us":cp["median_us"],"speedup":base/cp["median_us"],
                     "native_drift_percent":100*(cb["median_us"]/ca["median_us"]-1)})
    with (RESULTS/"silu-comparison.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    sources=("silu_nt.py","benchmark_silu.py","run_silu.sh","analyze_silu.py")
    max_error=max(c["max_abs_vs_native"] for c in qc["cases"])
    faster=sum(r["speedup"]>1 for r in rows)
    summary={"status":"SILU_FP32_VALIDATED_H3_END_TO_END_NOT_RUN",
             "correctness_native":len(nc["cases"]),"correctness_nine":len(qc["cases"]),
             "max_abs_vs_native":max_error,"benchmark_cases":len(rows),"faster_cases":faster,
             "comparison":rows,"source_sha256":{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                                  for name in sources}}
    (RESULTS/"silu-summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    team=json.loads((ROOT/"team.json").read_text(encoding="utf-8")) if (ROOT/"team.json").exists() else {}
    table="\n".join(f"| {r['rows']} | {r['channels']} | {r['native_a_us']:.2f} | "
                    f"{r['native_b_us']:.2f} | {r['nine_jit_us']:.2f} | {r['nine_plan_us']:.2f} | "
                    f"{r['speedup']:.2f}× |" for r in rows)
    report=f"""# MiniMax-H3 FP32 SiLU 九齿优化与 8 卡资源申请报告

日期：{datetime.now().isoformat(timespec='seconds')}。成果范围：H3 真实算子的独立验证及性能优化。完整 H3 FL2VA 尚未运行，本算子尚未接入 C++ 模型正式调用路径。

## 1. 队伍信息与申请目的

- 队伍：{team.get('team_name','待填写')}。
- 队长：{team.get('leader','待填写')}。
- 联系邮箱：{team.get('email','待填写')}。
- 申请 8 卡资源，按官方指定模型、配置和样例完成 FL2VA、多卡与端到端验证。目标卡型、显存和通信条件以主办方分配为准。

## 2. 算子来源及环境

InfiniLM 固定提交 `65deb91f840ebaef752d028fad2f3217ce0950f4` 的 `MiniMaxH3TimeEmbedder` 两层 Linear 显式使用 FP32，中间调用 `infinicore::op::silu`。位置：`csrc/models/minimax_h3/minimax_h3_transformer.cpp:350`（构造），`:383`（调用）。本实验保持该处实际 FP32 精度，**没有将 BF16 模型临时改为 FP32 来构造结果**。

[官方 FL2VA Transformer 配置](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/FL2VA/transformer/config.json) 的 `time_embed_hidden_size=5376`。实验以 5376 为结构相关宽度，257 仅用于尾部掩码验证；行数为构造覆盖，没有完整视频形状跟踪数据。

基线为指定 InfiniCore `minimax-h3` 提交 `f32e2fbb2a9b2ca912b5711f4f3836c403196a74` 的原生 SiLU，通过 `infinicore.nn.functional.silu` 调用原生 C++ 算子。原生实现本身已将 SiLU 数值计算放在算子内，不将它描述为未融合的多内核实现。未修改基线源码和评测程序。

实际环境：单卡 Pod，7 CPU 核、60 GiB 内存，设备 {plan['device']}，gfx906，约 15.98 GiB 可见显存。Python 3.11.9，Torch {plan['torch']}，HIP 6.3.26093，NineToothed 0.26.0，Triton 3.8.0，xmake 3.0.5。原生与九齿采用独立 venv/进程，避免 ntops 自动切换原生分派。

## 3. 实现与优化

九齿 arrangement 按行与通道分块，block=1024、num_warps=4、num_stages=1；application 表达 `x * (1 / (1 + exp(-x)))`，输入输出均为 FP32，使用验证通过的编译器完整 `exp` 数学内建。原生 FP32 源码使用 `__expf`；九齿快速内建候选在本平台未通过验证，因此未采用。边界掩码和设备代码由九齿编译链生成，无手写 Triton、CUDA 或 HIP 数值内核。

除普通九齿 JIT 调用外，提供稳定缓冲区执行计划：准备阶段正常调用九齿生成器并完成设备指针验证，之后复用生成二进制、元数据和编译器提供的 C 宿主启动器。使用该启动器明确支持的整数指针参数，避免重复 data_ptr 调用、HIP 指针属性查询和临时宿主包装。每次仍重新读取输入并执行完整 SiLU；保留张量、流和生成二进制生命周期。输入内容可原地更新，更换存储、形状、精度、设备、流或插桩配置后须重建计划。

本实验的优化包含宿主执行路径，不能将全部收益归因于设备内核本体。C 启动器绑定仅用于本内核无需全局/性能分析临时空间、无协作启动且无插桩回调的情况，其余情况保留正常启动路径。准备过程单线程，依赖固定 NineToothed/Triton 版本的内部宿主接口，升级后需重新验证。

## 4. 正确性结果

原生 {len(nc['cases'])} 例、九齿 {len(qc['cases'])} 例均通过。覆盖行数 1/4/17，宽度 257/5376，输入标准差 1/10，并包含 -80、-20、-1、0、1、20、80。相同确定性输入下，分别对比 Torch SiLU 和原生输出，使用 atol=rtol=1e-5。

九齿对原生最大绝对误差：**{max_error:.9g}**。每例另原地改变输入内容并重新调用，确认结果随新输入变化；普通九齿 API 与执行计划结果一致。性能测试在计时前后也检查输出。

## 5. 性能结果与口径

按原生 A、普通九齿 JIT、九齿执行计划、原生 B 串行运行，避免 GPU 并发干扰；各后端统一将 Torch CPU 及 interop 线程设为 1，避免输入生成的线程池使 7 核 Pod 发生 CPU 配额抖动。每例预热 40 次，测量 {plan['repeats']} 轮，每轮 {plan['iterations']} 次。每轮起止均同步设备，以整轮墙钟时间除以调用次数。下表包含宿主分派与设备完成时间，**不是纯设备内核时间或视频端到端时间**。

输入输出分配、输入初始化、JIT 和计划准备在计时外，准备耗时单独记录；原生张量包装也在计时外。原生合并中位数由 A/B 共 {2*plan['repeats']} 个批量样本计算，加速比为其除以九齿计划中位数。全部轮次、A/B 漂移和慢于基线的情况均保留。

单位：微秒/次。

| 行数 | 通道 | 原生 A | 原生 B | 九齿 JIT | 九齿计划 | 相对原生 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{table}

{len(rows)} 组构造覆盖中，{faster} 组计划执行中位数快于原生合并中位数。行数 128/1024 用于规模覆盖，不能冒充 H3 实际调用频率。尚未测量完整 H3 中 SiLU 的耗时占比，不能从本表推导视频生成加速比。

## 6. 代码和复现

服务器目录：`/work/home/scnefxtr39/Projects/h3_ops`。正式代码为 `silu_nt.py`、`benchmark_silu.py`、`run_silu.sh`、`analyze_silu.py`。源文件 SHA256 记录在 `results/silu-summary.json`；逐轮原始结果为 `results/silu-*.json`，汇总为 `results/silu-comparison.csv`。

在已配置 Pod 中、设置持久化 WORK_HOME 后运行：

```bash
bash "$WORK_HOME/Projects/h3_ops/run_silu.sh" > "$WORK_HOME/logs/h3-silu-run.log" 2>&1
python "$WORK_HOME/Projects/h3_ops/analyze_silu.py"
```

`silu.status=COMPLETE` 表示整套算子验证和性能测试完成。九齿启动器 `run_ninetoothed.sh` 加载独立 venv，并通过实际 HIP 属性 API/平台 Torch 获取真实设备信息；没有伪造架构或属性。包中附基础环境版本、运行脚本及适配器源文件。模型权重、私钥和 Kubernetes 凭据不打包。

## 7. 失败案例、辅助组件与局限

前期时间步编码在时间步接近 1000 时出现约 5.6e-5 误差，部分元素不满足其 2e-5 容差；替换数学位码实验也未解决数值问题。低精度 SwiGLU 和 SiLU 快速指数内建候选中也观察到明显错误，部分情况依赖编译布局，根因尚未完全定位。这些代码和结果没有作为成功优化使用，也不在本 SiLU 推理路径中。

第三方辅助组件包括 Torch 的分配、视图、宿主元数据与测试参考，NineToothed 及其 Triton 编译链、DTK 工具链和 Python 标准库。参赛算子数值计算均由九齿表达产生，未转调第三方现成张量计算内核；未采用在线服务、历史结果复用、训练、微调、量化或权重替换。

当前成果仅支持连续二维 FP32 SiLU。正式 C++ 模型集成、多卡流依赖、完整 H3 加载、VAE、视频规格、相似度、峰值资源和端到端计时尚未完成。本次单算子计时不包含模型实际调度和动态缓冲区重建的成本。包是资源申请附件，不是完整参赛作品。

## 8. 8 卡阶段计划与提交

获得资源后先核对设备/通信/权重及公开样例路径、统一生成参数和基础算子许可，跑通原生 FL2VA 并建立输出规格与相似度基线；再采集真实形状、阶段耗时与显存，完成九齿算子 C++ 调用、流依赖及生命周期验证，根据真实热点选择后续优化。保持规定步数、帧数、分辨率和模型精度。

按所提供 Pod 配置指南第 1 页，填写队伍信息并提交简要报告、代码位置和复现方式至 `duhang@qiyuanlab.com`，以申请 8 卡资源。**本次只准备材料，尚未代发邮件，资源尚未申请或获批。**
"""
    (ROOT/"8卡资源申请技术报告.md").write_text(report,encoding="utf-8")
    print(json.dumps({"correctness":len(qc["cases"]),"benchmarks":len(rows),"faster_cases":faster,
                      "speedup_min":min(r["speedup"] for r in rows),
                      "speedup_max":max(r["speedup"] for r in rows),
                      "max_abs_vs_native":max_error},indent=2))


if __name__=="__main__": main()
