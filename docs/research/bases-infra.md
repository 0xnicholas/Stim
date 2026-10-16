# 基座与部署基建的现役选项(约 0.5B–30B 基座 / 训练算力 / 服务目标)

> 研究票:[#7](https://github.com/0xnicholas/Stim/issues/7) · 上游地图 [#1](https://github.com/0xnicholas/Stim/issues/1)
> 用途:供决策票 **#9(基座抉择)**、**#14(服务栈与部署路径)**、**#3(算力与预算的边界)** 使用。**本文件只做取数与盘点,不含「Stim 应该选哪个」的建议。**
> 主撰写日期:2026-10-16(抓取日)。「现役」= 抓取日仍可取到、有官方文档或一手证据支持的选项。
> 相邻研究(勿重复引用本文覆盖之外的内容):`research/kev-recipe:docs/research/kev-recipe.md`(Kev 四个 checkpoint 的基座、配方、Modal 用量与单价)、`research/ecosystem:docs/research/ecosystem.md`(Jev / AutoJev / Decision Index 生态)、`research/system-one-protocol:docs/research/system-one-protocol.md`(System One 线协议)。

---

## 0. 摘要(先读这段)

1. **Jev 式头的基座已经收敛到 Qwen3.5/3.6/3.8 的混合线性注意力家族。** 三代的 27B 与 35B-A3B 是同一套形状(64 层 = 48 Gated DeltaNet + 16 full;40 层 = 30 + 10、256 专家 top-8),`0.8B/2B/4B/9B/35B-A3B` 有官方 `-Base` 预训练权重,**27B 各代都没有 `-Base`**(只有后训练版),这是选型时最硬的一条可得性边界。([Qwen3.5-9B-Base 卡](https://hf-mirror.com/Qwen/Qwen3.5-9B-Base/raw/main/README.md)、各仓库 `config.json`)
2. **竞争者在同一批基座上已经跑出 Jev 式家族,且全部 Apache-2.0:** Cloudflare `clef`(Qwen3.8-27B + joint schema head)与 `clef-flash`(Qwen3.5-9B)、vLLM 官方示例侧的 `vllm-sr/d3-mini`(Qwen3.5-4B)、OptiQ 的 MLX 决策模型(Qwen3.5-0.8B、gemma-4-E2B)。**基座这一层不再是空白,空白在训练数据与配方。**(各模型卡)
3. **服务栈已经原生实现了 System One。** vLLM 内置 `POST /v1/systemone`,并在文档里把这类模型正式命名为 **Structured Decisions Models (aka JEV Models)**;同时提供 OpenAI `POST /v1/decisions` 的 MVP 实现。基座走的是 `NextTokenStrategy`(读标签 token 的 logprob),与 Kev 的 pointer head 是两条不同的读法。(vLLM 文档)
4. **Modal 的现役价目(按秒计费)** 与 kev 仓库自带的价目完全一致:H100 SXM5 $3.95/h、H200 SXM $4.54/h、B200 $6.25/h、B300 $7.10/h、L40S $1.95/h、L4 $0.80/h、T4 $0.59/h;Starter 计划 **$30/月免费额度**、Team **$100/月**;GPU 需要绑定支付方式;区域选择 ×1.15–1.75,**非抢占执行 ×3**。(modal.com/pricing、Modal GPU 文档)
5. **成本量级(按需、单机时价,可直接对比):** Modal H200 $4.54 ≈ HF Jobs H200 $5.00 ≈ RunPod Serverless H200 $5.93 ≈ RunPod Pods H200 $5.29 ≈ Lambda 自营 B200 $6.69;ROCm 侧最便宜的自助选择是 Hot Aisle 的 MI300X $2.99/GPU·h(192GB HBM3)。(各厂商定价页)
6. **演示位是最受限的一环。** HF Spaces 的 ZeroGPU 是共享 RTX Pro 6000 Blackwell(large 48GB / xlarge 96GB),**只支持 Gradio SDK**;免费账号每日 5 分钟 GPU 额度、最多 2 个 ZeroGPU Space,PRO 40 分钟 / 10 个,超出按 $1 / 10 分钟计。自家 GPU Space 按小时计费(A100-80 $2.50/h、L40S $1.80/h),**H100 已于 2025-12 下线**。([ZeroGPU 文档](https://hf-mirror.com/docs/hub/spaces-zerogpu)、[Spaces GPU 文档](https://hf-mirror.com/docs/hub/spaces-gpus))
7. **四个会拖累「自训自调自部署」的依赖链已经现成存在:** ① 27B 档没有 `-Base`(起点只能是别人的后训练权重);② Qwen3.5 系 `-Base` 仓库是**带视觉塔的多模态包装**(`Qwen3_5ForConditionalGeneration`),Kev 的做法是丢弃 vision tower 与 LM head,这条路径有隐式耦合;③ Kev 的 fused/图路径建在 **flash-linear-attention 的 Triton 核 + CUDA graphs** 上,ROCm 明确退回 eager;④ MLX 侧只有 Apple Silicon 可用,且 9B 级以上的长文与 27B 在 Mac 上的实测仍缺(kev 自述「expected, not measured」)。(kev 源码与模型卡)

---

## 1. 方法与来源分级

| 等级 | 来源类型 | 本文件中的例子 |
|---|---|---|
| **一手** | 厂商定价页 / 官方文档 / 官方配置与模型卡 | `modal.com/pricing`、`modal.com/docs`、`runpod.io/pricing`、`lambda.ai/pricing`、`hotaisle.xyz/pricing`、`docs.vllm.ai`、`rocm.docs.amd.com`、`apple.com/mac-studio`、各 HF 仓库的 `config.json` 与模型卡 |
| **一手(经镜面)** | 同一份 HF 内容经 `hf-mirror.com` 代理读取(内容与 huggingface.co 同源,仅传输路径不同) | 全部 `hf-mirror.com/api/...`、`hf-mirror.com/<repo>/raw/...` 与 `hf-mirror.com/docs/...` |
| **一手(源码)** | 项目仓库源码/文档 | `jaredpalmer/kev`(`modal_app.py`、`kev/budget.py`、`kev/model.py`、`kev/fused_qwen35.py`、`kev/serve.py`、`docs/model-cards/`)、`ggml-org/llama.cpp`、`ml-explore/mlx-lm` |
| **未取回/不可达** | 抓取环境不可达或需要登录 | 见 §7 |

抓取环境实况(本日实测):`huggingface.co`、`raw.githubusercontent.com` **不可达**;`hf-mirror.com`(含其 `/docs` 镜像)、`modal.com`、`runpod.io`、`vast.ai`、`lambda.ai`、`hotaisle.xyz`、`docs.vllm.ai`、`rocm.docs.amd.com`、`pytorch.org`、`apple.com`、`github.com`/`api.github.com`(经 `gh` CLI)、`archerhume.com`、`docs.typesafe.ai`、`registry.npmjs.org`(本次未用)**可达**。HF 仓库的 `resolve/` 路径会 307 到 `api/resolve-cache`,本文件统一用 `/raw/` 或 `/api/`。

---

## 2. 基座候选(约 0.5B–30B,含 MoE)

### 2.1 情报表

「注意力构成」取自各仓库 `config.json`(经 `hf-mirror.com`);`linear` = Gated DeltaNet 类线性注意力层,`full` = 全注意力层;`sw` = 滑动窗口。上下文为 config 的 `max_position_embeddings`(声明值,非实测稳定值),另有出处则单独注明。

| 基座(HF id) | 规模 | 注意力构成 | 声明上下文 | `-Base` 版 | 许可 | MLX / 量化成熟度 |
|---|---|---|---|---|---|---|
| `Qwen/Qwen3.5-0.8B-Base` | 0.8B dense | 18 linear + 6 full(24 层) | 262,144 | ✅ | Apache-2.0 | 高:mlx-community MLX-8bit/OptiQ-4bit、GGUF(unsloth/ggml-org)、已有人做成 decision 模型 |
| `Qwen/Qwen3.5-2B-Base` | 2B dense | 18 + 6(24 层) | 262,144 | ✅ | Apache-2.0 | 中:OptiQ-4bit;GGUF |
| `Qwen/Qwen3.5-4B-Base` | 4B dense | 24 + 8(32 层) | 262,144 | ✅ | Apache-2.0 | 高:MLX-8bit、OptiQ-4bit、GGUF |
| `Qwen/Qwen3.5-9B-Base` | 9B dense | 24 + 8(32 层) | 262,144(卡面称「原生 262K,可扩到 1.01M」) | ✅ | Apache-2.0 | 高:MLX-8bit、OptiQ-4bit、GGUF |
| `Qwen/Qwen3.5-35B-A3B-Base` | 35B 总 / 3B 激活 MoE | 30 linear + 10 full(40 层),256 专家 top-8 | 262,144 | ✅ | Apache-2.0 | 中高:MLX-4bit、OptiQ-4bit、GGUF |
| `Qwen/Qwen3.5-27B` | 27B dense | 48 + 16(64 层) | 262,144 | ❌ **无 Base** | Apache-2.0 | 高:MLX-4bit、FP8/GPTQ-Int4 |
| `Qwen/Qwen3.6-27B` | 27B dense | 48 + 16(64 层) | 262,144 | ❌ | Apache-2.0 | 中:OptiQ-4bit |
| `Qwen/Qwen3.6-35B-A3B` | 35B / 3B MoE | 30 + 10,256 专家 top-8 | 262,144 | ❌ | Apache-2.0 | 中高:OptiQ-4bit(+REAP-19B)、MLX |
| `Qwen/Qwen3.8-27B` | 27B dense | 48 + 16(64 层) | 262,144 | ❌ | Apache-2.0 | 高:MLX-4bit、FP8、GPTQ-Int4;**Kev-27B 的基座** |
| `Qwen/Qwen3.8-Flash-Next` | MoE(`qwen4_exp_text`) | 36 + 12(48 层),512 专家 top-10 | 262,144 | ❌ | **other(非 Apache)** | 低 |
| `Qwen/Qwen3-4B-Base` / `-8B-Base` / `-30B-A3B-Base`(上一代) | 4B / 8B / 30B-A3B | 全注意力(无线性层) | 32,768 | ✅ | Apache-2.0 | 极高:GGUF/MLX/AWQ/GPTQ 全覆盖;vLLM 官方 `/v1/systemone` 示例就用 `Qwen/Qwen3-0.6B` |
| `google/gemma-4-12B` | 12B | 40 sw(1024)+ 8 full(48 层) | 262,144 | ✅(卡面明说「pre-trained 与 instruction-tuned 两种变体」) | 卡面 Apache-2.0(链接到 `ai.google.dev/.../gemma_4_license`;**仓库内无 LICENSE 文件**) | 高:lmstudio/mlx-community MLX-4/5/6bit、QAT w4a16 |
| `google/gemma-4-E4B` | 4B 有效 | 35 sw(512)+ 7 full(42 层) | 131,072 | ✅ | 同上 | 高:MLX-4bit(QAT) |
| `google/gemma-4-26B-A4B` | 26B / 4B MoE | 25 sw(1024)+ 5 full(30 层),128 专家 | 262,144 | ✅ | 同上 | 高:MLX-4bit(QAT) |
| `mistralai/Ministral-3-3B-Base-2512` / `-8B-Base-2512` / `-14B-*` | 3B / 8B / 14B | 全注意力(无滑窗) | 262,144 | ✅ | Apache-2.0 | 中:MLX 多量化只覆盖 `-Instruct` 版,`-Base` 需自行转换 |
| `mistralai/Mistral-Nemo-Base-2407` / `Mistral-Small-24B-Base-2501` | 12B / 24B | 全注意力 | 131,072 / 32,768 | ✅ | Apache-2.0 | 低(Nemo 有 bf16/量化) |
| `ibm-granite/granite-4.1-8b-base` / `granite-4.2-8b` / `-30b` | 8B / 8B / 30B | 全注意力 | 131,072 | ✅(4.1 有 `-base`;4.2 主推 post-trained) | Apache-2.0 | 低(MLX 覆盖集中在 granite-4.0 系) |
| `allenai/Olmo-3-1025-7B` / `Olmo-3.1-32B-*` | 7B / 32B | 24 sw(4096)+ 8 full / 48 + 16 | 65,536 | ✅(7B 为基座;32B 为 think/instruct) | Apache-2.0(数据全开源) | 低(mlx-community 有 OLMo-2 系,3.x 未覆盖) |
| `HuggingFaceTB/SmolLM3-3B-Base` | 3B | 全注意力 | 65,536 | ✅ | Apache-2.0 | 高:MLX-3/4/5/6/8bit 全覆盖 |
| `LiquidAI/LFM2.5-8B-A1B-Base` / `LFM2.5-24B-A2B-Base` | 8B / 1B、24B / 2B MoE | 18 conv + 6 full(24 层) | 128,000 | ✅ | **other(`lfm1.0` 专有许可,非 OSI)** | 高(1.2B 全量化;24B 由官方发 MLX) |
| `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B` | 30B / 3B MoE | `nemotron_h` 混合(52 层),128 路由专家 top-6 + 1 共享 | 262,144 | ✅(BF16 仓库) | **other(OpenMDW-1.1,非 OSI)** | 中:NVFP4 官方版、GGUF |
| `moonshotai/Moonlight-16B-A3B(-Base)` | 16B / 3B MoE | `deepseek_v3`(MLA + 64 专家 top-6 + 2 共享) | **8,192** | ✅ | MIT | 低 |
| `moonshotai/Kimi-Linear-48B-A3B-Base` | 48B / 3B(超 30B,列出作参考) | `kimi_linear`(KDA 线性注意力混合) | — | ✅ | MIT | 低 |
| `meta-llama/Llama-3.2-1B/3B`、`Llama-4-Scout-17B-16E` | 1B / 3B / 17B-16E | 全注意力 / MoE | 128K / 10M | ✅(非 `-Base` 命名,`Llama-3.2-1B` 即预训练) | **Llama 社区许可,非 OSI 开源** | 极高(社区量化最全) |

> 表外补充:`Qwen/Qwen3.5-122B-A10B`、`Qwen3.5-397B-A17B`、`Qwen3.8-2.4T-A95B` 超出本票范围;`zai-org/GLM-5.x`、`deepseek-ai/DeepSeek-V4` 多为 MIT 但最小体量也在 100B+。

### 2.2 与「Jev 式头」适配直接相关的四个硬性维度

**① tokenizer 与特殊 token 的余量。** Kev 的做法是**复用 `vocab` 里已有的冷门 token**(`<|fim_prefix|>` `<|fim_middle|>` `<|box_start|>` `<|box_end|>` `<|fim_suffix|>`)当 state / question / option / decide 的分隔符,**不新增 embedding 行**,靠 LoRA 改写它们的语义(`kev/model.py:9-11`)。这给出两条可迁移结论:
- 一个候选基座的 tokenizer 必须**有足够的未充分训练的保留 token**,否则就要改词表并重训 embedding(代价与风险都上一个量级)。
- Qwen3.5 的官方基座卡明确写着控制 token `<|im_start|>`/`<|im_end|>`「被训练成支持高效 LoRA 式 PEFT」([Qwen3.5-9B-Base 卡](https://hf-mirror.com/Qwen/Qwen3.5-9B-Base/raw/main/README.md))——这是与 Jev 式头契合的现成特性,其他家族没有同等表述。
- Qwen3.5 系词表 248,320(填充到),Qwen3 系 151,936;Gemma 4 为 262,144;Granite 100,352;OLMo 3 100,278;SmolLM3 128,256。**词表越大,预留特殊 token 的空间通常越充裕,但 embedding 层也越大。**

**② 注意力实现的长文可行性。** 只有 Qwen3.5/3.6/3.8、LFM2.5、Nemotron-3.5、Kimi-Linear、Gemma 4(滑窗混合)、OLMo 3(滑窗混合)属于「混合/线性注意力」;其余是全注意力,32K–131K 以上 prefill 的代价随长度平方增长。Kev 在 Qwen3.5 上把状态推到 **65,536 token 服务上限**,并在三个尺寸上实测出「8,192 token 内精度不降」([Kev README](https://github.com/jaredpalmer/kev))。

**③ 长上下文的训练代价。** Kev 的 `ROW_PASS_TOKENS = 16384`、`SERVE_MAX_STATE = 65536`、`SERVE_MAX_BRANCH = 73728`,而**训练状态的默认上限只有 384 token**(`kev/model.py:14-24`)。换句话说:基座能读 256K 与「配方能训 256K」是两件事,后者由显存与 pass 形状决定。

**④ 量化/MLX 路径的成熟度。** 判据是「该基座的**你要的那一版**(base / 你的 checkpoint)是否有现成转换路径」,而不是「同名模型有没有 4bit」。MLX 侧 `mlx-lm` 已内置 `qwen3_5.py`、`qwen3_5_moe.py`、`gemma4.py`、`ministral3.py`、`lfm2_moe.py`、`granitemoehybrid.py`、`olmo3.py`、`smollm3.py`、`nemotron_h.py`、`gated_delta.py`、`kimi_linear.py`、`falcon_h1.py` 等实现([mlx-lm 仓库](https://github.com/ml-explore/mlx-lm),抓取日 `pushed_at` = 2026-10-16T02:49Z),所以**自转 base 版在架构层面是支持的**;现成的 mlx-community 量化**几乎只覆盖 post-trained 版**。

### 2.3 逐族要点

**Qwen3.5 / 3.6 / 3.8(事实上的默认选择)**
- 三代共享同一套混合形状;`Qwen3.5-9B-Base` 卡给出隐藏布局 `8 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`,即 **`full_attention_interval = 4`**;`partial_rotary_factor = 0.25`(3.6/3.8 显式写明,3.5 的 `-Base` 卡给出 RoPE 维度 64 / head_dim 256 = 0.25)。
- `-Base` 仓库的 `architectures` 是 **`Qwen3_5ForConditionalGeneration`(多模态包装)**,`pipeline_tag` 为 `image-text-to-text`,自带 vision token(`image_token_id`)。Kev-27B 卡写明「vision tower、LM head 与 MTP 层都不加载」,即 **Jev 式头实际只用文本主干**。
- `Qwen3.5-35B-A3B-Base` 是现役代里唯一的**官方预训练 MoE**,256 专家 top-8、`moe_intermediate_size` 512、40 层、hidden 2048(上一代的 `Qwen3-30B-A3B-Base` 同为官方预训练 MoE,但上下文只有 32,768)。
- **27B 无 `-Base`(三代皆然)。** 检索 `Qwen` 组织下 `Qwen3.5-27B` 只有后训练版与 FP8/GPTQ 量化;旁证是 Qwen 自己的 SAE 发布里出现 `SAE-Res-Qwen3.5-9B-Base-*` 却只有 `SAE-Res-Qwen3.5-27B-*`(无 `Base` 字样)。**Kev-27B 因此站在 Qwen3.8-27B 的后训练权重上,且全参微调**(模型卡自述「we don't know what that was trained on」)。

**Gemma 4(许可最宽松的非 Qwen 家族)**
- 卡面:「本发布包含 **pre-trained 与 instruction-tuned 两种变体**的开放权重」;上下文「最高 256K」。E2B/E4B/12B 支持音频输入。
- 架构是**滑窗/全注意力混合**(12B:40×sw1024 + 8×full),不是线性注意力。
- **许可标注需要人工确认:** 卡面 `license: apache-2.0`,但 `license_link` 指向 `ai.google.dev/gemma/docs/gemma_4_license`,且**仓库内没有 LICENSE 文件**(经 `hf-mirror.com/api/models/google/gemma-4-12B` 的 siblings 核对,无 LICENSE/NOTICE)。

**Mistral / IBM / AllenAI / HuggingFaceTB(全注意力、许可干净)**
- 可取到真正的 `-Base`(Ministral-3 `-Base-2512`、Granite 4.1 `-Base`、SmolLM3 `-Base`、OLMo 3 7B 基座),许可均为 Apache-2.0。
- 代价是**全注意力**:32K–262K 的 prefill 成本与线性注意力不是一个量级;MLX 现成量化对 `-Base` 覆盖差。

**MoE 的最小可选面**
- 范围内可取到的 MoE:`Qwen3.5-35B-A3B-Base`(Apache-2.0,官方预训练)、`Qwen3-30B-A3B-Base`(Apache-2.0,上一代全注意力)、`gemma-4-26B-A4B`(Apache-2.0 表面)、`nvidia-Nemotron-3.5-Lightning-30B-A3B`(OpenMDW-1.1)、`LFM2.5-8B-A1B/24B-A2B`(专有)、`Moonlight-16B-A3B`(MIT,但 8K 上下文)。
- **MoE 的隐性成本**:Kev 在自己的记录里写过用 `Qwen3.6-35B-A3B` 做基座的结果是「**+1.2 pp,校准更差,8× 显存**」(`kev/PLAN.md`)。

---

## 3. 训练基建

### 3.1 Modal:机型、定价与额度机制

**机型(GPU 参数取值)**([Modal GPU 文档](https://modal.com/docs/guide/gpu)):`T4`、`L4`、`A10`、`L40S`、`A100`/`A100-40GB`/`A100-80GB`、`RTX-PRO-6000`、`H100`/`H100!`、`H200`、`B200`/`B200+`、`B300`。单容器最多 8 张(B300/B200/H200/H100/A100/L4/T4/L40S),`A10` 最多 4 张;`"H200:8"` 这种写法即单容器 8 卡。**B300 要求 CUDA ≥ 13.1**。文档另有两条操作性提醒:① **用 GPU 必须绑定有效支付方式**;② 请求 >2 卡通常显著拉长排队。

**单价(按秒计费;兑换成小时为本文换算值)**([Modal 定价](https://modal.com/pricing)):

| GPU | $/秒 | ≈ $/小时 |
|---|---|---|
| B300 | 0.001972 | 7.10 |
| B200 | 0.001736 | 6.25 |
| H200 SXM | 0.001261 | 4.54 |
| H100 SXM5 | 0.001097 | 3.95 |
| RTX PRO 6000 | 0.000842 | 3.03 |
| A100 80GB | 0.000694 | 2.50 |
| A100 40GB | 0.000583 | 2.10 |
| L40S | 0.000542 | 1.95 |
| A10 | 0.000306 | 1.10 |
| L4 | 0.000222 | 0.80 |
| T4 | 0.000164 | 0.59 |

其它资源:CPU **$0.0000131 / 核·秒**(≈$0.047/核·h,每容器最低 0.125 核)、内存 **$0.00000222 / GiB·秒**(≈$0.008/GiB·h)、卷存储 **$0.09/GiB·月**(含 1 TiB 免费)、出口 **$0.04/GiB**。**倍率项**:区域选择 **×1.15–1.75**;非抢占执行 **×3**。

> 交叉验证:`kev/budget.py` 内自带价目 `GPU_HOURLY = {"H100": 3.95, "H200": 4.54, "B200": 6.25, "T4": 0.59}`,`CPU_HOURLY = 0.04730`、`MEMORY_GIB_HOURLY = 0.008`,与上表**逐项一致**;说明 kev 的预算模型可以直接当作 Modal 计费的本地化实现。

**额度与预算机制**(modal.com/pricing、[Modal Budgets 文档](https://modal.com/docs/guide/budgets)):
- 计划:**Starter** $0 + 计算费,**$30/月免费额度**,3 席,100 容器 + 10 GPU 并发,1 TiB/月免费出口;**Team** $250 + 计算费,**$100/月免费额度**,5000 容器 + 50 GPU 并发,10 TiB;**Enterprise** 定制。另有 startup / academic credit grant 申请入口。
- **Workspace budget** = 计费周期内**使用量**月度上限(在抵扣额度之前);**spend limit** = 抵扣额度后**实付**上限,超出即停止会继续产生实付费用的工作;未自定义时默认 = 周期使用上限 − 已持有额度。**Environment budget** 仅 Team/Enterprise 可用,只覆盖计算,不含存储/预留/出口。
- 超时:Modal 单次调用上限 24 小时(kev 的 `MAX_TIMEOUT = {False: 28800, True: 86400}` 即按此设计,并靠 continuation 续跑)。

### 3.2 替代方案与成本量级(按需、单机时价)

| 平台 | 机型 | $/GPU·h | 备注(来自各自官方页) |
|---|---|---|---|
| **Modal** | H100 SXM5 / H200 / B200 / B300 | 3.95 / 4.54 / 6.25 / 7.10 | 按秒;真 serverless,scale-to-zero;区域/非抢占倍率 |
| **HF Jobs** | H200 / RTX PRO 6000 / A100-80 / L40S / L4 / T4 | 5.00 / 2.75 / 2.50 / 1.80 / 0.80 / 0.40–0.60 | 按分钟;**只在 Starting/Running 计费**(build 不计);8×H200 $40/h;暴露端口另收 $0.01/h |
| **RunPod Pods** | H200 / B200 / H100 SXM / L40S / RTX 5090 / L4 | 5.29 / 7.99 / 3.99 / 1.09 / 1.19 / 0.59 | 30+ 区域;按秒;页头标注 "Updated October 8, 2026" |
| **RunPod Serverless** | H200 / H100 PRO / A100 / L40S / 4090 / L4 | 5.93 / 4.79 / 2.72 / 1.75 / 1.10 / 0.69 | 有 idle timeout 的 scale-to-zero;**等待期仍计费** |
| **Lambda** | B200 / H100 SXM / A100-80 | 6.69 / 3.99 / 2.79 | 自营数据中心;1-Click Clusters: H100 $6.16、B200 $9.86 /GPU·h(16 卡起、2 周–1 年) |
| **Hot Aisle(AMD/ROCm)** | MI300X | 2.99(VM)/ 3.39(8 卡裸金属) | 1/2/4 卡 VM,192GB HBM3/卡,按分钟;裸金属 1.5TB HBM3 |
| **Vast.ai** | 市场定价 | 页面为实时价,未在静态 HTML 中给出数值 | 三档:on-demand / **interruptible(便宜 50%+)** / reserved(1/3/6 月) |

> **量级结论:** 单卡按需价在 **$0.6(L4/T4)–$8(B200/B300)** 区间;同型号跨平台差价约 ±20–35%(H200:H200 Jobs $5.00 vs Modal $4.54 vs RunPod Serverless $5.93)。8×H200 一夜(8h)在 Modal 的实付量级约 **$290**(8 × 4.54 × 8),这也与 kev 自己的封顶设计一致(`MAX_BUDGET = {False: 250, True: 1000}`,full-weight 上限「covers one 8 × H200 day」)。

### 3.3 自有硬件

- **Apple(MLX 路径的唯一硬件)**:Mac Studio 现役为 **M5 Max**(最多 128GB 统一内存,`614GB/s`)与 **M5 Ultra**(基线 96GB,可配 256/512GB,**1.2TB/s**;512GB 选项官方标注「late October」) ([apple.com/mac-studio/specs](https://www.apple.com/mac-studio/specs/))。**Kev-27B 的 51GB bf16 权重**因此落在 M5 Max 128GB 的上沿、M5 Ultra 的舒适区(kev 自述 96–128GB「expected, not measured」)。
- **单卡 NVIDIA 自购**:与云上按需价相比,80GB 级卡(H100/H200)一次性支出通常需数千美元级;是否有意义取决于利用率。本票未取到官方自购报价页,**不给出数值结论**(见 §7)。

### 3.4 单次训练的算力锚点(引用 `research/kev-recipe`,不重复)

- LoRA-on-frozen 路线:0.8B–9B 的单次 trial 在 L4/L40S/H100 级即可完成;
- 27B 全参:`Qwen3.8-27B` + FSDP2 on **8×H200**,fp32 masters、bf16 autocast、每小时 resume 点;Modal 侧一轮实验的 workspace 级实付曾达 **$2,432.57**(kev PLAN.md round 19 记录)。
- kev 的 Modal 应用默认 `GPU = os.environ.get("KEV_GPU", "H100")`,并注明「**H100 需要 workspace 绑定支付方式;免费额度请用 `KEV_GPU=T4`**」(`modal_app.py:42-43`)。

---

## 4. 服务目标

### 4.1 运行时 × 硬件支持矩阵

| 运行时 | NVIDIA CUDA | AMD ROCm | Apple MLX | 备注(一手出处) |
|---|---|---|---|---|
| **PyTorch + transformers**(Kev 的默认路径) | ✅ | ✅(`torch.version.hip` 分支) | ⚠️ 仅 MPS/CPU,混合骨干不推荐 | kev:ROCm 走 **eager**;fused 核与 CUDA graphs 只在 NVIDIA 打开(`kev/serve.py:344-346`) |
| **vLLM** | ✅ 预编译 CUDA 12.9 | ✅ **ROCm ≥ 6.3**;预编译 wheel 覆盖 rocm700(vLLM 0.14.0–0.18.0)与 rocm721(nightly) | ✅ 经社区插件 **vLLM-Metal**(以 MLX 为后端,吃 mlx-community 权重) | [vLLM GPU 安装文档](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html) |
| **llama.cpp / GGUF** | ✅ `-DGGML_CUDA=ON` | ✅ `HIP` | ✅ macOS 默认 Metal | 架构支持:`LLM_ARCH_QWEN35`、`LLM_ARCH_QWEN35MOE` 已在 `src/llama-arch.cpp`;后端列表含 Metal/SYCL/CUDA/HIP/Vulkan |
| **MLX / mlx-lm** | ✗ | ✗ | ✅ 唯一目标 | 支持架构含 `qwen3_5`、`qwen3_5_moe`、`gemma4*`、`ministral3`、`lfm2_moe`、`olmo3`、`smollm3`、`nemotron_h`、`gated_delta`、`kimi_linear` |
| **GPU 供应商可用性** | Modal / RunPod / Lambda / Vast / HF Jobs / HF Spaces 全是 NVIDIA | 自助可得的 AMD 云:**Hot Aisle**(MI300X);**Modal / RunPod / Lambda 的公开价目里没有 AMD** | Apple 自有硬件 | ROCm 官方支持卡:MI355X/MI350X/MI325X/MI300X/MI300A/MI250X/MI250 等([ROCm 兼容性页](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)) |

**结论性事实:**「CUDA + Apple MLX」这条双栈是**现成可走**的(RunPod/Modal/HF 提供 CUDA 卡,MLX 只需 Apple 机器);「ROCm」这条栈**软件上成立、硬件与供应商上稀薄**——既没有主流 GPU 云的 AMD 供给,也没有 Kev 的 fused 加速路径(退回 eager)。

### 4.2 服务栈现状:vLLM 已原生实现 System One(重要)

vLLM 文档里有一整节 **Structured Decisions Models (aka JEV Models)**,并明确:

- **原生 `POST /v1/systemone`**(TypeSafe System One 端点),**默认开启**;
- 同时提供 **OpenAI `POST /v1/decisions` 的 MVP 实现**(请求契约对齐 2026-10-07 取回的 OpenAI OpenAPI schema);
- 读法(`ReadStrategy`):基座模型走 **`NextTokenStrategy`**——把 state + 问题 + 选项拼成提示,读**第一个 token 的 logprob**,对标签做 softmax;**每个标签必须是单 token**,否则 400;
- 支持架构表里点名 `Qwen3_5ForConditionalGeneration`(即 Qwen3.5 全系)与 `Qwen3_5MoeForConditionalGeneration`(35B-A3B)、`Qwen3ForCausalLM`(Qwen3 系)、Cloudflare `ClefFlashForJointSchema`、`ClefOmniForJointSchema`、`D3ForSequenceClassification`(vllm-sr/d3)、`Lfm2VlForConditionalGeneration`、`DiffusionGemmaForBlockDiffusion`;
- **限制(与 Stim 的「完全兼容」承诺直接相关)**:每请求 **≤64 问题、≤26 选项、≤10 分档**;标签固定为 `A–Z` / `yes|no` / `0–9`;`confidence` 定义为「胜出标签在**整个词表**上的概率」;`/v1/systemone` 需要 `--logprobs-mode raw_logprobs` 或 `processed_logprobs`;**同一请求的概率会随 batch 组成与 prefix cache 命中而变化**,要严格可重复须设 `VLLM_BATCH_INVARIANT=1`(牺牲吞吐)。
- **Cloudflare Clef-Flash** 是走「pooling model + joint schema head」那条路的样例:vLLM 检测到 head 后按 pooling 模型服务,`state` 截断到 16,384 token(或 `max_model_len`),一次 forward 读完全部问题,`label_mass` 恒为 1。

### 4.3 托管形态与 scale-to-zero

| 形态 | scale-to-zero? | 计费 | 其他约束(一手) |
|---|---|---|---|
| **Modal**(Web Function) | ✅ 容器级弹性,空闲不计费 | 按秒,GPU + CPU + 内存 + 卷 | Starter 计划 Web Function 受限;区域/非抢占倍率 |
| **RunPod Serverless** | ✅ 队列空后等满 **idle timeout** 才停,**等待期仍计费**;**FlashBoot** 用于加速重启 | 按秒(见 §3.2) | 冷启动含镜像拉取与模型加载 |
| **HF Inference Endpoints** | ✅ **空闲 15 分钟自动降到 0 副本**;冷启动期间返回 **502**,**没有排队机制** | 按小时/机型 | 需要客户端自建重试队列 |
| **HF Spaces(自家 GPU)** | ❌ 按小时计费(GPU 机型) | A100-80 $2.50/h、L40S $1.80/h、L4 $0.80/h、T4 $0.40–0.60/h;4×L40S $8.30/h;**H100/H100×8 已于 2025-12 下线** | 存储与并发按 Space 计;文件系统有配额 |
| **HF Spaces(ZeroGPU)** | ✅ 共享池按调用给 GPU | 免费额度 + 超出 $1/10 分钟 | **只支持 Gradio SDK**;不支持 `torch.compile`(可用 AOT) |

### 4.4 演示位(HF Spaces / ZeroGPU)的限制

- **硬件**:共享 **NVIDIA RTX Pro 6000 Blackwell**,`large` = 半卡 48GB(配额 ×1)、`xlarge` = 整卡 96GB(配额 ×2,排队更长)。
- **每日配额**:未登录 2 分钟 / 免费账号 5 分钟 / **PRO 40 分钟** / Team 40 分钟 / Enterprise 60 分钟;24 小时滚动重置;**PRO 有 8× 额度与最高优先级**;超出后按 **$1 / 10 分钟**从预付额度扣。
- **托管上限**:免费账号最多 **2 个** ZeroGPU Space(需邮箱验证 + 账号 >30 天),PRO **10 个**,Team/Enterprise 50 个。
- **框架**:ZeroGPU **仅兼容 Gradio**;支持 PyTorch 2.8.0–2.13.0、Python 3.12.12/3.10.13;模型必须在模块顶层放到 `cuda`(ZeroGPU 外是 CUDA 模拟,里面才是真卡)。
- **对 Stim 的含义(事实,不构成建议)**:一个 4B 级 pointer-head 服务放进 ZeroGPU 是**可行但受限**的路径(单次调用时长上限靠 `@spaces.GPU(duration=…)` 显式声明,默认 60s);若演示需要长文(65K state),必须自己声明时长并承受排队与配额,或者改用自家 GPU Space(按小时计费、H100 已不可用)。

---

## 5. 现役同类工作的基座实证(谁在用哪个基座做 Jev 式模型)

这一节是给「基座抉择」用的**交叉证据**:同一批基座已经被别人用 Jev 式头跑通并公开了权重。

| 模型 | 基座 | 头/读法 | 许可 | 服务路径 |
|---|---|---|---|---|
| `Cloudflare/clef` | **Qwen3.8-27B**(含 vision encoder) | joint schema head,一次 forward 打分所有问题的所有选项;卡面自述「**fully compatible with Jev and SystemOne**」 | Apache-2.0 | vLLM(pooling),另有 GGUF/MLX 社区版 |
| `Cloudflare/clef-flash` | **Qwen3.5-9B** | 同上(joint schema head) | Apache-2.0 | vLLM 原生支持(`ClefFlashForJointSchema`),state 截断 16,384 |
| `Cloudflare/clef-omni` | Qwen3-Omni thinker | joint schema head | Apache-2.0 | vLLM |
| `vllm-sr/d3-mini` | **Qwen3.5-4B** | `D3ForSequenceClassification`(classify) | Apache-2.0 | vLLM |
| `mlx-community/Qwen3.5-0.8B-decision` | **Qwen3.5-0.8B** | LoRA + joint schema head(`optiq lora train --decision`) | Apache-2.0 | **MLX + OptiQ**(`optiq serve` 暴露 `/v1/systemone`) |
| `mlx-community/gemma-4-e2b-it-OptiQ-4bit-decision` | **gemma-4-E2B-it** | 同上 | gemma 许可标签 | MLX |

- Clef 一侧还公开了 **Decision Index 榜单**(`clef-evals.workers-ai-mle.workers.dev`),说明这一支已经在做可比较的评测。本次未能取回该页正文(超时,见 §7)。
- `mlx-community/Qwen3.5-0.8B-decision` 的卡面同时给出一个**训练量级的锚点**:该 0.8B decision 模型只用 benchmark 的 train split **1,080 条**训练,测试集上 accuracy 0.772;对照表里 27B 的 `clef-OptiQ-4bit` 在同一面板上 accuracy 0.719、p50 延迟 9.7s(M3 Max),0.8B 模型 0.34s。该卡还记录了一次**发布事故**:首个版本因适配器张量名 mlx-lm 无法放置而**没加载 LoRA**,accuracy 被误报为 0.512,2026-10-11 更正——**这是「自训自调自部署」链路上一个真实的失败模式**。

---

## 6. 风险标注(哪些选项会拖累「自训自调自部署」承诺)

按「许可 → 权重可得性 → 依赖链 → 运维」四类列出。**只标注,不排序,不拍板。**

### 6.1 许可

| 风险 | 具体项 | 依据 |
|---|---|---|
| **非 OSI 许可混入可选池** | `Llama 3.x/4`(Llama 社区许可)、`LFM2.5`(`lfm1.0` 专有)、`Nemotron-3.5`(OpenMDW-1.1)、`Qwen3.8-Flash-Next` / `Qwen3.8-2.4T-A95B`(标签 `other`) | 各仓库 `cardData.license` |
| **许可标注与其指向不一致** | `google/gemma-4-*` 卡面 `apache-2.0`,但 `license_link` 指向 Google 自有的 Gemma 4 license 页,**仓库内无 LICENSE 文件** | `hf-mirror.com/api/models/google/gemma-4-12B` |
| **上游许可变更风险被移除** | Qwen 三代(`3.5/3.6/3.8`)至今全部 Apache-2.0;但 `Qwen3.8-Flash-Next` 已转为 `other`,说明**同一厂商的不同分支许可并不统一** | 同上 |
| **数据集侧的许可** | 训练语料(如 banking77、CFPB complaints、commitpackft 等)各有许可,Kev 在模型卡里逐项列了 datasets 字段 | kev 模型卡 YAML |

### 6.2 权重与「起点」的可得性

- **27B 档没有 `-Base`**:Qwen3.5/3.6/3.8 的 27B 只有后训练版。选它就意味着**站在别人的后训练上**,而 Kev-27B 的卡直接把这件事写成限制(「we don't know what that was trained on」)。想要「全开放训练代码」的家族,这是一个结构性妥协。
- **`-Base` 仓库是多模态包装**:Qwen3.5 的 `-Base` 用 `Qwen3_5ForConditionalGeneration` 承载,含 vision token 与 vision tower;Kev 的做法是**不加载** vision/LM head/MTP 层,这依赖具体实现的加载裁剪逻辑(kev 有 `full_ft`、`interpolate` 等路径配合)。
- **`-it` 与 base 的混淆**:社区现成量化几乎只覆盖 post-trained;若选 base,转换与验证要自己做(架构层面 mlx-lm/llama.cpp 已支持)。

### 6.3 依赖链(软件)

- **CUDA 专属加速**:kev 的服务性能走 **flash-linear-attention 的 Triton 核 + CUDA graphs**(`kev/fused_qwen35.py` 头注、`kev/serve.py:344-346`)。ROCm 上默认退回 eager,**性能结论不可平移**。
- **MLX 只覆盖 Apple**:MLX 后端与 `KEV_BACKEND=torch` 是两套加载路径(kev:Apple Silicon + 混合 checkpoint 默认走 MLX;**MLX 全参路径的适配器合并行为不同**——adapter 会折叠进权重并短暂持有两份权重)。
- **演示位强绑定 Gradio**:ZeroGPU 只支持 Gradio SDK;若要暴露自定义端点(如 `/v1/systemone` 的原始形态),ZeroGPU 不是合适载体(kev 的 Space 就是把 `kev/model.py` 与 `kev/api.py` **逐字复制**进 Space 来跑)。
- **vLLM 的语义差异**:vLLM 的 `/v1/systemone` 用 next-token logprob 读标签,**每标签必须单 token**、选项上限 26、分档上限 10,且**概率会随 batch 组成变化**。与 pointer head 的概率语义和上限(255 选项)不同——把它当作「兼容」需要逐项对齐,而这一点 `research/system-one-protocol` 已在协议层展开。

### 6.4 运维与成本

- **非抢占执行 ×3**:Modal 上要把大规模训练跑成非抢占,单价直接三倍;抢占是默认经济性的一部分,意味着需要 checkpoint/resume 的工程投入(kev 为此做了 attempt lease、resume point、continuation)。
- **出口费**:Modal $0.04/GiB,权重在云与本地之间来回搬运(HF 镜像站不解决这个问题)。
- **额度陷阱**:免费/Starter 额度是**月度**且**先抵扣后实付**,而 GPU 需要绑卡;kev 明确写「免费层只能用 T4 级」。
- **演示位冷启动**:Spaces 的 GPU 是按小时计费、Inference Endpoints 是 15 分钟缩零 + 502 无队列,体验与成本的权衡是硬约束。

---

## 7. 不可达来源与未核项

| 来源 | 状态 | 影响 |
|---|---|---|
| `huggingface.co`(直连) | 不可达(000) | 全部 HF 读取经 `hf-mirror.com`;`resolve/` 路径会 307,本文件统一改用 `/raw/` 与 `/api/` |
| `raw.githubusercontent.com` | 不可达 | GitHub 文件改写经 `gh api ... -H "Accept: application/vnd.github.raw"` |
| `huggingface.co/docs`(直连) | 不可达 | 文档改经 `hf-mirror.com/docs/...`,并对 Spaces 类文档直接读 `huggingface/hub-docs` 仓库源文件 |
| `clef-evals.workers-ai-mle.workers.dev`(Decision Index 榜单) | 本次取回为空/超时 | §5 只记录该榜单存在,未引用其数值 |
| `www.amd.com/.../cloud-access*` | 两次取回均为空 | ROCm 云侧只用了 Hot Aisle(自助、有价)与 ROCm 官方兼容性页 |
| `vast.ai/pricing` 的具体 GPU 单价 | 页面为实时价,静态 HTML 未含数值 | §3.2 只记录其三档定价结构(interruptible 便宜 50%+) |
| `google/gemma-4-12B` 的 LICENSE 文件 | 仓库内不存在(返回 "Entry not found") | 许可只能引用卡面声明与外部链接 |
| `Lambda` 自购/自建硬件报价、NVIDIA 单卡零售价 | 未取 | §3.3 未给自购数值结论 |
| `moonshotai/Kimi-Linear-48B-A3B-Base` 的上下文声明 | config 未给出 `max_position_embeddings`(在代码侧) | 表中该列留空 |
| Modal 定价页 FAQ 的完整条款、Modal「free tier」的官方表述原文 | 未逐条核读 | 「免费额度」只用了 Starter 计划字段($30/月)与 kev 注释(免费层用 T4)两处一手证据 |

---

## 8. 未决问题(交给下游票)

1. **27B 档的「全开放」怎么定义?** 三代 27B 都无 `-Base`,起点只能是 Qwen 的后训练权重。是否接受「无 `-Base`」这一限制,决定 #9 是落在 9B/35B-A3B(有 `-Base`)还是 27B(无)。
2. **Qwen3.5 与 Qwen3.6/3.8 的关系未核清。** 三代配置形状一致(`3.6/3.8` 显式 `partial_rotary_factor 0.25`),但 Qwen 未公开说明 3.6/3.8 相对 3.5 改了什么(训练数据?后训练?);这影响「相同架构、不同版本」的迁移成本判断。
3. **MoE 在 Jev 式头下的性价比未验证。** kev 只有一条 `Qwen3.6-35B-A3B` 的记录(+1.2 pp、校准更差、8× 显存);官方 `35B-A3B-Base` 尚未见到公开的 decision 模型。这是 #9 与 #3 的交叉问题。
4. **ROCm 的真实可用性**:软件(ROCm ≥6.3 / llama.cpp HIP / vLLM wheel)都成立,但**主流 GPU 云没有 AMD 供给**,且 Kev 的融合核不支持。若要兑现「自部署到 AMD」,需要先确认目标硬件与供应商,以及是否接受 eager 性能。
5. **vLLM `/v1/systemone` 与 pointer head 的语义等价性**(单 token 标签约束、`confidence` 定义、batch 不变性)未在本文逐项比对,应落在 #14 / 协议票。
6. **MLX 侧的实测缺口**:9B 的长文(65K)与 27B 全参在 Apple 上的行为,Kev 自己标为「expected, not measured」;若 Stim 要走 MLX,需要补测。
7. **`gemma-4` 家族的实际许可文本**需人工打开 `ai.google.dev/gemma/docs/gemma_4_license` 确认(Apache-2.0 的表述是否伴随附加条款)。
8. **Vast.ai / CoreWeave / Nebius / Together 等平台的具体按需价与 SLA** 未取;若预算票 #3 需要更细的价格曲线,应补一轮。

---

## 9. 来源清单(按引用顺序)

**基座(经 hf-mirror.com)**
- `hf-mirror.com/api/models?author=Qwen`(分页 8 页)、`?search=Qwen3.5`、`?search=Qwen3.5-27B`、`?search=Qwen3.8-27B`
- 各仓库 `config.json`:`Qwen/Qwen3.5-{0.8B,2B,9B,35B-A3B,27B}-*`、`Qwen/Qwen3.{6,8}-*`、`Qwen/Qwen3-{4B,8B,30B-A3B}-Base`、`google/gemma-4-{12B,E4B,26B-A4B}`、`mistralai/Ministral-3-*-Base-2512`、`mistralai/Mistral-Nemo-Base-2407`、`mistralai/Mistral-Small-24B-Base-2501`、`allenai/Olmo-3-*`、`ibm-granite/granite-4.*`、`HuggingFaceTB/SmolLM3-3B-Base`、`LiquidAI/LFM2.5-*`、`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`、`moonshotai/Moonlight-16B-A3B`、`moonshotai/Kimi-Linear-48B-A3B-Base`
- 模型卡:`Qwen/Qwen3.5-9B-Base`(README)、`google/gemma-4-12B`、`google/gemma-4-E2B`、`Cloudflare/clef`、`Cloudflare/clef-flash`(API 元数据)、`vllm-sr/d3-mini`(API 元数据)、`mlx-community/Qwen3.5-0.8B-decision`

**训练与算力**
- `modal.com/pricing`、`modal.com/docs/guide/gpu`、`modal.com/docs/guide/budgets`
- `runpod.io/pricing`(页头标注 Updated October 8, 2026)、`docs.runpod.io/serverless/workers/overview`
- `lambda.ai/pricing`、`hotaisle.xyz/pricing`、`vast.ai/pricing`
- `huggingface/hub-docs`:`docs/hub/jobs-pricing.md`、`docs/hub/spaces-gpus.md`、`docs/hub/spaces-zerogpu.md`
- `apple.com/mac-studio/specs/`

**服务**
- `docs.vllm.ai/en/latest/models/structured_decisions_models/`、`docs.vllm.ai/en/latest/getting_started/installation/gpu.html`
- `ggml-org/llama.cpp`:`docs/build.md`、`src/llama-arch.cpp`
- `ml-explore/mlx-lm`:`mlx_lm/models/`(含 `qwen3_5.py`、`qwen3_5_moe.py` 等;仓库 `pushed_at` = 2026-10-16T02:49:02Z)
- `rocm.docs.amd.com/projects/install-on-linux/.../system-requirements.html`
- `hf-mirror.com/docs/inference-endpoints/autoscaling`

**Kev(一手源码/文档,浅克隆 `/tmp/kev-src`)**
- `README.md`、`PLAN.md`、`modal_app.py`、`kev/budget.py`、`kev/model.py`、`kev/fused_qwen35.py`、`kev/serve.py`、`kev/checkpoint.py`、`docs/model-cards/kev-{0.8b,4b,9b,27b}.md`、`space/README.md`
