# Kev 1.0 的复现配方与成本

研究票:Stim #5「Kev 1.0 的复现配方与成本」·路线图 #1
调查对象:`jaredpalmer/kev`(一手来源:仓库本身)
快照:浅克隆 `/tmp/kev-src`,HEAD `1f04cfa`(2026-10-15,即 Kev 1.0 发布后、含 round 28/29 记录);历史版本经 `gh api` 读 tag `research-archive-2026-09-24` 下的 `PLAN.md`。
引用格式:`kev/train.py#L588` 指仓库内文件行号;`runs/...` 为仓库内已提交的记录目录。

Kev 1.0 = 2026-10-01 用 `v1.0` tag 统一锁定的四个 checkpoint,**没有新训练、没有新评测读**(`docs/releases/kev-1.0.md#L3`,PLAN.md "Released: Kev 1.0" L2674-2712)。以下配方按发布时的实际训练阶段(0.8B 15 轮、4B 8+10 轮、9B 18 轮+27 轮重选、27B 22+23 轮)整理。

---

## 0. 四个 checkpoint 一分钟速查

| | Kev-0.8B | Kev-4B | Kev-9B (v2) | Kev-27B (v2) |
|---|---|---|---|---|
| 形态 | LoRA adapter + head | LoRA adapter + head | LoRA adapter + head | 全参 bf16 权重(51.3 GB) + head |
| Hub 权重 revision | `9a45d25e` | `139fdd94` | `b5d8c18e` | `28be62e9`(Hub `v1.0` tag 指向卡提交 `af0e6d55`) |
| 温度 T | 2.35(2.3511) | 2.41(2.4061) | 2.19(2.1936) | 1.32(1.3195) |
| 训练方 | 轮 15:`r15-08b/00-trial-0` | 轮 10:`r10-skills/00-trial-0` | 轮 18:`r18-9b/00-trial-0`(轮 27 重选) | 轮 23 blend `27b-k-w85`(轮 22 SFT) |
| validated context | 8,192 | 8,192 | 8,192 | 65,536 |

来源:`docs/releases/kev-1.0.md#L7-12`(形态/基座/T)、`PLAN.md` L37-43(Kev 1.0 baseline 表,含 trial 路径与 Hub revision)、`docs/model-cards/kev-*.md`。

---

## 1. 基座(名称/版本/许可;LoRA 与全参的界线)

### 1.1 四个基座与许可

| 尺寸 | 基座 | revision | 层结构 | 许可 | 卡内出处 |
|---|---|---|---|---|---|
| 0.8B | `Qwen/Qwen3.5-0.8B-Base` | `dc7cdfe2` | 24 层:18 Gated DeltaNet + 6 full attention | Apache-2.0 | kev-0.8b.md#L78 |
| 4B | `Qwen/Qwen3.5-4B-Base` | `1001bb4d` | 32 层:24 DeltaNet + 8 full;hidden 2,560 | Apache-2.0 | kev-4b.md#L73 |
| 9B | `Qwen/Qwen3.5-9B-Base` | `68c46c4b` | 32 层:24 DeltaNet + 8 full;hidden 4,096 | Apache-2.0 | kev-9b.md#L71 |
| 27B | `Qwen/Qwen3.8-27B`(Qwen 的 **post-trained** 版本,非 `-Base`) | `1d4bf0f2` | 64 层:48 DeltaNet + 16 full;hidden 5,120;只加载 25.6B backbone(vision tower/LM head/MTP 层不载) | Apache-2.0 | kev-27b.md#L91-93 |

- 家族许可:仓库 Apache-2.0;基座亦 Apache-2.0(`README.md#L503`,显式声明 "The Qwen3, Qwen3.5 and Qwen3.8 base models are also Apache-2.0")。
- 选型线索(为什么是这组基座):全系押注 Qwen3.5/3.8 的混合 Gated DeltaNet 架构;README 说 Kev "built on Qwen3.5 and Qwen3.8 and based on the architecture described in Jev's Architecture Unmasked"(`README.md#L13`);此前的 Qwen3 世代(Kev-0.5b/0.6b/4b/8b)保留在 Hub 但不再开发(`README.md#L470-480`)。
- 知识量由基座决定:MMLU-Pro 0.230 / 0.565 / 0.590 / 0.675(四尺寸),Jev 0.840(`docs/releases/kev-1.0.md#L24`);PLAN "What we have learned" #8 注明 "Knowledge is set by the base"(PLAN.md L~240)。
- 27B 用 post-trained 而非 `-Base`:候选探针(A2)三项门槛过二(MMLU-Pro 0.635 < 0.65),由 Jared 记录 override(`PLAN.md` Record 表 "A2",行 ~3130);README 明说 "we don't know what it was post-trained on"(`README.md#L318`)。教训证据:post-trained 9B 的 date arithmetic 掉得更多(0.70→0.47,round 4.8 负结果,PLAN.md "What we have learned" #9)。

### 1.2 LoRA 配置(0.8B/4B/9B 全同构)

- rank 16、α = 2r = 32(代码里 `lora_alpha=2*lora`)、dropout 0.05、`task_type=FEATURE_EXTRACTION`(`kev/model.py#L265-275`)。可训练参数:0.8B 11.3M、4B 33.8M、9B 45.4M(卡 L79/L74/L72)。
- target modules(`kev/model.py#L268-273`):
  - 默认 `all` = `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`,混合基座上再追加 Gated DeltaNet 投影 `in_proj_qkv,in_proj_z,in_proj_a,in_proj_b` 和 mixer 的 `out_proj`;
  - 其余消融档:`dense`(去掉 DeltaNet 投影)、`attn`(仅 attention)、`qv`。
- 指针头(pointer head)从零训练:`PointerHead(hidden_size, head_dim=256)`,把每个 option 的闭合 token 隐藏态与问题的 `<decide>` 隐藏态对齐打分,softmax 出概率(`kev/model.py#L276`;机制见 `README.md#L321-323`;27B 卡写明 "two 5,120 → 256 projections",kev-27b.md#L92)。
- 基座冻结;`bf16 autocast over fp32 master`;服务时把 adapter merge 进 base(bf16),评估走 fp32(`kev-0.8b.md#L81` 等三张卡)。

### 1.3 LoRA vs 全参的界线(实际判据)

- 界线定在 **27B v2**:0.8B/4B/9B 全部是 LoRA-on-frozen;27B v1 也是 rank-16 LoRA(`runs/r6-27b-v2/01-trial-1`,tag `v1-lora`),v2 改为全参 SFT,发布为 51 GB 全权重(`kev-27b.md#L91-101`)。
- 判据不是尺寸本身,而是一次受控对照(round 19:同一数据下 full weights vs LoRA,精度 "no accuracy gain anywhere (breadth −0.1, Kev panel −0.8), short states −3.0 [−4.3, −1.9], scienthoon −3.7 [−6.0, −1.5], ECE +0.043~+0.048";而换数据(full weights on broad data vs on Kev's own data)带来 Kev panel +9.1 [+7.8, +10.5]。结论:"the broad data carries the gains, and the scienthoon and short-state costs come with full weights"(`PLAN.md` L382,Round 19 attribution)。
- 因此 27B v2 = 全参(数据宽)+ 与 v1 权重平均 0.85:0.15(把 v1 的 LoRA 在 fp32 里 merge 后混),用来抵消全参漂移;混合比例在六个 blend 里按 development 选,加权后 ECE、short-state 等恢复(`kev-27b.md#L181`;round 23)。
- 相关负结果(不要再试):DeltaNet-frozen LoRA(night 2 #5)、question-side LoRA(A1:4B/9B −3.4~−6.0 pp)、`lora_scale`/WiSE-FT-as-default(PLAN.md "What we have learned" #13)。
- 代码入口:`--full_ft 1` 走 `kev/full_ft.py`(bf16 权重 + fp32 masters,`MasterAdamW`;多卡 torchrun+FSDP2),`kev/train.py#L418`;LoRA/head 预热起点用 `--init_from`(`#L415-417`: "warm-start LoRA and the pointer head from an existing run (local directory or hub id)…keeps the released model's in-domain skill while adapting to a new domain")。

---

## 2. 数据(构成、规模、许可、冻结方式、只读套件)

### 2.1 共用基座配方 `decision-v7`(四尺寸第一/第二阶段的训练集)

文件:`evals/v7/decision-v7/`(manifest + 分片);规模 train 12,576 / calibration 968 / development 1,204 / test 1,176(`evals/v7/decision-v7/manifest.json`)。

| 成分 | 数量 | 生成配方 | 标签来源 |
|---|---|---|---|
| 十个公开分类数据集 | 10,000(各 1,000) | `kev/data.py#L157-232` 的转换器:banking77、boolq、ag_news、multi_nli、sst5、yelp_review_full、trec、dbpedia_14、amazon_reviews_multi_en、imdb | 数据集原生标签 |
| 政策 minimal pairs | 896 | `kev/contrastive.py`:每条 = 政策 + 案例,同一 item 只改一句话使标签翻转;每对自动跑两个代码检查——**ablation**(去掉任一证据句 → UNDETERMINED)与 **invariance**(去掉任一 filler → 标签不变);按 family 分 split,兄弟不跨 split | `evaluate(facts)` 纯函数,代码计算 |
| 规则结构 | 1,680 | `kev/composition.py`:60 个随机规则树 × 4 种渲染;14 种 shape 中 train/dev/test 按 shape 切开(`TRAIN_SHAPES` 前 8 / `DEV_SHAPES` 3 / `TEST_SHAPES` 3) | 代码计算 |

`decision-v7` 的 eval-only 源(永不入训):`mmlu, emotion, tweet_offensive, qnli, paws, sciq, contrastive(holdout 档), legacy_holdout, composition_holdout`(manifest 字段;训练时由 `kev/suite.py#L181` `validate_training` 强制)。

### 2.2 日期与缺失证据(night 2 delta;1,425 条)

`evals/night2/`(`scripts/build_night2_data.py` 生成,seed `night2-20260920`;文件 dates 900 + unknowable 525,合并 `dates_unknowable.jsonl` 1,425):
- `dates.jsonl` 900:带日期的政策案例,三种渲染各随机——plain / 附一句关系式日计数("The return request was submitted 23 days after the purchase.")/ 带 `date_facts` 字段(与 `kev.api.with_date_facts` 服务时输出一致);仅可训 family(`deadline`、`authorization` 的 held-out family 不用)。
- `unknowable.jsonl` 525:255 条抽掉决定性证据、soft target = 均匀分布;另 270 条完好对照(硬标签)。教 "无证据→无置信"(`scripts/build_night2_data.py` docstring L8-18)。

### 2.3 documents-v1(真实文档)

- 来源:`davidheineman/consumer-finance-complaints-large@44cfa170`,美国 CFPB 消费者投诉库,manifest `rights` 写 "US government work (public domain)"。
- 标签协议(manifest `label_protocol`):train 用两个 open-weight teacher(deepseek-v3.2、qwen3-235b-a22b-thinking)与原生标签**同时一致**才保留(5,219 篇 / 7,488 问);dev/test 用三家闭源评委(claude-opus-4.5、gpt-5、gemini-3-flash)全票或双 Devin 子代理裁决(见 `scripts/label_documents_v1.py` 头部;manifest `label_report` 记每档 kept/dropped)。
- 冻结:manifest 有 sha256;test locked;`documents-v2` 为私有 held-out test(git 内只有 manifest,镜像 `jaredpalmer/kev-private-evals`)。

### 2.4 hard-v1(程序化技能)

- train 6,000 条,七个 family:long policy(带例外/sublimit)、trade-off(按给定优先级)、prob/EV、多跳、日期与算术、judge(评估候选答案)、missing-fact abstention(卡表 L142)。
- 生成器:`scripts/hard_v1_families.py`(tradeoff / multi_hop / ambiguous)+ `scripts/hard_v1_numeric.py`(probability / temporal_numeric / judge)+ `scripts/hard_v1_policy.py`(long_policy);每 family 6 个 surface template,**template 4-5 held out of training**;标签一律来自生成器的 `solve()`,不从生成器取("never from the generator")。
- test split 只含 trained generator 的 held-out templates;标签程序化(卡表注明 "programmatic labels")。

### 2.5 devtools-v1(开发者工具)

- train 5,320 条,四个可训源(manifest `trainable_sources`):`codereviewer`(Zenodo DOI 10.5281/zenodo.6900648,CC-BY-4.0;逐仓库 licence 用 GitHub API spdx_id 过滤,只留 permissive,GPL/AGPL/LGPL/MPL 等丢弃)、`commitpackft`(MIT + per-row repo licence)、`flakeflagger`、`aegis`。
- eval-only 源(只读不训):`when2call`、`prompt_injection`(manifest `eval_only_sources`)。
- 每个源的 licence / attribution / label_provenance / pins(zenodo md5+sha256)都写在 `evals/devtools-v1/manifest.json` 里。

### 2.6 round15 joint(0.8B/9B 第三阶段数据)

`evals/round15/joint/manifest.json`:`concat_of: ["evals/documents-v1 train", "evals/round10/skills/train.jsonl"]`,16,539 条;git 里只有 manifest,train.jsonl 在 HF `jaredpalmer/kev-suites`(大于 10 MB 的训练分区出 git,见 §2.8)。

### 2.6b 27B v1(LoRA 版)训练集: `b1v2`(15,401 条)

`evals/round6/b1v2/manifest.json` 的 `construction` 字段写明了拼法:"decision-v7 train with threshold-0.8 ambiguity targets in place (MNLI targets removed) + dates/unknowable + 1,400 long v2 records (200/200/500/500)"——即 decision-v7 train(12,576,含 soft target 化)+ night2(1,425)+ `evals/round5/longstate-v2`(1,400);manifest 还记 `code_sha256` 与三个输入的 sha256;payload 同样从 `kev-suites` 拉取。

### 2.7 27B 全参语料 `sft-v2-r22`(私有大语料)

`evals/sft-v2-r22/manifest.json`(只公开 manifest;payload 在私有数据集 `jaredpalmer/kev-private-train @ f8d59fbb`,1.796 GB)。规模:**145,840 条 / 337,130 问**(soft-target 题 11,563),states ≤ 32,768 tokens(p50 = 327,p99 = 26,882,max 32,765,合计 3.32 亿 state token;按 shared_prefix 计的 row token 3.55 亿);一个 epoch。十块成分(kev-27b.md#L157-172):

| 成分 | 条数 | 内容 |
|---|---|---|
| Kev corpus | 78,786 | v1 训练集(十公开集+政策/规则+日期/缺失证据+长状态埋点)+ hard-v1/devtools-v1/documents-v1 的 **train** split + 24 个公开集各 ≤500 条 + 7 个生成 family |
| Licensed task families | 24,000 | 公共多任务集的 119 个 task family,licence 允许商用;family 名单不公开 |
| Long states | 3,200 | Kev-corpus state 埋进 8k–32k 文档,标签原样继承 |
| Long documents | 7,617 | 代码拼装,标签代码计算 |
| Out-of-domain decisions | 7,312 | 多域生成决策 |
| Tone | 7,544 | 同一文本 calm/frustrated/angry 的 minimal pairs |
| Prompt injection | 2,699 | 识别间接注入(防御向) |
| Agent sessions | 4,875 | 关于代码生成的 agent session 日志的问题 |
| PII | 4,152 | 代码插入的**虚构**个人数据分类 |
| Grounding | 5,655 | 论断是否被文档支持 |

- 许可:四个 capped 公开集是 share-alike(ARC CC-BY-SA-4.0、HotpotQA CC-BY-SA-4.0、Natural Questions CC-BY-SA-3.0、SNLI CC-BY-SA-4.0),其余 CC-BY-4.0 / MIT / Apache-2.0;CFPB 为美国政府作品;GLM-5.3 为 MIT 风格(超大规模 MaaS 运营者有附加条件)(kev-27b.md#L174)。
- 生成文本与标签来自代码或 open-weight 模型(GLM-5.3、DeepSeek-V4-Pro、Inkling、Mistral Large 3、gpt-oss-120b、MiMo-V2.6-Pro);CFPB 标签来自 open-weight 模型,再经闭源评委过滤与裁决。
- 去污染:训练前逐条筛 102 个评测分区(76,549 参考条),命中 exact-normalised 字符串 / word-8-gram Jaccard > 0.2 / containment ≥ 0.5 即删,共删 6,388 条(kev-27b.md#L176)。

**铁律:任何阶段都没有使用 Jev(TypeSafe 托管模型)的输出**——Jev 只做 AI Gateway 上的参考读(`PLAN.md` L301 "No Jev output in training, ever";各卡均注明)。

### 2.8 冻结方式(suite freeze)

- 每个 suite 目录一个 `manifest.json`,记录每个分区的 **sha256 + 记录数**;`load_split` 逐字节校验 sha256 与条数后才给数据(`kev/suite.py#L110-118, #L204-221`)。
- 分区分四档:`train / calibration / development / test`;test 可标 `locked`;`--allow-test` 才允许读 test,发布时每个模型对 test 只读一次。
- 大文件(> 10 MB)不进 git,首用时从 HF 数据集 `jaredpalmer/kev-suites`(revision `cc4bac80`)拉取并按 manifest sha256 校验;私有 held-out suite 在 manifest 里自带 `"mirror": {"dataset": "jaredpalmer/kev-private-evals", ...}`,git 里只留 manifest(发布哈希、不发布文本)(`kev/suite.py#L32-41`)。
- `validate_training` 强制 "训练记录只能来自 manifest 声明的 trainable sources,不能来自 eval-only 或 held-out source"(`kev/suite.py#L181-198`)。
- suite 冻结后不可改;发现不健全的 suite 直接**删除并登记**:`kev/suite.py#L49-85` `REMOVED_SUITES` 记录 scienthoon-v1(2026-09-27 删)、wanli-v1/v2、typesafe-v1(2026-09-30 删)及理由;`load_split` 会直接拒绝读,**旧轮次的已提交 rows 作为历史记录保留**。

### 2.9 只读不训(evaluation-only)套件清单

| 套件 | 内容 | 规模 | 存放 |
|---|---|---|---|
| breadth-v1 | 14 个 held-out 公开数据集(五领域) | dev 3,075 / test 3,089 | 私有镜像(kev-private-evals) |
| tasksource-heldout-v1 | 公共多任务集的 24 个整 task family(名单私有) | dev 2,788 / test 2,024 | 私有镜像 |
| transfer-v4 | 6 个从未训练源(QNLI/SciQ/TweetEval-offensive/PAWS/MMLU/Emotion)+ held-out 政策/规则结构 | dev 656 / locked test 656 | git(`evals/v4/transfer-v4`) |
| transfer-v9 | transfer-v4 + 10 选项 MMLU-Pro + buried + unknowable | dev ~1,046 | git(`evals/v9`) |
| documents-v1(v2) | CFPB 文档;v2 私有 test | test 936 / v2 953 | git + 私有 |
| longdoc-v1 | CUAD 合同 + 生成协议包,4k–64k states | dev/test | 私有镜像(manifest `eval_only_sources: longdoc_cuad, longdoc_synthetic`) |
| hard-v1 test / devtools-v1 test | held-out templates / 全部源 | 1,088 / 1,071 | git |
| ood-v2 / agents-ood-v1 / guardrails-ood-v1 | 生成器的 held-out 域 | 4,988 / 2,084 / 4,949 | 私有镜像 |
| When2Call、prompt-injection | devtools-v1 内的 eval-only 源 | — | git |
| SemIf(外部) | 144 条人工决策,近饱和,只报不判 | 144 | `evals/external/`(retired 的 scienthoon/WANLI/TypeSafe 已删) |

清单依据:各 suite manifest 的 `eval_only_sources`/`holdout_sources`、`README.md#L434-450` 的套件表、`kev/suite.py#L49-85`。

---

## 3. 训练(超参、硬件与时长、Modal、`--init_from`)

### 3.1 `kev/train.py` CLI 关键默认值(`#L372-438`)

| 参数 | 默认 | 说明 |
|---|---|---|
| `--base` / `--base_revision` | `Qwen/Qwen3-0.6B-Base` / "" | 基座与 revision 钉死(suite manifest 未钉时用) |
| `--epochs` / `--lr` | 1 / 2e-4 | OneCycleLR,max_lr=[lr, head_lr or lr],`pct_start=0.1`(10% warm-up,`#L588`) |
| `--head_lr` | 0.0(= 同 lr) | head 从零训练可单独给 lr |
| `--lora` / `--lora_targets` | 16 / `all` | target 映射见 §1.2 |
| `--batch` / `--accum` | 1 / 8 | 每 step 的 records = batch × accum × world_size |
| `--dtype` / `--weights_dtype` | fp32 / fp32 | 实际实验都用 `bf16`(autocast over fp32 masters);full-ft 要求 `--weights_dtype bf16` |
| `--checkpointing` | 0 | 长状态阶段开 1 |
| `--max_state` | 384 | 训练上下文;实际实验 7,552(0.8/4/9B)或 32,768(27B);`kev/model.py#L14-38` |
| `--p_none` / `--p_none_distract` / `--p_distract` | 0.1 / 0.12 / 0.15 | 随机插入 "none of the above"/distractor |
| `--p_none_pair` | 0.0(实际 0.25) | 1/4 Choice 记录额外产出 none-present / none-absent minimal pair;27B 用 `--none_pair_max_state 8192` 限制只在短 state 上做 |
| `--replay` | 0 | 与 `--data`/`--suite` 联用,混合 N 条 suite 训练记录(防遗忘) |
| `--init_from` | "" | delta 模式:从既有 run(本地目录或 hub id)热启 LoRA + head |
| `--full_ft` | 0 | 全参 SFT(见 §1.3) |
| `--shared_prefix` / `--length_sort` / `--pass_tokens_max` | 随 full_ft / 0 / 0 | 混合 backbone 的显存与吞吐控制(27B 全参用了这三个) |
| `--max_steps` / `--save_every_*` / `--resume` / `--snapshot_fractions` | 0 / 0 / 0 / "" | full-weight 的断点续训与快照;study 默认快照 0.25/0.5/0.75(`kev/experiment.py#L56` `SNAPSHOT_FRACTIONS`) |
| `--seed` | 0 | |

损失:对每题 options 的 cross-entropy(有 soft target 时用 soft target),单一 loss,不加 perm_kl/ord_w(README L~348 "The released models don't use the optional --perm_kl or --ord_w losses";负结果为 label smoothing/CE+Brier/focal,PLAN.md #13)。

### 3.2 Kev 1.0 各阶段实际超参(逐 trial 的 `config`)

| 阶段 | trial(`runs/...`) | epochs | lr | batch×accum | dtype/checkpointing | 其他 | 出处 |
|---|---|---|---|---|---|---|---|
| 0.8B 基座 | `q35-08b/02-trial-2`(tag `v7-base`) | 2 | 1e-4 | 8×1 | bf16 / 0 | seed 2 | provenance.json |
| 0.8B 日期 | `night2-08b-du2/00-trial-0` | 1 | 4e-5 | 8×1 | bf16 / 0 | `--replay 2000`,`--init_from jaredpalmer/kev-0.8b` | 同上 |
| 0.8B 文档+技能 | `r15-08b/00-trial-0` | 1 | 2e-5 | 4×2 | bf16 / 1 | replay 6000,max_state 7552,seed 1,2,818 steps | 同上 |
| 4B 基座 | `q35-4b-s23/00-trial-0` | 2 | 5e-5 | 4×2 | bf16 / 1 | seed 2 | 同上 |
| 4B 日期 | `night2-4b-du/00-trial-0` | 1 | 2e-5 | 4×2 | bf16 / 1 | replay 2000 | 同上 |
| 4B 文档 | `r8-small/00-trial-0`(tag `r8-documents-release`;该目录未提交 `provenance.json`) | 1 | 2e-5 | 2×4 | —(卡未列) | replay 2000,max_state 7552 | kev-4b.md#L151;trial 路径见 kev-4b.md#L301 |
| 4B 技能 | `r10-skills/00-trial-0` | 1 | 2e-5 | 2×4 | bf16 / 1 | replay 4000,seed 1,1,915 steps | provenance.json |
| 9B 基座 | `q35-9b/01-trial-1` | 2 | 5e-5 | 2×4 | bf16 / 1 | seed 1 | 同上 |
| 9B 日期 | `night2-9b-du/00-trial-0` | 1 | 2e-5 | 2×4 | bf16 / 1 | replay 2000 | 同上 |
| 9B 文档+技能(v2) | `r18-9b/00-trial-0` | 1 | 2e-5 | 2×4 | bf16 / 1 | replay 10000,seed 1,3,318 steps | 同上 |
| 27B v1(LoRA) | `r6-27b-v2/01-trial-1` | 1 | 5e-5 | 1×8 | bf16 / 1 | max_state 7552,seed 2 | 同上 |
| 27B v2(全参) | `r22-27b-lr2e6/00-trial-0` | 1 | 2e-6 backbone / 1e-4 head | 8×2(8 卡,128 records/step) | bf16 weights / 1 | AdamW β0.9/0.999 wd 0.01,fp32 masters,grad clip 1.0,one-cycle 10% warm-up,seed 0,1,140 steps,max_state 32768,shared_prefix 1,length_sort 1,pass_tokens_max 40960 | 卡 L180 + provenance |

第三阶段的 `--data` 分别为:0.8B `evals/round15/joint/train.jsonl`、9B 同、4B `evals/round10/skills/train.jsonl`;27B 为私有 corpus(见 §2.7)。

### 3.3 硬件与时长(训练 compute,来自模型卡 Compute 节)

| 阶段 | 硬件 | 时长 | 备注 |
|---|---|---|---|
| 0.8B 基座 | 1× H100 | ~20 min | README 同(kev-0.8b.md#L295,README#L336) |
| 0.8B 日期 | 1× H100 | 9 min | kev-0.8b.md#L296 |
| 0.8B 文档+技能 | 1× H200 | 52 min(峰值 23.9 GB) | kev-0.8b.md#L296 |
| 4B 基座 | 1× H100 | ~56 min(峰值 24.6 GB) | kev-4b.md#L294 |
| 4B 日期 | 1× H100 | 9 min | 同上 |
| 4B 文档 | 1× H200 | 43 min | kev-4b.md#L295 |
| 4B 技能 | 1× H200 | 1.4 h(峰值 47.7 GB) | 同上 |
| 9B 基座+日期(v1) | 单 H100(时长卡未给) | — | kev-9b.md#L274 |
| 9B 文档+技能(v2) | 1× H200 | 2.6 h(峰值 74.1 GB) | kev-9b.md#L275 |
| 27B v2 全参 | 8× H200(FSDP2) | 16.2 h 训练时间 = **129 GPU-hours**(不含重启与评估) | kev-27b.md#L299 |
| 27B 权重平均 | CPU | ~6 min | kev-27b.md#L300 |

评估/服务校验:单卡 H100/H200/L40S(Modal),MLX 在 Apple M5(各卡 Compute 节)。8 卡 27B 的逐 step 速度参考:`docs/autoresearch.md#L207-213`(lr 2e-5 的 1-epoch skills delta ≈ 8.8 s/step;fp32 读约 9B 的三倍;locked read 需 `--timeout 14400 --memory-mb 131072`)。

### 3.4 Modal 用法(`modal_app.py`)

- 入口与部署:`KEV_GPU=H200 uv run modal deploy modal_app.py`;`uv run python -m kev.rounds launch experiments/rounds/r<N>.json` 每 study 一个 `::study`(间隔 60 s);`watch` 拉取+读分+写 readout;`pull`/`benchmarks`/`locked_test`/`evaluate`/`base_probe`/`smoke_base`/`interpolate`/`mirror` 等 entrypoint(`docs/autoresearch.md#L95-119`;`modal_app.py#L1-20`)。
- 容器:一 trial 一 GPU 容器;GPU 由 `KEV_GPU` 环境变量决定(**默认 H100**,免费层用 T4;`modal_app.py#L42`);镜像 Debian-slim + py3.13 + `uv sync --extra serve`,预装 `flash-linear-attention==0.5.2`、`triton>=3.7.1` 与 causal-conv1d wheel(DeltaNet 内核,`modal_app.py#L55-74`)。
- 资源:`run_trial` 4 CPU / 192 GiB 上限、timeout 28,800 s、`retries=0`(`#L103`);`run_full_trial` timeout 86,400 s、1 TiB ephemeral disk(FSDP2 需要,`#L110` + `kev/budget.py#L15-19`)。
- 卷:`kev-runs`(trial 输出)、`kev-hf-cache`(基座权重)、`kev-leases`(full-weight attempt 租约,心跳 60 s/过期 900 s;`kev/budget.py#L33-44`)。
- 预算闸门:`admit_study` 在启动前按 admission bound 拒绝超预算 study;**LoRA study 上限 $250 / 28,800 s;full-weight 上限 $1,000,每次 attempt ≤24 h,每个 trial ≤1+2 次 attempt**(`kev/budget.py#L22-32`)。
- 断点续训:full-weight attempt 超时由本地 `watch` 调 `modal_app.py::resume` 续(不从 Modal 重试,避免双重计费);快照 0.25/0.5/0.75 保留,27B 快照 ~154 GB/次(`docs/autoresearch.md#L150-175, #L207`)。

### 3.5 `--init_from` 用户微调路径(照做即可)

官方 skill(`skills/kev-finetune/README.md#L22-40`)把这条路径脚本化:
1. `workload.json` 描述决策(状态 + 三类问题);
2. `scripts/plan_size.py` 估算需要的记录数(3 问/条时约 1,000 条才能测出 +5 点);
3. 数据:自有标签 `convert_data.py` 或 LLM 生成 `generate_data.py`(gpt-4.1-mini ≈ $0.25/1,000 条);
4. `split_data.py` 按 state 切分;
5. `modal run scripts/kev_modal.py::train --data data/x --name x-v1 --init-from jaredpalmer/kev-4b`(打印 baseline vs fine-tuned、raw vs calibrated);
6. `modal deploy scripts/kev_modal.py`(Serve,scale-to-zero)+ TypeSafe SDK 指向 endpoint;`teardown --everything`。

同等价的仓库内命令(README `#L334-346` 为 0.8B/4B 首阶段;delta 用法见 `skills/kev-finetune/references/`)。已发布卡上给用户的推荐就是这一句:`kev.train --init_from jaredpalmer/kev-<size>`(每张卡 Intended uses)。

---

## 4. 校准(温度拟合流程与各 checkpoint 的温度)

### 4.1 发布温度(`scripts/calibrate_checkpoint.py`)

- 单一温度,最小化 NLL;写入 `head.pt["temperature_fit"]`,含 fit rows(各 rows 文件的 suite/partition/sources/题数)、方法、5-fold group-disjoint OOF 估计与 bootstrap 区间、以及(如手动指定)`--reason`。写回的值永远是全量 fit;serving/benchmark/Space 一视同仁按 T 出概率,`KEV_TEMPERATURE=1.0` 恢复 raw logits;argmax 与 T 无关。
- **MUST(round 21 起强制)**:拟合集必须是 held-out **数据集**,不能是训练语料自己的 calibration/development 分区。脚本会拒(除非显式 `--allow-in-distribution` 并记录);`kev.rounds validate/launch` 拒没有 `temperature` pool 的轮次。此前反面案例:round 19 的 SFT 臂按 sft-v1 development 拟合 T 0.955,breadth-v1 ECE 0.059;换成 held-out pool 后同 checkpoint 0.0085(`scripts/calibrate_checkpoint.py` 头部 docstring;`docs/autoresearch.md#L62-91`;PLAN.md #5)。
- 部署者路径:`kev/calibrate.py --rows runs/<name>/rows.json` 输出四个 arm(raw / shipped / workload(全量 in-sample)/ workload_oof(group-disjoint OOF))+ record-clustered paired bootstrap;只报不改模型(`kev/calibrate.py#L1-56`)。

### 4.2 各 checkpoint 温度值

| checkpoint | 发布 T | 拟合行数 / 池 | 备注 |
|---|---|---|---|
| Kev-0.8B | **2.35**(head.pt 2.3511) | stage-3 trial 的 decision-v7 development 1,264 题(**in-distribution**) | round 28 held-out 重拟 2.52(90% CI [2.19, 2.83]),因在训练 family 上更差(hard-v1 ECE 0.124 vs 0.112 等,超容差 0.005)未采纳;服务端可用 `KEV_TEMPERATURE=2.52` 手动切换(kev-0.8b.md#L84,#L243) |
| Kev-4B | **2.41**(2.4061) | 同上 1,264 题(**in-distribution**) | round 28 held-out 重拟 2.30(90% CI [2.05, 2.52]),Brier 差 −0.0001 [−0.0005, +0.0003],未采纳(kev-4b.md#L79,#L239) |
| Kev-9B v1 | 2.30 | v1 自己的训练分布 development 行 | — |
| Kev-9B v2 | **2.19**(2.1936;90% CI [2.05, 2.41]) | **held-out 池 648 题**:transfer-r3 calibration split 六公开源 + 两个 held-out 生成政策 family(448)+ MMLU-Pro 200 | 首个在 held-out 数据集上拟合温度的 9B;写入 head.pt 时因脚本无法列出合并文件的源而以 `--reason` 记录(kev-9b.md#L77,#L148) |
| Kev-27B v1 | 1.38 | 其 trial development 行 | 卡 tag `v1-lora` |
| Kev-27B v2 | **1.32**(1.3195;90% CI [1.20, 1.45]) | 同一 held-out 648 题池(两 parent 都没训过) | 5-fold group-disjoint CV:ECE 0.048(raw)→0.038(OOF)(kev-27b.md#L97,#L182,#L245) |

温度对准确率零影响,只动 Brier/ECE/coverage;"one temperature cannot reorder confidences" 是全系的公开限制(Kev 1.0 Known limitations,`docs/releases/kev-1.0.md#L54`)。

---

## 5. 评估与证据(套件存放、`runs/`、`claims.json`、模型卡规范、发布锁、autoresearch 循环)

### 5.1 冻结套件的存放

- git 内:`evals/<suite>/`(manifest + dev/test 分片;documents-v1 5 MB、hard-v1 5 MB、devtools-v1 12 MB 等小分区直接入库)。
- 大训练分区/私有 held-out:HF 数据集 `jaredpalmer/kev-suites`(公开,revision `cc4bac80`)/ `jaredpalmer/kev-private-evals`(私有);git 只留 manifest 与 sha256(`kev/suite.py#L32-41`)。README 徽章即指向 `kev-suites`(README#L8)。
- Kev 1.0 发布说明明确 "The documents-v1 and hard-v1 training partitions are in the `jaredpalmer/kev-suites` dataset, so the small models' training data can be fetched and hash-checked"(`docs/releases/kev-1.0.md#L40`)。

### 5.2 `runs/` 的组织

| 目录/文件 | 内容 | 范例 |
|---|---|---|
| `runs/<study>/<NN-trial-N>/{result.json, provenance.json, development/, transfer/, calibration/, snapshots/}` | 一次 trial 的结果、证明(config/代码哈希/suite 哈希/GPU/包版本/时间)、各分区的 rows 与报告 | `runs/r15-08b/00-trial-0/provenance.json` 里 `config_sha256`、`suite_sha256`、30 个 `kev/*.py`+`modal_app.py`+`pyproject.toml`+`uv.lock` 的 sha256、`git_commit`、`torch`、`gpu` |
| `runs/r<N>-readout/round<N>.json` + 表格 | 一轮的注册读分:每 arm × 每 criterion 的区间、verdict、失败项 | `runs/r28-readout/round28.json` |
| `runs/r<N>-verdict/<size>-<stage>.json` | 确认阶段(test/locked 只读一次)的裁决 | `runs/r27-verdict` |
| `runs/locked/<name>-ungated/` | locked read 的落地(`-ungated` 后缀规则见 `docs/autoresearch.md#L212-213`) | `runs/locked/kev-4b-r10-ungated/` |
| `runs/release/kev-<size>-r<N>.json` | 发布数字锁(`scripts/release_numbers.py --release <name>` 生成) | `runs/release/kev-1.0.json`(整份 1.0 发布记录) |
| `runs/fam-*/`、`runs/serve-*/`、`runs/mlx-*` | 家族横向读、服务基准、MLX 对等 | `runs/fam-breadth-test-report/` |
| `runs/<study>.spawn.json / .watch.json / autoresearch-sessions.jsonl` | 启动/守望状态与花费台账 | `docs/autoresearch.md#L150-154` |

每个数字都要求 "checkpoint + suite + partition + n + report path"(`docs/autoresearch.md#L186-190`)。

### 5.3 `docs/claims.json` 与 CI

- 1,019 条 claim 记录(实测条数),每条:`{"printed": 文档里逐字出现的字符串, "in": [必须出现该字符串的文件], "source": 仓库内证据文件, "path"/"paths"/"select"/"derive"}`;`scripts/verify_claims.py` 逐条检查"字符串在文档里出现"且"源值四舍五入后等于 printed",README/模型卡的数字由此在 CI 中可追溯(`scripts/verify_claims.py` docstring;卡片 Provenance 节指向它)。
- 证据文件本身(rows.json 等)也在仓库:`.gitignore` 保留"报告"而不是预测 dump(`docs/autoresearch.md#L188`)。

### 5.4 模型卡规范(四张卡同构)

- 结构固定:summary → details → intended/out-of-scope uses → how to use → training data → training procedure → evaluation → limitations → bias/risks → compute → provenance → citation(`docs/releases/kev-1.0.md#L42`: "All four cards follow one structure …")。
- 每张卡顶部 front-matter 用 Hub 可解析字段:`license: apache-2.0`、`base_model` + `base_model_relation`(`adapter`/`finetune`)、`datasets`(列出用到的公开数据集)、`metrics`、`model-index`(任务/数据集/指标值)。
- 每张卡必须给:backbone revision、adapter/head 参数数、context 与 validated context、T、语言、许可、Hub revision、评测方法学(dev 选型/test 只读一次/locked 一次/paired bootstrap 2,000 次、以**整条记录**为 resample 单位)、Compute 与 Provenance(指向 trial、readout、release numbers、Hub revision 与文件 sha256)。
- 卡的发布方式:卡文件即 Hub `README.md`;Kev 1.0 的做法是"仅卡提交(card-only commit)+ annotated tag `v1.0` 指向该提交",权重字节不变(`docs/releases/kev-1.0.md#L85`;`runs/release/kev-1.0.json` `hub` 节)。

### 5.5 Kev 1.0 发布锁

- 资产:`docs/releases/kev-1.0-assets.json`(每个 tarball 的 Hub repo/revision/卡文件/locked read/期望的 adapter+head sha256)+ `scripts/build_release_assets.py` **确定性**重建(相同字节,SHA256SUMS 两次构建一致)。
- GitHub release `kev-1.0`(tag 打在 PR #206 merge `6b719c3`):三个 tarball(`kev-0.8b/4b/9b.tar.gz`,46/131/172 MB)+ `SHA256SUMS.txt`;27B 因 51 GB 超 GitHub 2 GB/asset 上限,只放 Hub(`docs/releases/kev-1.0.md#L75-85`)。
- tarball 内容:权重 revision 下的 Hub snapshot(adapter、含温度的 `head.pt`、tokenizer、trial 的 `result.json`/`provenance.json`/`training_config.json`/`training_metrics.json`/log)+ 1.0 卡作 `README.md` + locked transfer-v4 read 作 `locked_test.json`;每个成员 mtime 固定 2026-10-01T00:00:00Z(SOURCE_DATE_EPOCH),`kev.serve` 据此报 release date(`docs/releases/kev-1.0.md#L75`;`runs/release/kev-1.0.json`)。
- Hub 上的锁:四个 repo 的 `v1.0` tag(tag 指向卡提交;权重 commit 见 §0 表);`@v1.0` 匿名加载验证:adapter/head/27B shard 哈希匹配、温度 2.35/2.41/2.19/1.32、smoke 读与 tarball logits 一致(PLAN.md L2690-2712 "Released: Kev 1.0")。
- 发布前逐轮确认:test 面板先读、locked 后读、"one read each, no exceptions",`confirm` 由人显式触发而不是自动(`docs/autoresearch.md#L111-114`)。

### 5.6 `docs/autoresearch.md` 的无人值守循环(怎么转)

1. **开头 30 分钟只读**:AGENTS.md、PLAN.md、`.agents/skills/kev-modal-study` 等、`kev/rounds.py`(docstring 即 spec schema)与 `kev/autoresearch.py`;工作区用 worktree + `research/<session>` 分支,每 commit 即 push;先读 `modal billing summary --json` 记 baseline(`#L10-29`)。
2. **预算规则**:每次启动前 `(metered_now − baseline) + 所有在跑任务的 admission bound < authorization`;留 ~10% 储备;每次读数带 UTC 时间;Modal workspace spend limit 只能从 dashboard 调(`#L30-42`)。
3. **注册一个 round**:PLAN.md 章节 + `experiments/rounds/r<N>.json` **同一 commit、先于任何训练或读**;新数据 = `evals/` 新目录 + manifest;若读校准指标必须注册 `temperature` pool(§4.1);`kev.rounds validate` 打印 `ok` 后才算注册(`#L44-93`)。
4. **端到端执行**:`modal deploy` → `kev.rounds launch`(每 study 一个 `::study`,60 s 间隔)→ `nohup watch`(轮询、每 study 只 pull 一次、每 arm 一批 reads,写 `runs/r<N>-readout/round<N>.json` 与表格;可重启)`#L95-119`)。
5. **能做什么/不能做什么**:可以写 spec/PLAN、建新冻数据、启动 studies/reads、改基础设施常量;**未经 Jared 明确同意不得**:发布/改 Hub、部署公开 endpoint、提 main/合 PR、改 `evals/` 下任何已冻结物或评测器(`kev/experiment.py` `EVALUATOR_FILES`、gates、`kev/metrics.py`、`kev/rounds.py` 的 paired 读)、`--allow-test`/跑 locked、把 Jev 或闭源模型输出放进训练数据、本地训练、删 checkpoint/snapshot(`#L121-145`)。
6. **韧性**:状态文件 `runs/<session>-state.json`(runs/ 默认 gitignore,`git add -f`);detached job 活过本地退出;watcher 是本地进程,需 `nohup`+`caffeinate`,可重启;full-weight 超时续训由 watcher 发起;时网断线先 `modal volume get` 拉目录再决定是否重启(`#L150-175`)。
7. **报告**:每轮在 PLAN.md 记注册、读分表、确认结果与 verdict(负结果也记);PLAN "Where we stand"/"What we have learned"/Record 表更新;session 小结记花费、pending、incident、至多三条 next steps(`#L177-190`)。

`kev/rounds.py` 把 round 变成数据:R/S/A spec schema(studies/reads/parents/arms/rule/confirm/temperature/drop_ids…),`launch/watch/readout/confirm` 四段式引擎;比较一律 served-vs-served、paired record-clustered bootstrap(2,000 resamples, seed 0)(`kev/rounds.py#L1-60`)。多轮连续跑用 `kev.autoresearch session <specs...> --spend-start --spend-cap`(校验→启动→守望→readout,到 cap 停,**从不自动跑确认**)(`docs/autoresearch.md#L115-119`)。

---

## 6. 成本(美元与 GPU 小时量级)

### 6.1 单价(Modal,仓库内自带的价目)

- GPU:`H100 $3.95/h`、`H200 $4.54/h`、`B200 $6.25/h`、`T4 $0.59/h`;CPU `$0.0473/core-h`;内存 `$0.008/GiB-h`(按 limit 计;磁盘按内存 20:1)(`kev/budget.py#L4, #L45`)。
- 容器级 hourly_rate(trial 资源:4 CPU、192 GiB 上限):单 H100 ≈ **$5.675/h**、单 H200 ≈ **$6.27/h**、8×H200 ≈ **$41.17/h**(PLAN.md L1608 与 `kev/budget.py#L69-75` 的算式一致;round 28/29 用的 "H100 at $5.675/h")。
- 另有一套 README 服务用价目(不含 CPU/内存):L4 0.80、L40S 1.95、H100 3.95、H200 4.54、B200 6.25 USD/h(`README.md#L408-414`)。

### 6.2 单次训练(一个 checkpoint 的一个阶段)

| 阶段 | GPU-hours | 按容器价的 $\approx$ | 出处 |
|---|---|---|---|
| 0.8B 三阶段合计 | ≈ 0.33 + 0.15 + 0.87 ≈ 1.4 GPU-h | ~$8 | 卡 Compute(§3.3;H100 20min + 9min,H200 52min) |
| 4B 四阶段合计 | ≈ 0.93 + 0.15 + 0.72 + 1.4 ≈ 3.2 GPU-h | ~$18 | 同上 |
| 9B 文档+技能(单阶段) | 2.6 GPU-h(H200) | ~$16 | kev-9b.md#L275 |
| 27B v2 全参一个 epoch | **129 GPU-h**(8×H200 × 16.2 h) | 8 卡容器 16.2 h × $41.17 ≈ **$667**;裸 GPU $4.54 × 129 ≈ $586 | kev-27b.md#L299 |
| 27B v1 LoRA trial | ~1.8–2.1 GPU-h/个 | 3 个 trial 实测 ≈ **$40**(110/114/124 min,卡 archive PLAN "Spend") | archive PLAN.md(Round 6 B1 段) |
| 4B LoRA trial(早期) | ~0.7–0.8 GPU-h | ≈ **$3.5–4.5**/trial | archive PLAN.md "Compute and spending"(2026-09-20) |
| 12 个小 delta(早期) | — | ≈ $1.5/个 | archive PLAN.md night-2 段 |

用户侧微调(skill):**400 条数据 ≈ 12 min、≈ $1 一跑(Kev-4B、单 H100)**;1,000 条约 15 min;`SKILL.md` 写 "about $1-3 per run";数据生成 gpt-4.1-mini ≈ $0.25/1,000 条(`skills/kev-finetune/README.md#L16-22`、`SKILL.md#L54`)。

### 6.3 一轮完整研究(round)

| 轮 | 内容 | 实际/预期 Modal 花费 | 出处 |
|---|---|---|---|
| 轮 4(2026-09-22,12 个廉价杠杆) | LoRA 小 trial + 读 | **+$58.37**(在 $60 信封内;期末 metered $464.31) | archive PLAN.md L575 |
| 轮 5(2026-09-22) | 长状态+soft target,全尺寸 | **+$110**(期末 metered $557.31) | archive PLAN.md L103 |
| 轮 6(2026-09-23 过夜,22 个 delta trial) | 一夜多项 hill-climb(长状态/soft target 变体) | **+$312**(02:03Z $543.10 → 08:45Z $855.19) | archive PLAN.md L421 |
| Night 3(2026-09-23 晚 → 09-24,轮 7-18 + 27B B1 v2) | 一夜内产出 0.8B/4B/9B v1 与 27B v1 的 delta | **+$291.69**($1,085.32 → $1,377.01);即全家四尺寸的 delta 工作量一夜 ≈ $292 + 前期基座 trial | archive PLAN.md L15, L123 |
| 轮 19(27B 全参首测) | 3 arms × 8×H200 + 读 | **+$834.75**;预算:bound $2,347、预期 $800-950、程序上限 $2,000 | PLAN.md L360, L175-178 |
| 轮 20(post-hoc 温度+插值,无训练) | 74 reads + 2 插值 | **+$54.87**(bound $472.04、预期 ~$110) | PLAN.md L439, L497 |
| 轮 21(27B 全参,启动即 OOM) | 失败,无读 | ≈ **$135** | PLAN.md L1053, Record 表 |
| 轮 22(27B 全参,一键 1 arm、4 候选) | 3 次 attempt + 68 reads + probe | **+$903.36**(study bound $987.99;预期 ~$947) | PLAN.md L931-955 |
| 轮 23(post-hoc 6 个 blend) | 2 插值 + 每候选 ~$30 读(16 reads) | **+$190.12** 实测($3,787.06 → $3,977.18;规则阶段 ~$175 + 确认 ~$15,确认 bound $213.02) | PLAN.md L1608-1618, L1842 |
| 轮 25(27B 全参连训,2 studies) | 4h/attempt 设计,8 候选读 | 当夜 **+$494.5**;单 study 预期 ~$138 | PLAN.md L2063-2078 |
| 轮 26(27B 全参,1 study,4 候选) | — | 当轮 **+$226-257**(单 study 预期 ~$106) | PLAN.md L2376-2392 |
| 轮 28+29(post-hoc,120 reads) | 0.8B/4B/9B 温度与上下文 | 两轮合计预期 **≈ $185**(bound $588,cap $300) | PLAN.md L3050-3068 |
| 单次 read(评估一个 checkpoint × 一个 suite) | ~15 min 典型 | 预期 ~$2-7/read;1,800 s bound $2.84(H100)/$6.27(H200);locked/longdoc 用长 timeout | PLAN.md L439, L3050-3068 |

**全项目量级**:Modal 计费从 $392(2026-09-22)走到 **$4,633.77**(2026-09-30 注册读数,含工作区其它 app),即四个尺寸的完整研究 + 发布 ≈ **$4.6K 量级**;最高一夜 ceiling $5,980(PLAN.md L169-178, L3068)。Modal 官方赞助 $5,000 额度(more on request)。另有:**AI Gateway 上合成数据花费 $1,666**(对 $2,460 额度,发生在 round 19 注册前,属数据生成不属训练);Jev 参考读全程仅 $0.13(`PLAN.md` L178, L301, L360)。

### 6.4 一个尺寸"发布"的代价

- 发布动作本身几乎免费:Kev 1.0 = 四次卡-only Hub 提交 + tag + 一次本地确定性资产构建 + GitHub release,**无训练、无新读**(`docs/releases/kev-1.0.md#L87-114`)。Kev-27B v2 的发布(从 volume 复制 51 GB、上传、一次 ~10 min 的 H200 读验证)约 **$5.66**(PLAN.md L1842 附近 "$3,971.71 → $3,977.37")。
- 贵的部分是"到一个可发布 checkpoint"的路径:小尺寸单轮 delta 约 $1.5-10/个 trial;27B 全参一轮 **$0.5K-0.9K**;确认阶段(14 reads tests + locked + serving check)每候选 ~$60 预期 / ~$213 bound(PLAN.md L934-949;`docs/autoresearch.md` 的预算规则)。
- 开销上限护栏:LoRA study $250/8h、full-weight study $1,000/24h attempt、`MAX_SNAPSHOTS=8`、27B 快照 ~154 GB/trial 进 1 TiB 卷的硬上限(`kev/budget.py#L11-25`)。

---

## 7. 照做清单(把 Kev 1.0 从零复现的最短路径)

1. 环境:`git clone jaredpalmer/kev && uv sync`;GPU 走 Modal(`uvx modal setup`;`KEV_GPU=H200 uv run modal deploy modal_app.py`)。
2. 取数据:`evals/` 里的 suite 自动从 `jaredpalmer/kev-suites` 拉大分片并校验 sha256;私有套件(kev-private-evals)拿不到就跳过对应面板。
3. 第一阶段(0.8B 示例):
   `uv run python -m kev.train --suite evals/v7/decision-v7 --base Qwen/Qwen3.5-0.8B-Base --base_revision dc7cdfe2… --epochs 2 --lr 1e-4 --batch 8 --dtype bf16 --p_none_pair 0.25 --device cuda --out runs/kev-0.8b`(README#L336-338;4B 换 lr 5e-5/batch 4/accum 2/checkpointing 1)。
4. 后续阶段全部 `--init_from` 上一阶段 checkpoint + `--data evals/night2/dates_unknowable.jsonl`(后接 `evals/round15/joint/train.jsonl`)+ `--replay` 2,000/6,000/10,000,超参见 §3.2 表逐字抄 config。
5. 27B:基座换 `Qwen/Qwen3.8-27B@1d4bf0f2`,语料换私有 corpus(`sft-v2-r22` manifest 可复刻十成分),8 卡 `torchrun --standalone --nproc_per_node 8 -m kev.train --full_ft 1 --weights_dtype bf16 --dtype bf16 --checkpointing 1 --shared_prefix 1 --length_sort 1 --pass_tokens_max 40960 --max_state 32768`(超参见 §3.2);训完用 `scripts/interpolate_checkpoint.py` / `scripts/checkpoint_average.py` 与 v1 做 0.85/0.15 平均。
6. 温度:`scripts/calibrate_checkpoint.py --run <ckpt> --rows <held-out rows> --rows <mmlu_pro rows>` 写入 head.pt(必须 held-out 数据集)。
7. 评测:`uv run python -m kev.benchmark --run <ckpt> --suite evals/v4/transfer-v4`(dev)→ 按注册规则读一次 test(`--allow-test`)→ locked(`modal_app.py::locked_test`)。所有数字落 `runs/`,登记 `docs/claims.json`,跑 `scripts/verify_claims.py`。
8. 发布:`kev.publish`(卡文件作 README)+ Hub tag + `scripts/build_release_assets.py`(确定性 tarball + SHA256SUMS)。

---

## 8. 来源、不可达项与置信度

**一手来源(全部可在 `/tmp/kev-src` 或经 `gh api` 复查)**
- `docs/model-cards/kev-{0.8b,4b,9b,27b}.md`(完整配方、评测、Compute、Provenance)
- `docs/releases/kev-1.0.md`、`docs/releases/kev-1.0-assets.json`、`runs/release/kev-1.0.json`
- `kev/train.py`、`kev/model.py`、`kev/data.py`、`kev/suite.py`、`kev/budget.py`、`kev/rounds.py`、`kev/calibrate.py`、`scripts/calibrate_checkpoint.py`、`scripts/build_release_assets.py`、`scripts/verify_claims.py`
- `modal_app.py`、`docs/autoresearch.md`、`PLAN.md`(含 tag `research-archive-2026-09-24` 存档版)、`README.md`、`docs/claims.json`
- 各 trial 的 `provenance.json`(`runs/q35-08b/02-trial-2`、`runs/night2-*`、`runs/r15-08b`、`runs/r10-skills`、`runs/r18-9b`、`runs/r6-27b-v2`、`runs/r22-27b-lr2e6` 等)、各 `evals/*/manifest.json`

**不可达/未独立核验(显式列出)**
- `huggingface.co` 与 `raw.githubusercontent.com` 在本环境不可达:权重本身、Hub 卡的显示、`kev-suites`/`kev-private-evals` 数据集内容**未直接读取**;本文所有 Hub revision/哈希/温度均转引自仓库内记录(模型卡、`docs/releases/kev-1.0-assets.json`、`runs/release/kev-1.0.json`),这些记录之间互相一致(如同一 adapter sha256 同时出现在卡、assets spec 与 release 记录)。
- 私有语料与私有 held-out 套件(`kev-private-train`、`kev-private-evals` 内的 breadth-v1/tasksource/longdoc/ood 等)只有 manifest;其文本无法核验,规模与构成以 manifest+卡为准。
- `evals/round6/b1v2/train.jsonl`(27B v1 训练集)与 `evals/round15/joint/train.jsonl` 的 payload 不在 git(从公开 `kev-suites` 镜像拉取,本次未拉取);`evals/sft-v2-r22/*.jsonl` 的 payload 在私有镜像,只验证到 manifest 层。
- `modal_app.py` 的云端行为(admission/lease/续训)按代码与文档为准,未实际部署运行。

**置信度与未决问题**
- 高置信:配方数字(超参、数据构成、温度、发布哈希)因为全部来自 trial provenance、manifest 与发布记录三层互证;成本数字因为每个都有 "bound/expected/actual" 三元组。
- 中置信:9B 基座/日期阶段的时长(卡只给硬件不给分钟);4B 文档阶段(r8)的 trial 目录缺 `provenance.json`,超参取卡面。
- 未决:(a) `sft-v2-r22` 的十成分具体条数与去污染明细只在私有 manifest 里;(b) `kev-suite` 各面板在 Hub 上的实际字节未验证(仅 sha256 记录);(c) "一轮完整研究"没有单一平均价——同一轮从 $55(post-hoc)到 $900(27B 全参)相差一个量级,取决于是否训练、尺寸与候选数,决策票如需单一数字建议按 §6.3 分层引用。
