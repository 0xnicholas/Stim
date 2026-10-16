# 决策模型生态盘点:TypeSafe/Jev、AutoJev、Decision Index 与同类

> 研究票:[#6](https://github.com/0xnicholas/Stim/issues/6) · 上游地图 [#1](https://github.com/0xnicholas/Stim/issues/1)
> 用途:供定位票 #2 与评估体系票 #12 使用。**本文件只做盘点,不含「Stim 应该怎么做」的建议。**
> 主撰写日期:2026-10-16。所有时间戳、版本号、分数均为该日抓取的公开数据。

---

## 0. 摘要(先读这段)

1. **Jev 是闭源托管模型,不是开源项目。** TypeSafe 只开放 HTTP API(`POST /v1/systemone`)、Python/JS SDK、agent skill 和文档;权重与 RLCD 算法均未公开。官方公开价格 **$42 / Btok**(即 $0.042 / Mtok),**输出 token 不计费**。([docs.typesafe.ai/models](https://docs.typesafe.ai/models.md))
2. **生态规模远大于本票的假设。** 截至抓取日,Hugging Face 上决策模型家族的公开权重以百计,Decision Index 收录 **167 个 entrant**,Benchmark Heaven 的 JevBench 收录 **94 个 system**;GitHub 上仅 "awesome-jev" 类清单就有十几个,其中 `yibie/awesome-jev` 达 2,367★。
3. **驱动复现潮的是一篇反向工程长文,不是官方规范。** Archer Hume《Jev's Architecture Unmasked》(2026-09-17)用 **10,000 次 API 调用**推断出 Jev 的计算形状(共享 state 编码 + 隔离 question 分支 + 概率 readout),并明言「TypeSafe 拒绝公开研究,所以我来(尽力)」。`jaredpalmer/kev`(9,214★)的 README 直接写着自己是「基于该文描述的架构」。这是生态里最重要的单一事实。
4. **AutoJev-27B 已改名并并入了 Perplexity 的模型线。** 它最初以 `denis-pplx/autojev-27b` 发布(1×H200、全参 SFT、73,000 条样本、286 步、发 checkpoint 200),现在是 `perplexity-ai/pplx-decider-v1-27b` → `v1.1`。Decision Index 里同一条记录(engine 名仍是 `autojev-27b`)显示名已改为 pplx-decider,**0.3.x 版面板中已不再出现 AutoJev 这个名字**。
5. **公开复现的做法高度同质:全是 SFT / 蒸馏 / 推理技巧,没有一个公开的 RLCD 实现。** Decision Index 把 entrant 分为 full fine-tune / LoRA / LoRA+head / head / inference technique 等类;没有一类对应 TypeSafe 声称的训练路径。
6. **两个互相独立的公共评测 + 一个同名但完全不同用途的第三方基准**,构成本领域的事实评分层:Decision Index(社区维护、含不可训练的 private 部分)、JevBench by Benchmark Heaven(sealed 题、含成本/延迟轴)、以及 metamorphic coherence 的 JevBench(只测一致性的同名项目)。
7. **校准是公开模型最普遍的短板,而非准确率。** JevBench 榜上排在 Jev 前面的模型(如 Quyet-1.0-Large)主要靠 intelligence 取胜,calibration 轴 Jev 仍在前列;而 Kev 的 PLAN.md 记录了自家 LoRA 路线在 `scienthoon` 上反复失败、AutoJev 在 89 条 TypeSafe 题上打平而在 `hard-v1` 上领先 4.9 pp。
8. **最大的空白不是"再造一个模型",而是:训练数据不随权重发布、RLCD 无公开实现、交互/游戏类评测至今未启动、长上下文只在个别模型上被验证、以及评分口径与元数据的可核查性。**

---

## 1. 方法与证据分级

| 等级 | 来源类型 | 本文件中的例子 |
|---|---|---|
| **一手** | 官方文档/官方 API 响应/官方模型卡/官方仓库文件 | `docs.typesafe.ai`,HF 模型卡,`denis-pplx/autojev` 仓库内 `configs/training.json`、`assets/provenance.json`,`jaredpalmer/kev` 的 `PLAN.md` 与 `runs/autojev-h2h/report.json` |
| **一手(经镜面)** | 同一份 HF 内容,经 `hf-mirror.com` 代理读取 | 所有 `hf-mirror.com/.../raw/...` 与 `hf-mirror.com/api/...`;**内容与 huggingface.co 同源**,仅传输路径不同 |
| **社区一手** | 项目作者自己发布的评测/长文/榜单 | Decision Index(`multimodalart/jev-decision-index`)、JevBench( Benchmark Heaven)、Archer Hume 长文 |
| **不可达/未使用** | 抓取环境不可达的来源 | 见 §8 清单 |

抓取环境实况:直连 `huggingface.co`、`raw.githubusercontent.com`、`registry.npmjs.org` 不可达;`github.com` 经 `gh` CLI 与 `curl` 可达;`hf-mirror.com` 可达并用于全部 HF 读取。

---

## 2. TypeSafe 与 Jev

### 2.1 产品脉络

Jev 是 TypeSafe 的旗舰模型,也是所谓「**第一个 System One 模型**」。System One 的定位不是聊天,而是「让代码能直接消费的快速结构化判断」:给定一份 **state**(字符串/对象/数组)与若干 **typed questions**,返回类型化答案与概率分布,**不做文本生成**(官方原话:「No text generation, no parsing」)。([Introduction](https://docs.typesafe.ai/introduction.md))

**问题原语(primitives)**([primitives.md](https://docs.typesafe.ai/primitives.md)):

| 类型 | 目标 | 返回 |
|---|---|---|
| `Choice` | 从列表选一项 | `choice`,`probabilities`,`confidence` |
| `Score` | 按有序描述性等级打分 | `score`,`probabilities`,`confidence` |
| `Noul` | 判断陈述是否为真 | `noul`(0–1) |
| `Box` | 在图中框出目标(图像能力) | box 四边坐标(占宽高比例) |

一次请求可混合任意数量的原语,**并行且相互隔离**地针对同一 state 求值;官方称增加问题几乎不增加响应时间,且不会产生 context rot。官方设计主张是「原子问题 + 在代码里组合」([Introduction](https://docs.typesafe.ai/introduction.md))。

### 2.2 接入

- **端点**:`https://api.typesafe.ai/v1/systemone`,`Authorization: Bearer $TYPESAFE_API_KEY`。([models.md](https://docs.typesafe.ai/models.md))
- **模型与别名**:`jev-latest` → `jev-1.13.0`;`jev-preview` → `jev-1.14.0`(最新版,2026-10-14 起可用)。每个响应的 `model` 字段回传实际应答的版本化 ID。官方明确提醒:alias 会随发版移动,**答案可能在代码不变的情况下改变**;若已按某版本调过阈值,应直接指定版本 ID。([models.md](https://docs.typesafe.ai/models.md))
- **上下文**:默认 **per-request 64k / per-question 32k**;发送 `TypeSafe-Danger-Zone-Long-Context: true` 头可提升到 **262k**(两档同价,但官方声明更慢、更不准,只应在必要时开启)。([models.md](https://docs.typesafe.ai/models.md))
- **定价**:`$42 / Btok`(输入),`$0.042 / Mtok`;**输出 token 免费**。Jev 1.13 与 1.14 同价。([models.md](https://docs.typesafe.ai/models.md))
- **限流**:按组织配置,以 tokens/s 与 requests/s 计量,超限返回 `429`;官方 SDK 默认按 `retry-after` 重试。更高额度需走 custom/enterprise(`sales@typesafe.ai`)。([models.md](https://docs.typesafe.ai/models.md))
- **SDK**:Python(`typesafe_sdk`,含 async/sync 客户端、`RetryPolicy`)与 JavaScript/TypeScript 两套。([sdk.md](https://docs.typesafe.ai/sdk.md))
- **Agent skill**:官方 `typesafe-ai/skills`(GitHub ★2,713),可 `claude plugin marketplace add typesafe-ai/skills` 或 `npx skills add typesafe-ai/skills --skill typesafe-ai` 安装。([quickstart](https://docs.typesafe.ai/introduction/quickstart.md))
- **Playground / console**:`console.typesafe.ai` 取 API key;文档页提供浏览器 Playground。([quickstart](https://docs.typesafe.ai/introduction/quickstart.md))

### 2.3 训练路线与官方叙事

TypeSafe 把自己的路线称为 **RLCD(Reinforcement Learning for Calibrated Decisions)**,并把它与 RLHF / RLVR 并列为「预训练模型适配的第三条路」。关键主张([AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md)):

- 模型不生成文本,而是返回**决策与概率**;概率越高,正确答案出现的频率应越高(`p=0.2` 的事件约 20% 发生)。
- RLHF 会奖励谄媚与"听起来自信"的幻觉,并造成 **mode dropping**;人类偏好与机器可信度是**两个不同的优化目标**。
- 公司叙事核心是「**Machine Native Intelligence**」:假设大规模自动化的交互结构约为 **99% 机器对机器 / 1% 人对机器**,因此设计目标从"读起来好"转向"在软件里行为可预测"。
- 官方称 RLHF 由 **Diogo Almeida**(TypeSafe 联合创始人)参与共同发明。([AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer.md))
- **Jev 不用客户数据做微调或 LoRA**:「Jev is not fine-tuned or LoRA-adapted with customer data…the same weights serve every account」,定制只能通过请求塑造。([models.md](https://docs.typesafe.ai/models.md))

官方还主动发布了自查的失败模式清单 [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13.md)(最后审阅 2026-10-02),列 9 类锯齿:字面理解、数学与数字、日期比较、多跳间接推理、充斥无关细节的长 state、对抗性内容、矛盾指令、选项顺序敏感、以及"需要生成"的任务。官方对多类的建议是「**把算术留在代码里**」「拆分问题后在代码中组合」——这与 §7 的空白清单直接相关。

### 2.4 围绕它形成的社区:反汇编与复现文化

这是本票中信息密度最高的一节。

**(a) 反向工程长文是生态的宪法。**
Archer Hume,《Jev's Architecture Unmasked》,2026-09-17,28 分钟阅读,方法自述为:**探测 TypeSafe API 10,000 次**,通过不同上下文长度下的延迟伸缩、问题重排序等特征寻找签名,再结合公开文档与先行研究,推断架构。文中明确写道:「There's just one problem, it's not open weight, and TypeSafe refuses to share their research… So I will (try my best).」([archerhume.com/posts/jevs-architecture-unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/))

其推断的七条架构主张(作者自己标注了"已发布证据 / 已观测 / 已推断"三级确定性):

| # | 主张 | 作者对确定性的态度 |
|---|---|---|
| 1 | **End inference with a readout**:用预测头替代 decode loop | 已发布证据支持(官方启动公告:「Jev outputs all probabilities in parallel instead of autoregressively generating by token」);作者另观察到 API 仍会回传 `output_tokens`,但对 noul 恰好等于「4 个共享 token + 每个答案 15 个 token」,说明它记的不是生成 |
| 2 | **Share the state, isolate the questions**:state 编码一次,Q 个 question 各建分支,Q 个分支之间**不能互相 attention** | 结构由 API 语义支撑 |
| 3 | **A causal backbone**:复用因果 transformer 的 prefill,**不等于 diffusion** | 「None of this requires diffusion」 |
| 4 | **Let the options interact before choosing** | — |
| 5 | **Train the distribution, then calculate confidence**:置信度由分布算出,而非生成的文本 | — |
| 6 | **Sparse capacity**(疑似稀疏 MoE) | 作者自己标为「The sparse backbone is the least certain part」 |
| 7 | **Schedule branches as a batch, not a conversation** | — |

其收益论证:S 个 token 的 state × Q 个 question,分开请求要处理约 Q·S 次 state token,共享后降到 S;question 仍须 attend state,但不必重复重建 state 表示。并给出未被官方公开承认的工程含义:`p(urgent) > 0.1` 这类阈值规则只有在概率可信时才有意义。作者另设「What would change my mind?」一节(本文件未逐条核读)。

**(b) Kev 直接以该文为架构依据。**
`jaredpalmer/kev`(GitHub ★9,214 / fork 622,2026-09-17 创建,Apache-2.0)描述为「Jev-like family of decision models built on top of Qwen3.5/3.8 you can train and run on your own」,README 原文:「Kev is a family of small decision models built on Qwen3.5 and Qwen3.8 and **based on the architecture described in [Jev's Architecture Unmasked]**」,并声明 API 与 TypeSafe System One 兼容,「**the TypeSafe Python SDK works against a Kev server unchanged**」。([kev README](https://github.com/jaredpalmer/kev);本地浅克隆 `/tmp/kev-src`,commit `1f04cfa`)

**(c) HF 侧的复现追踪器。**
Decision Index 的 News 页把自己定位为复现追踪器,把条目按类型着色:**Decoding**(在 stock 模型上做并行约束解码,无新权重)、**Diffusion**(文本扩散模型跑「Jev mode」)、**Trained**(类 Jev 打分头与微调)、**Prior art**、**Explainers**;条目按 trending 分排序(♥likes on X + 5×GitHub stars + 8×Hub likes + views÷500,再加按天减半的 recency credit)。并明确列出「**仍未开源的部分**:TypeSafe 的权重、RLCD 算法、以及任何公开模型匹配 Jev 校准主张的证据」。(Decision Index Space README)

**(d) 清单类仓库的规模与重复度。**
`gh api search/repositories` 抓到的 "awesome-jev" 类清单包括:`yibie/awesome-jev`(★2,367)、`v-modal/awesome-jev-tools`(★742)、`logicrw/awesome-jev-projects`(★683)、`AnotiaWang/awesome-decision-models`(★647)、`kydlikebtc/awesome-jev`(★601)、`AbdelStark/awesome-typesafe-jev`(★593)、`cobanov/awesome-jev`(★558)、`OmniJev/awesome-jev-gallery`(★507)、`walidboulanouar/awesome-jev-use-cases`(★432)、`hellogumbo/awesome-jev`(★221)、`valentynkit/awesome-jev-typesafe`(★193)、`kraayenjon/awesome-jev`(★191)。**同名清单大量重复**,是热度信号而非可靠的索引。另有中文教程 `datawhalechina/jev-cookbook`(★421)。

**(e) HF 侧的体量。**
`hf-mirror.com/api/models?search=jev&limit=100` 返回满 100 条(触顶),其中约半数是与决策模型无关的同名巧合(`jeveuxaider/*`、`Jevvan123/*` 等);明确属于决策模型家族的有 **约 50 个**可见仓库,包括 `autotrust/JEV-*`(5 个)、`alibiserikbay/JevK5*`、`chaoliangUNSW/Jev-Style-*`、`ZefanCai/Open-Jev-*`、`Maincode/matilda-jev-*`、`shgao/rsi-jev-*`、`berkay123001/turnike-jev-*`、`OmniJev/*`、`SimpleJev/JevAny-*` 等。`api/datasets?search=jev` 与 `api/spaces?search=jev` 同样触顶 100。

---

## 3. AutoJev-27B(denis-pplx)

### 3.1 做法

**一手来源**:HF 模型卡 `denis-pplx/autojev-27b`、仓库 `github.com/denis-pplx/autojev`、仓库内 `configs/training.json` 与 `assets/provenance.json`、`assets/results.json`。

- **定位**:多模态决策模型,基于 **Qwen3.8-27B**(base revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`),对给定选项在**每个问题一次前向**内返回概率;提供 TypeSafe 兼容 API 与浏览器 playground。自述为「**Independent implementation inspired by Jev**」。([模型卡](https://huggingface.co/denis-pplx/autojev-27b))
- **构建方式(值得单独记)**:卡面第一句是「Built with autonomous agents, from research and data generation to training, evaluation, and deployment. The user set the goals and refined the scope; agents executed the work.」——即**由自主 agent 完成研究、数据生成、训练、评测与部署**。
- **训练配方**(`configs/training.json`,`training_mode: "full_retained_weights"`,`objective: "cross_entropy"`):
  - 1×H200,**全参 SFT**(非 LoRA),**73,000 条唯一训练样本**,**286 updates**,发布的是 **checkpoint 200**。
  - 超参:`epochs 1`,`seed 20260920`,`lr 2e-6`,`weight_decay 0.01`,`batch_size 32`,`effective_batch_size 256`,`token_budget / max_length 8192`。
  - 校准:**单独**拟合一档标量温度,`selected_temperature = 2.207568021892729`,拟合约 3,500 行;监控面板 4,700 行。
  - 代码与数据:**MIT(代码)/ Apache-2.0(权重)**。模型卡明写「**The exact curated training corpus is not bundled.**」`provenance.json` 只列出 sanitize 后的研究源路径与 SHA-256,不含原始文本。
- **工程产物**:仓库 `src/autojev/` 含 `train.py`、`sft_pipeline.py`、`sft_synthetic.py`、`synthetic.py`、`sft_numeric.py`、`model.py`、`evaluate.py`、`report.py`、`server.py`、`data.py`、`playground.html`;`uv run autojev-serve` 启动,`POST /v1/systemone` 支持 `choice`/`noul`/`score` 与可选 base64 `images`。([无 hf-mirror 可达的 GitHub 内容,已用 `gh api` 读取仓库树](https://github.com/denis-pplx/autojev))
- **仓库活跃度**:★132 / fork 15 / MIT / 创建 2026-09-19 / 最后 push 2026-09-22 / open issues 0。**发布后即基本停止更新**。

### 3.2 自报结果

模型卡给出的三方对比(全部为 4,700 行面板上的一致口径):

| 模型 | Overall accuracy ↑ | ECE ↓ | Brier ↓ |
|---|---:|---:|---:|
| Qwen3.8-27B(基线) | 69.83% | 0.06483 | 0.40834 |
| **AutoJev-27B** | **84.60%** | **0.04282** | **0.22027** |
| Jev | 82.79% | 0.05274 | 0.25400 |

`assets/results.json` 中 Jev 一侧记录的模型为 `typesafe/jev-1.13-20260917`,完成于 2026-09-21,4,700 行上 accuracy `0.8278723`、ECE `0.052739`、Brier `0.254005`、NLL `0.652450`,与卡面一致。同一文件另记录 `calibration_fitted_on_panel: false`、`pretraining_exclusion_known: false`、`statistical_significance_claim: false`——即作者**主动声明**基线模型是否在面板上预训练**未知**、且不作显著性主张。

**注意口径差异**:AutoJev 卡面用的是自己的 4,700 行面板,Jev 数字来自作者自测;这与后文 Decision Index / JevBench / Kev 的独立口径**不可直接比较**。

### 3.3 与 Kev 的对比结论(head-to-head)

**一手来源**:`jaredpalmer/kev` 的 `PLAN.md`「External head-to-head」段,以及归档文件 `runs/autojev-h2h/report.json`(经 `gh api ... ?ref=research-archive-2026-09-24` 读取;浅克隆工作树中该目录不存在)。方法自述:**协议在任何 AutoJev 读数之前写好,全部 report-only、不设门槛**;AutoJev 由其**自己的 server** 服务,Kev-27B 在其随附的 `T 1.38` 下服务(脚本首版误用 trial 内拟合的 1.19,当天修正,准确率不变);只在**共享问题**上配对,且只用 development 或 public 分片。

| 套件(题数) | AutoJev | Kev-27B | Jev | AutoJev − Kev,acc [95%] | ECE AJ / Kev |
|---|---:|---:|---:|---|---|
| transfer-v4 dev (656) | 0.863 | 0.848 | 0.857 | +1.5 [−0.6, +3.7] | 0.041 / 0.043 |
| hard-v1 dev (1,083) | 0.782 | 0.733 | 0.777 | **+4.9 [+2.4, +7.3]** | 0.103 / 0.047 |
| devtools-v1 dev (1,072) | 0.708 | 0.702 | 0.715 | +0.6 [−0.7, +1.8] | 0.105 / 0.096 |
| documents-v1 dev (920) | 0.877 | 0.862 | 0.868 | +1.5 [−0.1, +3.3] | 0.015 / 0.088 |
| SemIf (144) | 0.993 | 0.972 | 0.965 | +2.1 [0.0, +4.9] | 0.048 / 0.068 |
| scienthoon (873) | 0.769 | 0.796 | 0.753 | **−2.7 [−4.9, −0.6]** | 0.071 / 0.044 |
| WANLI-v2 (1,002) | 0.764 | 0.745 | – | **+2.0 [+0.3, +3.7]** | 0.082 / 0.070 |
| TypeSafe (89) | 0.865 | 0.865 | – | +0.0 [−7.4, +8.1] | 0.082 / 0.057 |
| transfer-v9 dev (1,046) | 0.812 | 0.822 | 0.854 | −1.1 [−3.0, +0.9] | 0.045 / 0.050 |

Kev PLAN 的总结:**宏平均准确率 0.826 vs 0.816**;Brier 在 9 个套件中 **7 个** 是 AutoJev 更好。JevBench public:AutoJev 0.870(hard 0.739,hard ECE 0.075)vs Kev-27B 0.866(0.721,0.128),在 111 条 hard 题上配对 **+1.8 pp [−3.6, +7.2]**。

Kev PLAN 由此得出的(其自己的)结论有一条对定位票直接有用([PLAN.md](https://github.com/jaredpalmer/kev) 第 220 行附近):**「Full-weight SFT on broad data edges out our LoRA recipe on the same base」**——但作者同时明确标注这是**非受控比较**,因为 AutoJev 在**方法**(全权重)与**数据**(73k 广义合成决策 vs Kev 较窄的混合 + replay)上**同时不同**。另有一条对"数据还是方法"的追问结论:「Arm (c), full weights on Kev-27B's own data, is the worst (54 false positives). AutoJev, full weights on other data, makes 0.」——即**不是"全参"本身,而是数据广度**。

### 3.4 后续:AutoJev 已并入 Perplexity 模型线

这是本票中一个容易被漏掉但很重要的事实链:

| 阶段 | 仓库 / 记录 | Decision Index 上的表现 |
|---|---|---|
| 0.2(release-v2) | `denis-pplx/autojev-27b` | 记录名 **AutoJev-27B**,engine `autojev-27b`,**skill 50.94 / raw 63.37** |
| 0.2.1 | engine `autojev-27b`,显示名改为 **pplx-decider-v1-27b**,`weights_repo` = `perplexity-ai/pplx-decider-v1-27b` | skill **56.4** / raw 66.89 |
| 0.3.x | 面板中 **"autojev" 出现次数为 0**;另有独立行 **Perplexity Decider v1.1 (27B)**,kind `full fine-tune`,base `Qwen/Qwen3.8-27B` | skill **62.75** / raw 71.66 |

`perplexity-ai/pplx-decider-v1-27b` 的卡面写 `new_version: perplexity-ai/pplx-decider-v1.1-27b`,并给出 11 个基准的对比(Jev / Qwen3.8-27B / pplx-decider-v1-27b):Overall **84.51% / 74.76% / 85.71%**,其中 Jev 在 WinoGrande、JudgeBench、BBH、JevBench public hard、Belebele、TruthfulQA binary 上领先,pplx-decider 在 FinancialPhraseBank、RAGTruth、TabFact、Circa 上领先。该卡注明「The pplx-decider-v1-27b results were measured through the **Perplexity API**」——即它此刻是**托管服务**的一部分,而不再是那个"H200 自训 Demo"。

**推论(标注为推断,非事实)**:AutoJev 是 denis-pplx 的一条实验线,而名称前缀 `pplx` 与迁移到 `perplexity-ai` 命名空间表明它已被产品化收编。**这解释了为什么 0.3.x 面板里找不到 AutoJev**——它没有消失,而是改了名字。**但本文件未能找到一份明确说明这次改名的公告**,列为未决问题(§9)。

---

## 4. 公共评价

### 4.1 Decision Index(`multimodalart/jev-decision-index`)

**定位**:社区维护(作者 a.polinario,复现仓库 `github.com/apolinario/decision-index`)的第三方索引,明示「Unofficial and community-maintained; **not affiliated with TypeSafe AI**」。它同时是**排行榜**和**复现追踪器**。

**当前版本**:`Decision Index 0.3.2`,edition `release-v3`,panel id `decision-index-0.3`,基准日 2026-10-06,index.json 生成于 **2026-10-15T14:44:30Z**;参考模型 `jev-1.13.0`。([data/methodology.json](https://huggingface.co/spaces/multimodalart/jev-decision-index))

**评什么(怎么评)**:

- **面板规模**:公开套件 **37 个基准**,分 **5 个领域**——Knowledge & Reasoning(11)、Language Understanding(9)、Retrieval & Classification(6)、Tools & Automation(5)、Arts & Human Taste(6)。`categories` 中另有 `games` 领域但 **panel 与 suite 均为 0 项**。`index.json` 的 `suite` 字段另记 `benchmarks: 42`、`jev_benchmarks: 53`(含非面板/附加项,口径比 README 的 37 宽)。([index.json](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json),[methodology](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/methodology.json))
- **规模**:`requests: 110,201`,`scoreable: 109,759`,source cases 110,313,fields 536,776,seed 20260919;抽样方式为确定性 hash 无放回。
- **机会校正(chance correction)**:**每个基准都做了机会校正**,使 0 = 随机猜测、100 = 全对;题面自带的 random baseline 逐条记录在 `benchmarks[id].baseline`(如 BANKING77 的随机 baseline 0.0127、ToolRet 的 0.1341)。
- **三种公式**([`suite.formulas`](https://huggingface.co/spaces/multimodalart/jev-decision-index/blob/main/data/index.json)):
  - `balanced_raw`(Public index, raw accuracy):`100 * sum_c(0.2 * category[c])`——五领域等权。
  - `balanced_skill`(**Full score**):`0.20 * public + 0.50 * private_same_skills + 0.30 * private_new_domains`,private 部分通过 untuned stock model 映射到公开量纲。**主榜用这个**。
  - `breadth_skill`(Public index, breadth):`100*((0.1+0.9*c)^0.2 的连乘 − 0.1)/0.9`,shift 0.1。
- **private 部分永不可见**:README 明写「Most of the score comes from tests nobody can train on, so a public-only advantage counts for little」;「The private tests and the new-domain tasks are never published」。
- **校准口径**(`suite.calibration_note`,32 个有唯一正确答案的基准,ToolRet/BRIGHT/RouterBench 除外,各等权,1/6 请求抽样,confidence 取所选选项上的概率);**「Rows an entrant trained on count as wrong here too」**——provenance 污染在私榜按错计。
- **答案覆盖**:`answer_gaps` 逐条公开每个 entrant 未作答的数量与**原因**(如 Verdict 未作答 59.08%,其中 46.57% 是"选了自己的弃权选项";Solomon v1.1 22.85% 是"只支持 2–8 个答案选项";Tev1 系列 12.29% 是"只支持 A–X 共 2–24 个标签")。

**谁能上榜(入选规则)**——这是本票要求的重点:

- **硬性规则 7 条**(`rules.hard_rules`):**No truncation**(超上下文一律记为 unsupported,绝不截断)、**No label filtering**(完整选项集,不做候选剪枝)、**No prompt tuning**(用作者原生的 inference helper / prompt / readout;机械 schema 转换与概率换算须随 adapter 记录)、**No labels in the adapter**(gold 答案绝不进入 model state)、**No correctness-based retry**(真错不重试,仅传输错误可重试 2 次)、**Native abstentions stand**(弃权保留为终态,不得被硬塞进选项)、**Unanswered = wrong**(unsupported/refused/errored 全部计入完整分母)。
- **完整性门槛**:记录状态须为 `complete`,且处理量 ≥ **119,898** 行(120,340 请求减去对所有引擎都排除的 442 问)。
- **entrant 分类不靠名字猜**(`rules.entrants.principle`):kind 来自 run 记录的加载内容,base 来自 Hub 卡自己的 `base_model` 字段或 run 记录的 runtime model,**解不出就留 null**。六类:`inference technique`(只服务 vendor 命名空间的 stock checkpoint)、`full fine-tune`、`LoRA`、`LoRA + head`、`head / adapter`、`fine-tune`。
- **一个模型只出现一次,以"当下成立"的配置为准**(`superseded_policy`):作者在模型之外设的上限被抬高、或替换 checkpoint,旧 run 即**退役出榜**(文件留存但不再竞争)。已退役 20+ 条,例:`kev-4b` → `kev-4b-raised`(label window raised),`nanojev` → `nanojev-8k`,`jevfire` → `jevfire-uncapped`。
- **作者可要求撤榜**:`kev-0.5b-raised` / `kev-0.6b-raised` / `kev-8b-raised` 三条**应作者要求撤回**,理由是它们属于旧的 Kev 世代(Qwen2.5-0.5B / Qwen3-0.6B / Qwen3-8B),已发布的家族是 Qwen3.5 的 0.8B/4B/9B。([discussions/10](https://huggingface.co/spaces/multimodalart/jev-decision-index/discussions/10))
- **互动/游戏类至今未跑**:`interactive reproduction runs have not started; Jev trajectories are never reused as their scores`;Jev 自己在 miniwob 上成功率 1.0(9 episodes),但在 scienceworld 0.0(10)、codenames 0.0(66)、minigrid_babyai 0.0833(12)、rtfm 0.17(100);hanabi 与 leduc 为 null。**games 领域因而是空的。**
- **环境**:评测硬件 `1 x NVIDIA RTX PRO 6000`;延迟单测(750 行,n=1 串行),与吞吐无关。
- **Jev 一侧的口径提示(重要)**:Jev 的 latency 记录写明是「Hosted API, network round-trip from our lab」,**与榜上 on-card 单进程数字不可比**;并记录了一条 API 漂移:「Its answers on this sample differ from the Sep-19 board rows in **20 choices**; API output drifted since the board run; scores are unchanged.」——**托管模型的答案会漂移,分数却未变**。

**榜面快照(index.json,生成于 2026-10-15)**:167 个 entrant 全部有 skill 分。前 12 行:

| skill | raw | 名称 | org | kind | base |
|---:|---:|---|---|---|---|
| 66.26 | 74.01 | vLLM-SR d3 (27B) | vLLM Semantic Router | full fine-tune | Qwen3.8-27B |
| 65.61 | 72.13 | linne-oak 1.0 (31B) | MISA-AICORE | full fine-tune | gemma-4-31B-it |
| 64.62 | 73.24 | Decider 31B (NVFP4) | – | full fine-tune | gemma-4-31B-it |
| 63.81 | 71.08 | Sieve 31B | Sthānika AI | LoRA | gemma-4-31B-it |
| 62.83 | 69.08 | Bobcat 1.3 (31B) | – | LoRA | gemma-4-31B-it |
| 62.75 | 71.66 | Perplexity Decider v1.1 (27B) | Perplexity | full fine-tune | Qwen3.8-27B |
| 62.46 | 71.93 | Darwin-27B-ZTC | FINAL_Bench | full fine-tune | FINAL-Bench/Darwin-27B-RSI |
| **61.53** | **69.98** | **Jev 1.14** | **typesafe.ai** | **hosted API** | – |
| 60.96 | 69.76 | MORJ 1.0.1 (Gemma 4 31B) | – | inference technique | gemma-4-31B-it |
| 60.22 | 69.37 | AnyJev L0 · Gemma-4-31B | – | inference technique | gemma-4-31B-it |
| 60.21 | 69.61 | Fastino GLiDE no-thinking (28B) | Fastino | full fine-tune | – |
| 60.17 | 69.32 | Sieve 27B | Sthānika AI | LoRA + head | Qwen3.8-27B |

**与 Stim/Kev 直接相关的行**:`Kev 27B` skill **58.78** / raw 67.55(full fine-tune, Qwen3.8-27B);`Kev 9B v2` 43.27;`Kev 4B r10` 39.50;`Kev 0.8B r15` 13.66。Jev 1.13 的冻结分为 balanced_raw 68.55 / balanced_skill 57.96(frozen)/ breadth_skill 57.27。

**版本史**:本票题干说的「约 40 基准」对应 **0.2**(`release-v2`,panel id `decision-index-0.2`,64 个模型)——`index-v2.json` 里五领域 panel 为 10 / 10 / 7 / 6 / 7,**合计正好 40**;该版总计 121,057 个请求,`suite.benchmarks` 记 43、`jev_benchmarks` 记 42。那一版里 `AutoJev-27B` = 50.94、`Kev 9B` = 35.41、`Kev 4B` = 31.31、`Kev 0.8B` = 13.26、`Jev` = 51.67(全部与 `jaredpalmer/kev` PLAN.md 的引用逐位一致)。0.2.1 有 70 个模型;0.3.x 有 167 个。**注意 0.2 与 0.3.2 的面板不同(40 → 37 项),两版分数不可直接比较。**

**Changelog 里有一条值得注意的评分 bug 修复**(`methodology.changelog`,v0.3.2,2026-10-12):「three same-skill sets (the GPQA, MMLU-Pro and BBH skill sets) were left out of the same-skill score for **37 models** scored since the 0.3 launch」——即**主榜的一次静默重算影响到 37 个 entrant**。这提示公共榜单的分数会因评分器修正而变动。

### 4.2 JevBench(Benchmark Heaven)

**定位与归属**:「JevBench by Benchmark Heaven… **Maintained by Benchmark Heaven, independently of TypeSafe AI**」;「Jev is one of the systems evaluated, **not the benchmark owner**」。当前发布 **v1.6.1**(首次发布 2026-10-06,行测量截至 2026-10-07,amended through 2026-10-08)。方法与代码在 `github.com/fstandhartinger/jevbench`;结果数据集在 HF `benchmarkheaven/jevbench`。(Space README + `snapshot.json`)

**评什么**:

- **题量**:「The release draw contains **1,200 sealed and 300 public decisions**. Self-hosted systems answer **1,500 decisions**」;v1.6.1 的 full-set API 行答同一集合。**只发布系统级聚合,从不发布 item-level 数据;sealed 题永不放出**。
- **两根轴,不是一个分**(这是最容易读错的地方):
  - **Capability**:Intelligence 与 Calibration 的**算术平均**,取在**两个官方上限之内**的模型排名。开源权重board 的头条指标。上限为 **USD 0.0646 / 1,000 decisions**,调整后中位延迟 **1.23 s**。
  - **Composite**:Intelligence / Calibration / Speed / Cost 四轴的**加权调和平均**,再乘平方惩罚——`H = Σw / Σ(w/axis)`,`Composite = H × min(1, I/50)² × min(1, Speed/50)² × min(1, Cost/50)²`;Option A 权重各 25%,**Intelligence floor 50**。托管 API board 用 Composite。
  - 「Equal 25% weights do not mean an arithmetic average」——官方特意强调调和平均不是算术平均。
- **Intelligence 的定义**:**chance-corrected decision accuracy**,带 tier 加权与 public/sealed gap 调整。Calibration 评的是概率分布本身,「not simply accuracy among confident answers」。
- **缺失答案**:由打分器计入,不作静默剔除;`missing measurements are null, never zero`。
- **披露义务**:「**Training on the public split must be disclosed.**」
- **明示的局限**(官方自陈,对评估票 #12 有直接价值):题目覆盖有限;**sealed 输入使完全独立复跑不可能**;托管端点版本可能不透明;成本由公布基准建模,延迟依赖运行时与测量位置;「A leaderboard score does not establish reliability on your workload」。**不同方法版本之间的分数不可比**(历史版本保留其原定义)。

**榜面快照(v1.6.1,94 个 system,92 个 ranked)** — 前 20:

| rank | 名称 | Capability | Intelligence | Calibration | Composite | $/1k | p50 (s) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Quyet-1.0-Large (Chinh Nguyen, Gemma-4-31B 派生) | 81.7 | 73.4 | 90.0 | 71.4 | 0.0445 | 0.382 |
| 2 | Sage 1.3.0 (Levanto Labs) | 78.6 | 65.6 | 91.6 | **74.0** | 0.0247 | 0.152 |
| 3 | deck-31B (krishna765, frozen Gemma-4-31B-it) | 77.6 | 73.0 | 82.2 | 68.7 | 0.0475 | 0.405 |
| 4 | **Jev 1.13.0 (TypeSafe AI)** | 77.1 | 63.6 | 90.6 | 71.5 | 0.0323 | 0.239 |
| 5 | torchcast-decision-12b (Torchcast AI) | 71.7 | 60.5 | 82.9 | 69.9 | 0.0285 | 0.224 |
| 6 | Jev-Omni (akhilaaa3, Gemma-4-12B merged) | 71.3 | 55.5 | 87.0 | 67.7 | 0.0292 | 0.459 |
| 7 | Winnow-12B Q8 | 71.2 | 59.5 | 83.0 | 68.9 | 0.0281 | 0.384 |
| 8 | Cygnet (blockbrain, frozen Gemma-4-12B-it) | 70.9 | 54.8 | 87.0 | 68.6 | 0.0283 | 0.224 |
| 9 | Plumb-4B (crh225, JevK5 v0.2 + LoRA) | 65.4 | 43.0 | 87.9 | 48.3 | 0.0170 | 0.190 |
| 10 | decider-4b v2 (Mapika) | 64.9 | 40.1 | 89.6 | 41.2 | 0.0152 | 0.200 |
| 11 | JevK5 v0.3 (4B) | 64.2 | 38.5 | 89.9 | 37.4 | 0.0170 | 0.188 |
| 12 | Clef-Flash (Cloudflare, Qwen3.5-9B post-train) | 63.8 | 42.2 | 85.5 | 36.6 | 0.0593 | 0.356 |
| 13 | deck-4B v1.0 (krishna765, JevK5 + LoRA, FP8) | 63.6 | 38.6 | 88.6 | 37.3 | 0.0165 | 0.302 |
| 14 | JevK5 v0.2.0 | 62.8 | 36.3 | 89.4 | 32.1 | 0.0170 | 0.231 |
| 15 | Hopper | 62.4 | 34.7 | 90.1 | 28.9 | 0.0181 | 0.263 |
| 16 | Imajev-4B (RTX 5090) | 61.9 | 34.6 | 89.2 | 28.7 | 0.0168 | 0.292 |
| 17 | jqv (Qwen3-32B zero-shot) | 61.9 | 33.9 | 89.9 | 25.4 | 0.0424 | 0.345 |
| 18 | metask-jev-4b | 61.8 | 39.2 | 84.4 | 37.3 | 0.0256 | 0.258 |
| 19 | lev (Interfaze AI, Qwen3.5-4B + LoRA r32) | 61.7 | 41.1 | 82.3 | 42.2 | 0.0188 | 0.355 |
| 20 | Malkuth-4B (newfull5, Kev post-train) | 61.1 | 33.8 | 88.4 | 26.2 | 0.0297 | 0.299 |
| 40 | **kev 4B (research preview)** | 44.0 | 19.5 | 68.5 | 6.6 | 0.0138 | 0.294 |

**两点读法**:① Jev 的 **Composite 71.5 只排第 4**,虽然 Capability 第 4、Calibration 90.6 属第一梯队——**榜首 Quyet-1.0-Large 靠 Intelligence 73.4 取胜,不是靠校准**;② 榜上有大量 4B 级模型 Calibration 在 88–90,却 Intelligence 只有 30–40,**说明"校准好"在低能力模型上很容易被刷出来**。

**Kev 一方自己的读法**(`PLAN.md`,非官方榜):JevBench public:AutoJev 0.870 / Kev-27B 0.866;`PLAN.md` 资源表另记「JevBench | 09-24 | public items, unchanged harness, released family | report; **Kev-9B hard 0.568 vs Jev 0.741**」。

**谁能上榜**:有独立的 [Submit a model](https://benchmarkheaven.com/submit) 页(「Hand in a model and we will run it through…」);托管 API 与开源自托管走**两块不同的 board**(open-weights board 以 Capability 为头条,hosted API board 以 Composite 为头条)。公开 split 上的训练必须披露。

### 4.3 第三个「JevBench」:metamorphic coherence benchmark

**必须消歧**——Benchmark Heaven 自己在 README 里写了这一条:

> **Name disambiguation:** this is JevBench by Benchmark Heaven at benchmarkheaven.com. The [metamorphic coherence benchmark](https://jevbench.github.io/) is a separate project; its consistency scores measure a different property.

`jevbench.github.io` 那个 JevBench 测的是**一致性而非准确率**:

- 「It measures consistency, **not accuracy or speed**, and **needs no gold labels**。」
- 每条测试**只变换请求的问题侧**(绝不改被判断的输入),变换对答案的影响由概率论/选择理论**固定**;以概率单位度量答案偏离该定律的距离,**偏离 < 0.05 视为通过**。
- **50 条定律**,分 5 个维度:**REP**(同一事件的不同描述——改写、翻译、重排或改名选项——应得同一答案)、**BAT**(批量独立性:一个问题的答案不受同批其他问题、顺序或数量影响)、**MEA**(概率测度:正反两面和为 1;合并选项即合并概率)、**LOG**(逻辑序:合取不得比任一部分更可能)、**CHO**(选择集条件化:增删无关选项不改变其余选项的相对概率)。
- 榜单:`JevBench-mini`,**1,200 条测试 / 240 个 case / 12 个领域**;总分取五维均值并给 95% 区间,并按区间给出**排名范围**(区间重叠的模型不应视为有差别)。
- 发布形态:技术报告 PDF + `pip install jevbench` + HF dataset。

**同时**还有 `Praveenrajus/jev-bench`(HF dataset,★7,12,122 downloads)与 `Leanmcp/jevbench`(★1)、`aahf/JevBenchmark`(Space)、`benchmarkheaven/JevBench`(Space,★9)等多个同名/近名产物。**"JevBench" 这个词在生态中并不唯一。**

### 4.4 官方自评:Jev 1.13 jaggedness

TypeSafe 自己发布了一份失败模式清单(见 §2.3),涵盖字面理解、计数不可靠、日期比较、多跳间接、长 state 稀释、对抗内容、矛盾指令、选项顺序、以及"需要生成"的任务。这是**官方口径的一手"已知短板"来源**,在评估设计上与第三方榜单互补:第三方测"你赢没赢",这份文档测"你会不会用错"。

---

## 5. 其他同类:System One 兼容服务端 / 克隆 / 论文与博客

### 5.1 生态的分层

按"与 TypeSafe System One 的关系"可分为四层:

1. **权重与训练层**:直接训出决策模型的家族(Laya、CLM、Mapika/decider、AutoTrust、OneJev、Sieve、Tev1 等)。
2. **服务端与运行时层**:不改权重、让别的模型扮演 System One(OpenJev、AnyJev、von、HarnessRouter、laya-mlx、simple-jev、razorback16/openjev)。
3. **应用与集成层**:MCP connector、路由、记忆、语音浏览器等(`itsmostafa/system-one-connector`、`libingzheren/Jev-Mem`、`moritzkremb/jev-voice-browser`、`jexp/neo4jev` 等)。
4. **评价与研究层**:Decision Index、JevBench ×2、jaggedness 类长文、复现清单。

### 5.2 主要家族对照(GitHub 星数为抓取日快照)

| 项目 | 定位 | 做法 | 主要结果 / 备注 |
|---|---|---|---|
| **Kev** `jaredpalmer/kev` ★9,214 | 「Jev-like family of decision models… you can train and run yourself」 | 架构依据 Archer Hume 长文;0.8B/4B/9B 用 **冻结基座上的小 adapter**,27B **全权重微调**(从 Qwen3.8-27B post-trained 起,51 GB);每 checkpoint 附拟合温度;CUDA + Apple MLX;文档最长 65,536 tokens | breadth-v1 留出集指数:**Kev-27B 52.3 vs Jev 54.0**(Kev-0.8B 23.3 / 4B 38.0 / 9B 41.0);「New Sources」准确率 Kev-27B 0.851 dev / 0.889 test vs Jev 0.857 dev。TypeSafe Python SDK 可**不改代码**指向 Kev server |
| **Laya** `convaiinnovations/laya`(HF ♥5,628,48,894 dl) | 「Multilingual, **non-autoregressive** System 1 decision model」;**pip 可装**(`pip install laya`) | 用 **RLCD** + 严格 proper scoring rule 训练;声称一次前向约 33 ms,覆盖 **100+ 语言**,永不生成文本 | 官方卡自陈 8,192 token 上限是需显式开启的;实测 4,000 token 输入在 Apple GPU 上约 1.7 s;长文准确率在 ~4,000 token 后开始波动。**生态中唯一一个明确自称 RLCD 的公开模型** |
| **CLM** `Contrastive-LM/CLM` ★3,038 | 「**Contrastive Language Models**… a new class of System One model trained with a contrastive learning objective that connects states and actions」 | 对比学习目标(与 SFT 路线不同) | 在 Decision Index 有 `CLM-v0.1-8B` 条目 |
| **Mapika/decider** ★1,150 | 「A family of **System One-style** models fine-tuned from Qwen3.5」 | 一族多尺寸(2B / 4B / 31B / 35B-A3B + NVFP4) | JevBench:`decider-4b v2` Capability 64.9(rank 10),`decider-2b` 44.6;Decision Index 0.2.1 中 Decider 4B 40.7 / 35B-A3B 47.11 / 2B 28.97。**名字叫 "Decider" 的推理技巧变体也存在**(`Decider chat · Gemma-4-31B` 属 inference technique) |
| **AnyJev** `nokia-applied-research/AnyJev` ★1,295 | 「Turn **any LLM** into a Jev-style decision model: typed decisions, real probabilities, **no training**」 | 纯推理时改造,零训练 | Decision Index:`AnyJev L0 · Gemma-4-31B` skill **60.22** / raw 69.37(kind = inference technique)——**不训权重就排进前 10** |
| **von** `wfzyx/von` ★913 | 「The open-source System One decision model. Sub-15ms, **non-autoregressive**, local drop-in alternative to TypeSafe Jev」 | 非自回归,本地部署 | 主打延迟与本地替代 |
| **Open-Jev** `ZefanCai/Open-Jev`(HF dataset ♥108,7,009 dl)+ `Open-Jev-v1.1`(♥23) | **数据集**:「Open-Jev」合成 typed decisions,100K–1M 规模,CC0/CC-BY | 为复现提供公开训练数据 | 是生态里**引用最多的公开决策数据集之一**;另有 `pengchao1587/Open-Jev` 等派生 |
| **OneJev** `OmniJev/OneJev` ★171 + HF `OmniJev/*` | 「A **multimodal** System One decision model … about screens, photos, video and text in one forward pass」 | 多模态 | Decision Index:`OneJev1.5-27B` skill 59.90 / raw 72.46;`1.5-9B` 53.29;`1.5-4B` 49.17;`1.5-0.8B` 更低。**多模态克隆中的主力** |
| **Valen** `Liuziyu77/Valen` ★831 | 「Train a **Jev-like multimodal** model by yourself. System One Model, now with vision」 | 训练框架 + 视觉 | 面向"自己也训一个" |
| **AutoTrust** HF `autotrust/JEV-*`(JEV-27B-VL ♥5,369 / 1,600,363 dl;JEV-9B ♥169;JEV-27B ♥87;JEV-35B ♥943) | HF 上**下载量最大**的 Jev 家族命名空间 | 27B/9B/35B/26B 多尺寸;LoRA + head 与 full fine-tune 混用 | Decision Index:`JEV-35B` skill 56.95、`JEV-27B` 55.64、`JEV-9B` 45.21、`GEV-26B-Decide` 56.17。**注:其 `JEV-27B-VL` 卡面挂了两个 arXiv tag(`2609.14973`、`2512.16899`),经核查分别指向 PhysBrain 1.5 与 Multimodal RewardBench 2,与决策模型无关**——元数据不可尽信 |
| **vLLM Semantic Router** `vllm-sr/*` | 推理侧决策模型族,d3 全尺寸 | 主打 LLM 路由场景 | Decision Index:**`vLLM-SR d3 (27B)` skill 66.26 是全榜第一**;`d3-flash (9B)` 55.79;`d3-mini (4B)` 52.61。另有 `Decision 1.0-*`(Kai/Sol/Nox/Lux/Eos)旧线 |
| **Perplexity Decider** `perplexity-ai/pplx-decider-v1-27b` → `v1.1` | AutoJev 的产品化续作(见 §3.4) | 全参 SFT | Decision Index v1.1 skill **62.75** / raw 71.66;自测 11 基准 overall 85.71% vs Jev 84.51% |
| **Tev1** `togethercomputer/Tev1-{0.8B,4B}-experimental` + [Together AI 官方博客](https://www.together.ai/blog/how-to-train-your-own-jev) | 云厂商的"你自己训一个"示范 | **38,000 条样本**(MultiNLI/BoolQ/Banking77/AG News/SST-5/程序化策略/路由/研究分类),Qwen3.5-4B 起步,Together 微调服务 | 标题即定位:「**How to train your own Jev for $17**」。Decision Index 记录其只支持 A–X(2–24 个标签),12.29% 请求记 unsupported |
| **JevK5** `alibiserikbay/JevK5*` | 4B 级 LoRA 家族 | — | JevBench `JevK5 v0.3 (4B)` Capability 64.2(rank 11);Decision Index 48.35 |
| **Cloudflare** `Cloudflare/clef`、`clef-flash` | 大厂入场 | clef = Qwen3.8-27B 全参;clef-flash = Qwen3.5-9B post-train;另有 clef-omni(30B-A3B) | Decision Index:clef skill 53.08 / raw 71.07(注意 **raw 高但 skill 低**);JevBench `Clef-Flash` Capability 63.8 |
| **Sieve** `sthanika-ai/Sieve-*` | 多尺寸(0.8B/2B/4B/9B/27B/31B) | LoRA / LoRA+head | Decision Index:`Sieve 31B` 全榜第 4(63.81,LoRA);`Sieve 27B` 60.17 |
| **Jev-Omni** `akhilaaa3/Jev-Omni`(♥412) | Gemma-4-12B merged | LoRA + head | JevBench Capability **71.3**(rank 6);Decision Index 42.35 |
| **服务端/运行时** | `razorback16/openjev` ★713(「Open, Jev-compatible System One decision server on **DiffusionGemma**」)、`HarnessRouter/SystemOneHarness` ★194、`mizorewww/laya-mlx` ★6,949(MLX 原生运行时,宣称 M3 Max 上 7–14 ms)、`SimpleJev/JevAny-*`、`pngwn/system-one-qwen3.5-4b-scorer`、`Meanblock/JEV-CPU` | 不让权重改,让通用模型扮演 System One | Decision Index 里表现为 `inference technique` 一类;**DiffusionGemma 路线**是其中一支有代表性的非主流做法 |
| **周边与上层** | `itsmostafa/system-one-connector` ★352(MCP)、`libingzheren/Jev-Mem` ★213、`moritzkremb/jev-voice-browser` ★406、`0xNatoshi/jev-codex-router` ★274、`jexp/neo4jev` ★174、`angel291592/Intent-Router` ★763、`strands-labs/strands-decider` ★652 | 应用与集成 | 说明需求已外溢到路由、记忆、语音、图数据库等场景 |

### 5.3 论文与博客(一线来源)

| 来源 | 类型 | 要点 |
|---|---|---|
| [archerhume.com/posts/jevs-architecture-unmasked](https://archerhume.com/posts/jevs-architecture-unmasked/) | 长文(反汇编) | 见 §2.4;**Kev 的架构依据**,也是整个复现潮的公共参照 |
| [together.ai/blog/how-to-train-your-own-jev](https://www.together.ai/blog/how-to-train-your-own-jev) | 厂商博客(实操) | "$17 训一个";给出完整数据配比与脚本路径;并说明 Jev「has quickly become one of the most talked about model releases in the AI space」 |
| [vllm-sr.ai/blog/decision-models](https://vllm-sr.ai/blog/decision-models/) | 厂商博客 | vLLM Semantic Router 的决策模型叙事(其 d3 现居 Decision Index 榜首) |
| [invergent.ai/blog/rune](https://invergent.ai/blog/rune/) | 厂商博客 | Surogate Rune(Decision Index:`Surogate Rune 26B-A4B v3` skill 57.43) |
| [hf.co/blog/Hanno-Labs/decisionbench-bosun-v3-1](https://huggingface.co/blog/Hanno-Labs/decisionbench-bosun-v3-1) | HF 博客 | Hanno-Labs 的 Bosun v3.1 与自建 decisionbench |
| `arxiv:2609.14973` / `arxiv:2512.16899` | arXiv | **经核查与决策模型无关**(PhysBrain 1.5 / Multimodal RewardBench 2),系 autotrust 模型卡误挂的 tag——**不构成本领域的学术文献** |
| Decision Index `news.html` | 聚合 | 按 Decoding / Diffusion / Trained / Prior art / Explainers 五类追踪条目 |

### 5.4 公开训练数据(生态的"燃料层")

复现潮能跑起来,很大程度上靠这几份公开数据:

| 数据集 | 规模标签 | 许可 | 热度 |
|---|---|---|---|
| `ZefanCai/Open-Jev` | 100K–1M | CC0-1.0 | ♥108 / 7,009 dl |
| `ZefanCai/Open-Jev-v1.1` | 100K–1M | source-dependent(CC0-1.0 + CC-BY-4.0) | ♥23 / 2,279 dl |
| `tasksource/tasksource-jev-typed-decisions` | **1M–10M** | other | ♥21 / **12,977 dl**(下载量最高) |
| `SargeDev/jev-distill-corpus-v3` | 100K–1M | apache-2.0 | ♥38 / 3,625 dl |
| `samatv256/jev-decisions-v1` | — | — | ♥14 / 2,879 dl |
| `autotrust/jev-decision-index-results` | — | — | ♥3 / 3,736 dl(即 Decision Index 的衍生产物) |
| `benchmarkheaven/jevbench` | — | — | JevBench 官方结果数据集 |

相反,**AutoJev 的 73,000 条 curated 语料不随权重发布**;Kev 的 PLAN 记录了它自己另建 65,667 条合成记录(仅由开源权重模型标注)的过程。

---

## 6. 空白清单(供定位票 #2 与评估票 #12 参考)

以下每条都绑定到上面的一手证据。

**A. 训练与算法层**
1. **RLCD 没有公开实现。** Decision Index 的 entrant 分类里不存在对应"强化学习"的一类;榜首与前列全是 SFT/LoRA/推理技巧。唯一明确自称 RLCD 的公开模型是 **Laya**(其卡面自述用 proper scoring rule 做 RL)。([DI entrants 分类](https://huggingface.co/spaces/multimodalart/jev-decision-index);[Laya 卡面](https://huggingface.co/convaiinnovations/laya))
2. **TypeSafe 的权重、RLCD 算法、以及任何匹配其校准主张的公开证据仍未被发布**——这是 Decision Index 自己列的"仍未开源"三项。([Space README](https://huggingface.co/spaces/multimodalart/jev-decision-index))
3. **训练语料普遍不随权重释出。** AutoJev 明写「The exact curated training corpus is not bundled」;Kev 的合规化合成管线(65,667 条)与 Together 的 38k 配比是**少数**把数据配方写清楚的做法。数据层虽有多份公开集(§5.4),但**与权重一一对应的可复现"权重+数据+配方"三元组仍然稀少**。
4. **全参 vs LoRA 的因果仍未分离。** Kev PLAN 自己指出 AutoJev 的领先在方法学上是**混淆的**(全参 + 不同数据同时变化),而其自身的 arm (c) 反而更差——**"数据广度"而非"全参"才是主因**这一判断目前只有单方证据。

**B. 评测层**
5. **交互 / 游戏类评测至今未启动。** Decision Index 明确写「interactive reproduction runs have not started」;`games` 领域 panel 与 suite 均为 0 项;Jev 在 scienceworld(0/10)、codenames(0/66)、minigrid(1/12)上表现很差。**这是一个既已声明、又完全空着的领域。**
6. **私有测试不可复跑,sealed 题不可能独立验证。** Decision Index 的 50% + 30% 权重来自永不公开的 private 部分;JevBench 的 1,200 条 sealed 题同样永不放出,官方自陈「sealed inputs prevent complete independent reruns」。**没有任何一个公共评测具备端到端可复现性。**
7. **多模态评测仍薄。** Decision Index 有一个 Vision board,但公开榜单里视觉模型条目少;AutoJev 自己就警告「Published benchmarks measure text decisions; **image support is not a natural-image accuracy claim**」;JevBench 另有独立的 ImageJevBench / AudioJevBench board。**"决策模型在图像/音频上的真实能力"缺少公开的、经第三方测量的证据。**
8. **长上下文几乎没有被系统测过。** Decision Index 的面板中大量 entrant 因 8,192 token 上下文被记为 unsupported(Kev 家族、Solomon、Tev1 等);Kev 是少数按长度给出验证值的(0.8B/4B/9B 验证到 8,192,27B 到 65,536);Meta 侧的对照是 TypeSafe 官方 262k "danger zone"(自陈更慢更不准)。**"多长还准"的公开横评基本不存在。**
9. **"校准好"在低能力模型上容易被刷。** JevBench 上大量 4B 模型 Calibration 88–90 而 Intelligence 只有 30–40;Jev 自己是 Intelligence 63.6 / Calibration 90.6。**缺少把两者绑定的公开报告习惯**(Kev 的 ECE+Brier 双报是少见的反例)。
10. **评分口径与元数据的可核查性弱。** ① 三个不同的东西都叫 JevBench;② Decision Index 自己在 v0.3.2 修了一个影响 **37 个模型** 的 same-skill 评分 bug;③ autotrust 模型卡挂着与决策模型无关的 arXiv tag;④ Decision Index 记录了**托管模型 Jev 的答案在两次板次间漂移了 20 个 choice、而分数未变**。**"分数会变、口径不唯一、元数据会错"是生态的结构性问题。**
11. **中文与其他语种缺少上榜的决策模型。** 已有中文教程(`datawhalechina/jev-cookbook` ★421)与若干语种基准(`DaehanKim/korean-jev-benchmark`、`hayriyigit/jev-bench-tr`、`mballouch/jev-darija`、`ahmetege/turkish_jev_noul`、`cnfeichen/jev-rag-zh-bench`),但**在 Decision Index 0.3.2 的前 60 名中,没有一个 entrant 以多语能力为主打或以非英语为定位**;声称覆盖 100+ 语言的 Laya 未出现在该榜上。(此条基于榜面名称与列示字段的观察,非逐模型核读其训练数据。)

**C. 工程与运营层**
12. **成本与延迟的对照稀缺。** TypeSafe 公开价 $42/Btok(输出免费);JevBench 已把 $/1k decisions 与 p50 纳入排名(所以并非全无),但**"自部署的实际成本"依赖运行时,官方自陈成本是建模而来**。
13. **托管模型的版本漂移无契约。** 官方 alias(`jev-latest`)会随时移动,官方建议钉版本 ID;第三方观察到答案实际漂移。**"同一个 API 名字下答案会变"对以概率阈值驱动的生产系统是个未解的运维问题。**

---

## 7. 与本票题干的三处口径修正

1. 题干说 Decision Index「约 40 基准」——准确说法是:**0.2 版是 40 个基准(五领域 panel 10+10+7+6+7,64 个模型)**,**当前 0.3.2 的公开面板是 37 个基准 / 5 个领域**(`index.json` 的 `suite` 字段另记 42 / 53,口径更宽);chance correction 的说法成立(0 = 随机,100 = 全对)。两个版本的面板不同,**分数不可跨版比较**。
2. 题干把 AutoJev 与 Decision Index 并列描述——实际是:AutoJev **曾是** 0.2 面板上的一条记录,0.2.1 起 engine 名 `autojev-27b` 保留但显示名与权重指向改为 `perplexity-ai/pplx-decider-v1-27b`,**0.3.x 面板已无 "autojev" 字符串**。
3. 题干说 AutoJev「约 73k 合成决策数据」——一手来源的准确表述是:`training_rows: 73000`(**训练池总行数**),全部 286 步跑完 73,000 个样本(1 epoch),发布的 checkpoint 200 只见过 **51,200** 个样本(占 epoch 的 70.14%);模型卡的措辞是「73,000 unique training examples」。

---

## 8. 来源清单与可达性

**已核读的一手来源**

| 来源 | 取得方式 | 用途 |
|---|---|---|
| `docs.typesafe.ai`(introduction / models / primitives / ml-primer / quickstart / jaggedness / llms.txt) | `curl`(`.md` 变体) | §2 全部官方事实 |
| `typesafe.ai`(首页) | `curl` | RLCD vs RLHF vs RLVR 的对外叙事 |
| `github.com/jaredpalmer/kev`(README、PLAN.md、`runs/autojev-h2h/report.json`) | 本地浅克隆 `/tmp/kev-src`(commit `1f04cfa`)+ `gh api ... ?ref=research-archive-2026-09-24` | §3.3、§5.2 |
| `github.com/denis-pplx/autojev`(仓库树、README、`src/`、`configs/`) | `gh api` | §3.1 |
| HF `denis-pplx/autojev-27b`(卡面、`configs/training.json`、`assets/results.json`、`assets/provenance.json`) | `hf-mirror.com` | §3.1、§3.2 |
| HF `perplexity-ai/pplx-decider-v1-27b` 卡面 | `hf-mirror.com` | §3.4 |
| HF Space `multimodalart/jev-decision-index`(README、`data/index.json`、`data/index-v0.2.1.json`、`data/index-v2.json`、`data/methodology.json`、`news.html`) | `hf-mirror.com` | §4.1、§7 |
| HF Space `benchmarkheaven/JevBench`(README、`snapshot.json`)+ `benchmarkheaven.com/jev-models/methodology` + `/submit` | `hf-mirror.com`、`curl` | §4.2 |
| `jevbench.github.io` | `curl` | §4.3 |
| `archerhume.com/posts/jevs-architecture-unmasked/` | `curl` | §2.4 |
| `together.ai/blog/how-to-train-your-own-jev` | `curl` | §5.3 |
| HF API(`api/models`、`api/datasets`、`api/spaces`、逐模型元数据) | `hf-mirror.com` | §2.4e、§5.2、§5.4 |
| `gh api search/repositories` | `gh` | §2.4d、§5.2 |

**不可达或未能核实的来源(显式列出)**

| 来源 | 状态 | 影响 |
|---|---|---|
| `huggingface.co` 直连 | 不可达 | 已全部改用 `hf-mirror.com`;内容同源,但**镜面可能有缓存延迟**,抓取的 `downloads`/`likes` 计数可能与主站有细微时差 |
| `raw.githubusercontent.com` | 不可达 | `togethercomputer/tev1` 的源码、`apolinario/decision-index` 的实验室脚本(`run.py`/`compute_indices.py` 等)未逐行核读;相关描述只取其 README/文档口径 |
| `registry.npmjs.org` | 不可达 | 未核 JS SDK 的实际包内容;`npx skills add` 类安装路径只取自官方文档描述 |
| `vllm-sr.ai/blog/decision-models`、`invergent.ai/blog/rune`、`hf.co/blog/Hanno-Labs/decisionbench-bosun-v3-1` | 仅在 Decision Index 的 `news.html` 中见到链接,**本文件未取回正文** | §5.3 中这三条只记为"存在并已链接",其内容主张未验证 |
| Archer Hume 长文的「What would change my mind?」与「Sources and related work」两节 | 抓到首屏目录,正文未逐条核读 | 其七条架构主张的**证据分级**已抓取,但"什么会推翻它"未纳入 |
| Decision Index 的 Vision board(`data/vision.json`)与 `data/v03.json`、`data/own.json` | 文件存在(50 KB / 48 KB / 39 KB),**本次未解析** | §6 第 7 条关于多模态薄弱程度的判断基于卡面自述与 README,未基于 Vision board 的逐行数据 |
| `runs/autojev-h2h/report.md` | **在归档 ref 中 404**(只有 `report.json`) | §3.3 的数字取自 `report.json` 与 `PLAN.md` 的渲染表格 |
| TypeSafe 官方对复现潮 / Archer Hume 长文的回应 | **未找到** | §2.4 只陈述社区一侧;官方沉默与否无一手证据 |
| AutoJev → pplx-decider 改名的官方公告 | **未找到** | §3.4 的迁移链由 Decision Index 记录与两张模型卡互证,**改名动机与时间是推断** |
| JevBench 的 `key`/`name` 完整映射与 `levanto.ai/pricing` 等价格依据页 | 未逐一复核 | §4.2 表中 `$/1k` 取自 `snapshot.json` 的 `price.usd_per_1000_decisions`(kind 多为 `estimate`) |

---

## 9. 未决问题(交给下游票)

1. **改名的官方依据缺失**:`denis-pplx/autojev-27b` → `perplexity-ai/pplx-decider-v1-27b` 是否有公开公告?两个仓库是否同一批权重?(Decision Index 用 engine 名 `autojev-27b` 对齐二者,但模型卡未互相链接。)
2. **JevBench 的 `key` 与真实仓库的映射**:`snapshot.json` 只给 `name` 与 `source_url`,同族模型的多个版本(v1/v1.1、不同量化)在榜上如何折叠未核。
3. **Kev 的 27B 与 `Kev 27B` 在 Decision Index 上的 58.78 是否是 Kev 1.0 的 v2 权重**:Decision Index 记的 base 是 Qwen3.8-27B、kind full fine-tune,与 Kev README 的 Kev-27B v2 描述一致,但**未逐位对齐 Hub revision**。
4. **AutoJev 在 0.3.x 缺席的确切原因**:是作者未提交、是改名导致记录被替换、还是达不到 `complete_run_gate`(≥119,898 行)?`methodology.json` 的 `superseded` 列表里没有 autojev 条目。
5. **视觉 / 音频 board 的实际内容**:ImageJevBench、AudioJevBench、Decision Index Vision board 三处的榜单本次未解析,多模态空白的具体形状因此尚未量化。
6. **RLCD 的可复现性**:Laya 是唯一自称 RLCD 的公开模型,但其训练代码/奖励实现是否公开、是否可复跑,本文件未核。
