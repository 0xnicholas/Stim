# RLCD（TypeSafe / Jev）— research record

**观察日期：** 2026-10-17（UTC）。**文档版本：** `docs.typesafe.ai` 的 `sitemap.xml` 中最新 `lastmod` 为 2026-10-14；模型版本为 `jev-1.13.0`（别名 `jev-latest`）与 `jev-1.14.0`（别名 `jev-preview`，2026-10-14 起）。**怎么读的：** 抓取并阅读了 `docs.typesafe.ai` 的 128 个 sitemap URL（优先 `.md` 版本）、TypeSafe 官网 blog（sitemap 列出 8 篇，逐字读了其中 6 篇）、法律条款页、GitHub 组织与其 12 个公开 repo、官方 Python SDK changelog；用 arXiv API、Crossref、HN Algolia 做发现与元数据核验；读了 Latent Space 对 TypeSafe CEO Diogo Almeida 的访谈逐字稿、Laya 的 Apache-2.0 训练代码、OpenJev-RLCD 与 sbert2s1 两篇第三方论文、以及三份独立的 Jev 评测。**没有拿到的：** `evals.typesafe.ai` 返回 HTTP 451（“Typesafe is not available in your region”）；`huggingface.co` 连接被重置；Google Patents 超时；X、LinkedIn、YouTube、archive.org 不可达；本环境的 `WebSearch` 返回 403，因此网络发现主要靠 arXiv API、HN Algolia、Crossref 和直接抓取。**证据分层：** **[PUBLIC]** = TypeSafe 自己发布的话，逐字引用；**[INFERRED]** = 从公开 API/文档/行为推出的结论，明确标为推断；**[RECONSTRUCTION]** = 候选算法，不声称是 TypeSafe 的实现；**[SECONDARY]** = 第三方分析或测量，只作为线索或独立证据，不替代第一方事实。

---

## 1. 结论摘要

- **[PUBLIC] 缩写是公开的，而且就是“Reinforcement learning for calibrated decisions”。** TypeSafe 的 AI primer 把它写成小写形式，launch blog 写成 “Reinforcement Learning for Calibrated Decisions (RLCD)”。两处都把 RLCD 放在 RLHF/RLVR 之后的“第三条路”位置。
- **[PUBLIC] 但 TypeSafe 对 RLCD 的定位是“任务/North Star”，不一定是“新算法”。** CEO Diogo Almeida 在 2026-09-21 的访谈里说：“for us, RLCD is this new task”，并把 DPO 类比为“also do RLHF, despite not using the algorithm in that paper”。这句话是整份记录里最重要的一条校正。
- **[PUBLIC] TypeSafe 没有公开 RLCD 的算法。** 同一次访谈里，主持人问 “You’ve never actually published a paper or anything like that on it. No, right?”，CEO 回答 “No, not yet.”；launch FAQ 则说 “We do some pretty sophisticated stuff, but if you want to find out more, we’d have to hire you.”
- **[PUBLIC] Jev 的公开合同是“对调用方选项的概率分布”，不是生成文本。** `choice` 返回 `choice`、`probabilities`、`confidence`；`noul` 只返回 P(yes)；`score` 返回每个 level 的概率和概率加权均值；`box` 返回四个边和可选 confidence。`choice` 最多 255 个选项，`score` 最多 10 个 level。
- **[PUBLIC] TypeSafe 说概率“optimized against outcomes”，但没有说用的是什么 loss、reward、优化器或数据配比。** 原文是 “their probabilities are optimized against outcomes to reflect uncertainty”；除此之外，公开材料没有 Brier、log loss、proper scoring rule、REINFORCE、PPO、GRPO 等字样。
- **[PUBLIC] 数据是全合成的、自己造的、不训练客户数据。** CEO 在访谈中确认 “all your data is synthetic” — “Yep.”；launch FAQ 说 “We make all the data ourselves… We wouldn’t train on your data even if you asked us to.” 这支持“数据引擎”假设，但不公开循环细节。
- **[SECONDARY] “RLCD 没有公开实现”这句话，作为对“RLCD 这个方法”的描述，已经被反驳。** Laya（[`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya)，Apache-2.0）在 `laya/train.py` 里公开了 `rlcd_loss` 与 `proper_reward`；OpenJev-RLCD（arXiv:2609.38850）的论文声称公开代码和 API-compatible server（代码仓我未能直接读取）。它们都不是 TypeSafe 的实现，但都是公开的 RLCD 实现。
- **[SECONDARY] 独立评测显示 Jev 的校准是真实但有结构的。** Deußer 等人在 37 个数据集、279,925 个 Choice 答案上测得 pooled ECE 0.028、median per-dataset ECE 0.028、unweighted mean 0.061；Emotion 上 ECE 0.279；Noul 略欠自信（ECE 0.052），多标签 Noul 明显过预测（ECE 0.168）。这说明“校准”不是一个单一数字。
- **[INFERRED] 公开面更像“对选项的判别式 readout”，而不是“生成式 LLM 再解析出概率”。** 依据是 TypeSafe 自己说的 “does not generate text”“outputs all probabilities in parallel”“two-stage system of scoring independently then making an explicit choice”，以及 score 的每个 level “judged on its own”。如果这个推断成立，proper-score loss 就是普通的监督训练；“RL”要么指一种窄义 bandit/策略梯度，要么只是任务标签。
- **[RECONSTRUCTION] 最强的候选算法是：** 每个问题输出一个 categorical/ordinal 分布；用 strictly proper score（log + spherical + RPS，或 Brier/log loss）做目标；可选地加一层 policy gradient（logit 上加高斯噪声、group-mean baseline、CE 辅助）；在 held-out 数据集上拟合 temperature；用合成数据 + 选项扰动 + 难例挖掘做数据闭环。**这不是 TypeSafe 的实现，只是与公开证据最相容的候选。**

---

## 2. TypeSafe 到底公开了什么

### 2.1 缩写与“第三条路”

AI primer 的原文（[docs.typesafe.ai/introduction/machine-learning-primer](https://docs.typesafe.ai/introduction/machine-learning-primer)）：

> Pretrained language models have been adapted in two major ways. TypeSafe adds a third. RLHF and RLVR are shown here for context; TypeSafe's training path is RLCD.

> **Reinforcement learning for calibrated decisions** trains TypeSafe to return decisions and calibrated probabilities instead of generated text.

launch blog（[typesafe.ai/blog/introducing-system-one-models-and-jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)，2026-09-15，作者 Diogo Almeida）的原文：

> We built a new stack entirely focused on automation: with a new model architecture, parallel sampler for maximum efficiency, and training method we call Reinforcement Learning for Calibrated Decisions (RLCD).

同一篇 blog 的对照表把三者放在“Optimized with”一行：

| | Existing LLMs | System One + Jev |
|---|---|---|
| Optimized with | Reinforcement Learning with Human Feedback (RLHF) / Reinforcement Learning with Verifiable Rewards (RLVR) | Reinforcement Learning for Calibrated Decisions (RLCD) |
| Optimizes for | Human preference: writeups and chat responses that human raters prefer. / Verifiable rewards: outputs that can be programmatically verified. | Calibrated decisions: answers with epistemically honest probabilities on System One tasks. |

**结论：** “RLCD 是 RLHF/RLVR 之外的第三条路”是 TypeSafe 自己的说法，不是用户的杜撰。但要注意两点：

1. primer 的措辞是 “TypeSafe adds a third”，紧接着说 “RLCD optimizes for a different output contract”。它把差异放在**输出合同**上，而不是“第三种 RL 算法”上。
2. launch 说 “training method”，CEO 在访谈里说 “new task”。这两个词不矛盾，但份量不同：前者容易被读成算法，后者明确是优化目标。

launch FAQ 对 “Why was a new training algorithm needed?” 的回答，把 RLVR 的不足也说成**任务形状**的问题：

> RLVR is great for tasks with simple programmatic verification, but most real-world judgement tasks don’t fit into that shape. This tends to cause spikey / non-robust intelligence.

这句话重要，因为它说明 TypeSafe 的“第三条路”不是因为“RLVR 的算法不好”，而是因为“大多数判断任务不是可验证的”。这与 CEO 的 “new task” 说法一致，也解释了为什么 RLCD 的公开描述一直围绕输出合同，而不是优化器。

### 2.2 RLCD 优化什么：输出合同，不是算法

primer 对“calibrated decisions”的定义是：

> RLCD optimizes for a different output contract:
>
> * The model does not generate text.
> * It returns decisions and probabilities.
> * Higher probability should correspond to a greater chance that the answer is correct.

> Calibration makes uncertainty usable by software. Across many predictions from a well-calibrated model:
>
> * Outcomes assigned a probability of `0.2` should occur about 20% of the time.
> * Outcomes assigned a probability of `0.8` should occur about 80% of the time.
> * Outcomes assigned a probability of `1.0` should occur 100% of the time.

> These rates describe groups of predictions, not a guarantee about any single answer.

System One 文档（[docs.typesafe.ai/concepts/system-one](https://docs.typesafe.ai/concepts/system-one)）说得更直接：

> System One models are trained for calibrated decisions: their probabilities are optimized against outcomes to reflect uncertainty. Calibration is measured across groups of predictions; it does not guarantee that an individual answer is correct.

**这就是全部。** 没有 loss 名称、没有 reward 形式、没有训练数据配比、没有优化器、没有校准指标。`optimized against outcomes` 是唯一一句接近“训练目标”的话，但它可以被很多算法满足：监督 CE、Brier、log loss、policy gradient、甚至 post-hoc temperature scaling。

### 2.3 数据：全合成、自己造、不训练客户数据

launch FAQ 的原文（同一篇 blog，折叠项 “Where does our training data come from?”）：

> TypeSafe is primarily a data research lab, which is how the biggest results in AI get made. We make all the data ourselves. We wouldn’t train on your data even if you asked us to (no offense). We do some pretty sophisticated stuff, but if you want to find out more, we’d have to hire you.

models 文档（[docs.typesafe.ai/models](https://docs.typesafe.ai/models)）也明确：

> Jev is not fine-tuned or LoRA-adapted with customer data. It is trained with [RLCD](/introduction/machine-learning-primer) to return calibrated decisions, and the same weights serve every account.

CEO 在 Latent Space 访谈（[latent.space/p/jev](https://www.latent.space/p/jev)，2026-09-21）里进一步确认了“全合成”：

> Swyx: You’ve said, for example, that all your data is synthetic.
>
> Diogo Almeida: Yep.

以及为什么不用客户数据（节选，播客转写可能有误差）：

> The reason why we don’t want to train on our users’ data, even if we could… the real-world data has so much bias. There’s a power law of people asking the same things where you’ll end up overfitting to it… even if we had all of the data of the present, we would just overfit to the present, and then it wouldn’t work.

这段访谈还给了数据闭环的唯一公开线索（节选）：

> They find the jaggednesses, and then they address them surgically… in every single possible dimension, past, present, future.

**怎么读：** 这支持“数据是 TypeSafe 的差异化部分”“数据是全合成的”“会针对模型的 jaggedness 做定向修正”三点；但它没有说循环的输入、标签来源、筛选标准、reward 或重训频率。用户总结里的“construct → sample → obtain outcomes → compute calibration reward → identify weak regions → generate targeted synthetic tasks → retrain → validate”是一个合理的**重建**，不是 TypeSafe 公开的流程。

### 2.4 明确不公开的部分

最直接的两条第一方证据：

1. **没有论文。** Latent Space 访谈中，主持人问：“Which obviously you have some secret sauces to our knowledge. You’ve never actually published a paper or anything like that on it. No, right?” CEO 回答：“No, not yet.”（2026-09-21）
2. **没有细节。** launch FAQ：“We do some pretty sophisticated stuff, but if you want to find out more, we’d have to hire you.”

同一篇访谈的导语（Latent Space 编辑文字，不是 CEO 逐字）称 RLCD 是 “a novel, unpublished technique”。这可以作为“未发表”的旁证，但严格说它是二手描述。

TypeSafe 还明确拒绝公开 benchmark 表。launch FAQ：

> We deliberately chose **not** to publish performance against **public** benchmarks. In fact, we plan to only have one-off evals when we make product updates.

这让“用公开评测复现 Jev 的校准”变得不可能：没有固定的测试集、没有 per-item 概率、没有标签。第三方只能自建评测（见 §3.4）。

### 2.5 版本与产品面

models 文档（2026-10-17 读取）：

| | Jev 1.14 | Jev 1.13 |
|---|---|---|
| Model ID | `jev-1.14.0` | `jev-1.13.0` |
| 价格 | $42 / Btok；$0.042 / Mtok | 同左 |
| Context | 64k per-request / 32k per-question | 同左 |
| Input | Text | Text |
| 备注 | 2026-10-14 起；`box` 需要 1.14 | `jev-latest` 指向它 |

- `jev-latest` → `jev-1.13.0`；`jev-preview` → `jev-1.14.0`。1.14 在发布一周后才会成为 `jev-latest`（[Jev 1.14 blog](https://typesafe.ai/blog/jev-1-14)，2026-10-14）。
- Long context 通过 `TypeSafe-Danger-Zone-Long-Context: true` 打开，per-request 与 per-question 都到 262k；默认仍是 64k/32k。
- `box` 需要 `jev-preview` 或 `jev-1.14.0`，并且要 `TypeSafe-Danger-Zone-Images: true`；图片是 danger-zone 功能。
- 1.14 blog 说 “the Jev you all know and love (`jev-1.13.0`) was multimodal all along!”——这与“新架构/新模型”叙事有张力；它至少说明 1.13 的模型已经支持图像输入，具体是 backbone 还是 adapter 层，TypeSafe 没有说明。

### 2.6 公开陈述清单

| 主题 | TypeSafe 的原话 | 出处 | 它没有说什么 |
|---|---|---|---|
| 缩写 | “Reinforcement Learning for Calibrated Decisions (RLCD)” | [launch blog](https://typesafe.ai/blog/introducing-system-one-models-and-jev) | 没有说这是“新算法”还是“新任务” |
| 第三条路 | “TypeSafe adds a third.” | [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) | 没有说三者是互斥的数学类别 |
| 优化目标 | “Calibrated decisions: answers with epistemically honest probabilities on System One tasks.” | launch blog | 没有说 reward/loss |
| 概率含义 | “Outcomes assigned a probability of 0.2 should occur about 20% of the time.” | AI primer | 没有说用什么指标测 |
| 训练信号 | “their probabilities are optimized against outcomes” | [System One](https://docs.typesafe.ai/concepts/system-one) | 没有说 outcome 是标签、环境回报还是人类反馈 |
| 数据 | “We make all the data ourselves.” | launch FAQ | 没有说合成方法、标签来源、验证方式 |
| 合成 | “all your data is synthetic” — “Yep.” | [Latent Space](https://www.latent.space/p/jev) | 没有说“全合成”是否包括 teacher 生成 |
| 不训练客户数据 | “Jev is not trained on customer requests or responses.” | [models](https://docs.typesafe.ai/models) | 没有说是否用客户数据做 eval |
| 论文 | “No, not yet.” | Latent Space | 没有说未来会不会发 |
| 细节 | “if you want to find out more, we’d have to hire you.” | launch FAQ | 没有说任何算法细节 |
| 公开 benchmark | “We deliberately chose not to publish performance against public benchmarks.” | launch FAQ | 没有说内部 eval 的指标 |
| 选项顺序 | “the order of a Choice’s options can affect the answer, and jev-1.13 leans toward the option that comes first.” | [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) | 没有说训练时是否做 order augmentation |

---

## 3. Jev / System One 的公开约束

### 3.1 每种原语的输出

| 原语 | 请求 | 回答 | 关键限制 | 出处 |
|---|---|---|---|---|
| `noul` | `instructions`；可选 `criteria: {true?, false?}` | `{"type":"noul","noul":p}`，p = P(yes) | **没有 `confidence` 字段**；“A Noul’s probability distribution has only two outcomes, yes and no, so the single `noul` value describes it completely.” | [Noul](https://docs.typesafe.ai/primitives/noul) |
| `choice` | `instructions`；`criteria` 是 option→description 的 map | `{"type":"choice","choice":..., "probabilities":{...}, "confidence":...}` | 最多 **255** 个选项；“The sum of all values is 1.”；“The model never sees the question id.” | [Choice](https://docs.typesafe.ai/primitives/choice) |
| `score` | `instructions`；`criteria` 是**有序数组** | `{"type":"score","score":s,"probabilities":{...},"legend":{...},"confidence":...}` | 至少 2 个、最多 **10** 个 level；`score` 是 `Σ i·p_i`；“The model doesn’t see a level’s number or its neighbours”；“Every level is evaluated separately.” | [Score](https://docs.typesafe.ai/primitives/score) |
| `box` | `instructions`；**没有 `criteria`**；需要图片 | `{"type":"box","box":{top,bottom,left,right},"confidence":...}` | 需要 Jev 1.14；“A Box answer is always a box. There is no ‘not found’ answer.” | [Box](https://docs.typesafe.ai/primitives/box) |

**选项集由调用方定义。** `choice` 的 `criteria` 是 option→description 的 map；`score` 的 `criteria` 是有序 level 数组；`noul` 的可选 `criteria` 只描述 true/false 的含义。`none of the above` 不是内建类型，而是调用方自己加的一个 option——Choice 文档明确建议：“Add an `other` or `none of the above` option when the list might not cover every input, so the model can say none of the others fit.” 这意味着“选项集怎么构造”是应用侧的设计问题，不是模型内建的分类体系；也意味着 A/B 测试不同 option 集合是 API 允许的。

`confidence` 不是模型直接吐出的一个头，而是从分布算出来的统计量。公开公式（[Confidence](https://docs.typesafe.ai/confidence)）：

- Choice：`(p_max − 1/K) / (1 − 1/K)`。
- Score：`max(0, 1 − Σ_i p_i·|i − m| / D)`，其中 m 是最大概率的 level，D 是均匀分布在中点处的平均绝对距离。
- Noul：没有 confidence；文档建议用 `|2p − 1|` 作为等价的“距离 0.5”量。

这套公式与 TypeSafe 官方 MIT adapter 的实现一致（Kev 的记录已经逐行比对过 `system_one_adapter/_utils/confidence_metrics.py`，见 [kev.md](kev.md) §5.4）。对本记录重要的是：**API 的“概率”是完整的分布，`confidence` 只是后处理的摘要。** 所以任何关于“模型输出一个概率”的说法，都应该精确到“输出一个分布，再派生 confidence”。

### 3.2 架构证据：TypeSafe 说的 vs 可推断的

**TypeSafe 自己说的（[PUBLIC]）：**

1. “new model architecture, parallel sampler for maximum efficiency”（launch blog）。
2. “The model does not generate text.”（AI primer）
3. “Jev outputs all probabilities in parallel instead of autoregressively generating by token.”（launch blog）
4. “Jev is neither small nor an LLM, hence being off the intelligence Pareto curve.”（launch FAQ）
5. 高基数 Choice 用两阶段：“For the higher cardinality choices, we do a two-stage system of scoring independently then making an explicit choice.”（launch blog）
6. Score 的每个 level 是单独判断的：“The model doesn’t see a level’s number or its neighbours.”（Score 文档）
7. `box` 是一个连续的四个数输出，不是分类（Box 文档）。
8. 选项顺序会影响答案，且 1.13 “leans toward the option that comes first”（Jaggedness 文档）。
9. 同一请求里的问题是独立的：“Every question in a request sees the same state, is evaluated independently”；“one answer does not become context for another question.”（[Primitives](https://docs.typesafe.ai/primitives)）
10. 1.13 已经是多模态的：“the Jev you all know and love (`jev-1.13.0`) was multimodal all along!”（Jev 1.14 blog）

**可以推断的（[INFERRED]）：**

- 最自然的实现是：一个共享的 state encoder，每个问题一个 branch，branch 的最终表示通过一个 **对选项的 readout**（slot head 或 pointer head）产生 logits，再 softmax。理由：输出被约束在调用方给的选项上；问题之间互不影响；高基数时先逐项打分再显式选择；score 的 level 是逐项判断的。
- “Jev is neither small nor an LLM” 与 “pretrained language models… TypeSafe adds a third” 并不矛盾：它大概率是一个**预训练 backbone + 非生成式 decision head**。CEO 说 “neither small nor an LLM”，强调的是产品形态/输出合同，不是“没有用预训练 LM”。
- 一个纯生成式 LLM + 解析概率的架构，很难同时满足“不生成文本”“一次前向返回所有概率”“选项顺序有位置偏差但准确率不变”“score level 逐项判断”这些约束。**所以“生成式 LLM 吐一个概率向量”这个假设，与公开面不太相容。**
- **没有公开证据支持 ensemble/router。** 响应里的 `model` 是单一 versioned ID，TypeSafe 描述的是“一个模型”在回答；第三方探测也没有发现 routing 迹象。但这只是“没有证据”，不是“排除”——如果内部有多个 expert，API 仍然可以只暴露一个 `model` 字段。

**第三方探测（[SECONDARY]，只能当线索）：**

archerhume 的 [Jev’s Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)（2026-09-17）做了约 10,000 次 API 调用，结论是：causal transformer、共享 state、问题 branch 隔离、direct probability readout；sparse MoE 是“最不确定”的部分；`output_tokens` 是序列化计费，不是生成步数；255 个选项对应 256-slot head 的可能性存在，但 255 是请求校验限制，不是模型限制；选项之间有交互，pointer 与 slot head 都能解释。这些都不是 TypeSafe 的披露，不能当作事实。

**独立评测对“架构”的约束（[SECONDARY]）：**

- Deußer 等（arXiv:2609.37647）在 37 个数据集、346,009 个请求上评测 `jev-1.13.0`；他们发现 “Rotating the options leaves Jev’s accuracy unchanged”，但**没有**说概率不变。TypeSafe 自己的 jaggedness 文档则说顺序会影响答案。两者可以同时成立：顺序改变概率/个别答案，但总体准确率不变。
- Vals（[Independent Evaluation](https://www.vals.ai/blogs/independent-evaluation-of-jev)，2026-10-06）在 400 条 claim verification 上让 Jev 与 12 个系统对比；Jev 准确率 0.975，成本 $0.02/1000 cases；在 1% error budget 下，Jev 自动化 95% 的案例但错误率 1.6%，超过预算。这说明**准确率、校准、选择性风险是三个不同的东西**。

### 3.3 版本、上下文、Box

| 事实 | 值 | 出处 |
|---|---|---|
| 稳定版 | `jev-1.13.0`（`jev-latest`） | [Models](https://docs.typesafe.ai/models) |
| 预览版 | `jev-1.14.0`（`jev-preview`，2026-10-14） | Models；[Jev 1.14 blog](https://typesafe.ai/blog/jev-1-14) |
| 默认 context | 64k per-request / 32k per-question | Models |
| Long context | 262k / 262k，需 `TypeSafe-Danger-Zone-Long-Context: true` | Models |
| Box | 需要 1.14 + `TypeSafe-Danger-Zone-Images: true` | [Box](https://docs.typesafe.ai/primitives/box) |
| 图片 | TypeSafe 说 1.13 已多模态；1.14 开放 danger-zone | Jev 1.14 blog |
| 别名行为 | `jev-latest` 会随新版本移动；建议把阈值钉在 versioned ID 上 | Models |

1.14 blog 还有一句对“版本与阈值”很重要的提醒：

> If you have tuned confidence thresholds against one version, send that version’s ID instead of the alias, and switch to the new version when you are ready.

这句话本身就是“阈值是应用侧的事”的证据（见 §6）。

### 3.4 已公布的指标与独立评测

**TypeSafe 自己：没有公开的 Brier/ECE/accuracy 表。** 它只发布了两类东西：

1. **Workflow evals**（launch blog）：不是 ground-truth 分类，而是“assume there is a correct compute graph… use the predictions of the largest, smartest, and most expensive external models as reference probabilities”。指标是 `exact_actions` / `primary_action` 与参考模型的一致性；代码在 [typesafe-ai/WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals)（Apache-2.0）。这不是校准评测。
2. **Jev 1.14 的 “Measuring Intelligence” 图**：blog 里有一张内部测量图，但正文没有给出数字、指标定义或测试集；并说 “This measurement is our best guess and will continue to evolve… don’t expect them to be consistent.” 这不足以复现。

**第三方独立评测：**

| 来源 | 规模/方法 | 关键结果 | 对 RLCD 的意义 |
|---|---|---|---|
| Deußer, Sparrenberg & Sifa, arXiv:2609.37647（2026-09-29） | `jev-1.13.0`，37 个公开数据集，346,009 请求，15-bin ECE，Brier，AURC | pooled Choice ECE **0.028**；median per-dataset **0.028**；unweighted mean **0.061**；Emotion **0.279**；Noul ECE **0.052**（略欠自信）；多标签 Noul ECE **0.168**（过预测） | 校准真实存在，但强烈依赖数据集/聚合方式；不能用一个 ECE 概括 |
| Vals AI（2026-10-06） | 400 claim verification + LegalBench 12 子任务；12 个系统 | 1% error budget 下 Jev 自动化 95%、错误率 1.6%；LegalBench 原始准确率最后一名 | 准确率 ≠ 选择性风险；校准好不代表阈值 0.5 好 |
| JevBench / Benchmark Heaven v1.6.1（2026-10-06） | results JSON 140 行；Calibration 轴 = hard tier ECE + 对 gold 分布的 fidelity；Brier 是 multiclass **sum**；ECE 是 top-label **10 equal-width bins** | Jev 1.13.0 的 Calibration 轴 90.63；总分 63.29 | 第三方把“校准”做成复合轴；方法可读，但 sealed 部分不可完全复跑 |
| Decision Index 0.3（[apolinario/decision-index](https://github.com/apolinario/decision-index)） | 37 个 benchmark，五个领域，chance-corrected，coverage-adjusted；public index 可复跑，private 部分由维护者跑 | Jev 在 0.2.1 榜上是 57.89（README 的 parity 测试提到）；0.3 的 live board 分数本记录未取到 | 这是“能力”榜，不是校准榜；与 JevBench 不是同一把尺 |

**TypeSafe 与这两个榜的关系：** 文档、blog、GitHub 都没有提到 Decision Index 或 JevBench（我 grep 了全部 128 个 docs 页面、blog 索引和已读的 6 篇正文）。JevBench 的 README 明确说 “Maintained by Benchmark Heaven, independently of TypeSafe AI.”；Latent Space 的导语则说 JevBench 是 “something Diogo has rejected publicly”。所以这两个榜都是第三方，不是 TypeSafe 的官方评测。

### 3.5 这些约束对 RLCD 重建意味着什么

1. **输出合同是分布，不是文本。** 所以 proper scoring rule 是自然的训练目标；这一点用户的直觉是对的。
2. **如果模型是 softmax/pointer head，proper-score loss 就是普通监督学习。** 直接对 log loss/Brier 反传，不需要 RL。此时 “RLCD” 的 “RL” 要么是营销/任务标签，要么指别的层次（数据闭环、探索、策略梯度）。
3. **如果模型用 policy gradient，公开面也完全容得下。** 在 logit 上加噪声、用 proper score 做 reward、用 group-mean baseline，仍然是“一次前向 + readout”的架构；Laya 就是这么做的。
4. **公开面无法区分 2 和 3。** 这是本记录最重要的“不知道”：我们能看到输出合同，看不到优化器。
5. **`box` 是统一 proper-score 故事的唯一硬缺口。** 分类 proper score 不直接适用于四维连续输出；需要 CRPS、Gaussian NLL 或分位数损失。TypeSafe 没有说 `box` 的 confidence 是怎么来的，也没有说它是否校准。

---

## 4. 没有公开的部分

这一节的存在本身需要证据。我搜过的范围：`docs.typesafe.ai` 的 128 个 sitemap URL（全部抓成 `.md` 并 grep 过 `RLCD`、`calibrat`、`reward`、`loss`、`Brier`、`train`）、TypeSafe 官网 blog（8 篇，逐字读了 6 篇）、`typesafe.ai/legal/terms`、GitHub 组织 `typesafe-ai` 的 12 个 repo（含 README、changelog、code search）、arXiv 的 `all:"TypeSafe AI"`、`all:"RLCD"`、`all:"calibrated decisions"` 查询、HN Algolia、Crossref。**在这个范围内，TypeSafe 没有公开下列任何一项。**

| 缺失项 | 为什么重要 | 如果公开，会长什么样 |
|---|---|---|
| Reward / loss 的名称与形式 | 决定 RLCD 是监督、bandit 还是策略梯度 | 一行公式，或一句 “we use log loss + Brier” |
| 优化器与训练阶段 | 决定 “RL” 是否名副其实 | PPO/GRPO/REINFORCE/CE 的说明 |
| 训练数据配比、生成方法、标签来源 | 决定“数据是护城河”是否可复现 | 数据卡、生成 prompt、去重/污染筛查 |
| 模型规模、backbone、head 结构 | 决定能否复现架构 | 模型卡、参数量、context 训练长度 |
| 校准指标与 eval set | 决定 “calibrated” 是否可验证 | Brier/ECE/reliability + 每项概率 |
| 是否做 post-hoc temperature scaling | 决定“校准”是训练出来的还是拟合出来的 | 温度值、拟合集 |
| `box` confidence 的定义与校准 | 统一 proper-score 故事的唯一硬缺口 | 误差分布/CRPS/覆盖率 |
| 是否做选项顺序增强 | 决定位置偏差是数据问题还是架构问题 | 训练时的 shuffle 概率 |
| 合成标签如何验证 | 决定数据闭环是否循环 | 程序化 solver、人类审计、teacher 一致性 |
| 1.13 与 1.14 的训练差异 | 决定跨版本比较是否有效 | 版本变更说明 |
| “第三条路”的正式分类 | 决定 RLCD 与 RLHF/RLVR 是否同一层面的概念 | 一篇 position paper |

**一个重要的“没有”：TypeSafe 没有公开任何 per-item 的 Jev 输出与标签。** 第三方评测（Deußer、Vals、JevBench）都是自建输入、自己拿 API 回答；他们能测校准，但不能复现 TypeSafe 的训练分布。这也是为什么 §3.4 的数字只能当“行为证据”，不能当“训练目标证据”。

---

## 5. RLCD 的重建

**这一节整体是 [RECONSTRUCTION]。** 它不是 TypeSafe 的算法，也不声称与 TypeSafe 的内部实现一致。它的标准是：与公开 API/文档/行为相容；与公开的 RLCD 实现（Laya、OpenJev-RLCD）和 proper-scoring 理论一致；能被实验区分。

### 5.1 决策形式化

对 state `x`、问题 `q`、选项集 `O = {o_1,…,o_K}`，模型输出

```
p_θ(· | x, q, O) ∈ Δ^{K−1}
```

四种原语的统一方式：

| 原语 | K | 目标 `y` | 输出 | 备注 |
|---|---|---|---|---|
| `noul` | 2 | `y ∈ {0,1}` | `p = P(yes)` | 二分类；没有独立 confidence |
| `choice` | 2–255 | `y ∈ {1..K}` 或分布 `t` | `p ∈ Δ^{K−1}` | 选项名和描述都进模型；question id 不进 |
| `score` | 2–10 | `y ∈ {0..L−1}` 或分布 `t` | `p ∈ Δ^{L−1}`；`s = Σ i·p_i` | 有序；level 描述逐项判断 |
| `box` | — | `y ∈ [0,1]^4` | `μ, σ` 或四维分布 | 连续；分类 proper score 不直接适用 |

**统一似然。** 一个请求有 Q 个问题，问题之间独立（TypeSafe 文档明确说 “evaluated independently”），所以

```
L(θ) = Π_{q=1}^{Q} p_θ(y_q | x, q, O_q)^{w_q}        (choice / noul / score)
```

`box` 用条件密度替换离散概率：

```
L_box(θ) = Π_{j∈{top,bottom,left,right}} N(y_j; μ_j(x,q), σ_j(x,q)^2)
```

或者用 CRPS（连续 ranked probability score）作为严格 proper 的替代。`box` 的 `confidence` 可以是 `1 − normalized σ` 的单调变换，但**这只是重建**；TypeSafe 没有公开公式。

**一个关键的接口事实：** `score` 的 level 是“逐项判断”的，模型看不到 level 的编号和邻居。这意味着内部很可能不是对 L 个 level 做一次 softmax，而是对每个 level 做一次“这个描述是否匹配”的判断，再归一化成分布。重建时应该把这两种实现都当成候选：

- **softmax-over-levels**：`z_i = f(x,q,level_i)`，`p = softmax(z)`。
- **per-level binary + normalize**：`r_i = σ(f(x,q,level_i))`，`p_i = r_i / Σ_j r_j`。

两者在输出上不可区分，但训练梯度、校准行为和“加一个无关 level 会不会改变其他 level”不同。§8 给出可证伪的探针。

### 5.2 目标函数

#### 5.2.1 Proper score

设 `S(p, y)` 是一个 negatively oriented scoring rule（越小越好）。严格 proper 的定义是：对所有 `q ∈ Δ^{K−1}`，

```
E_{Y~q}[S(q, Y)] ≤ E_{Y~q}[S(p, Y)]，且等号只在 p = q 时成立。
```

常用成员：

| Score | 定义 | 取值范围 | 备注 |
|---|---|---|---|
| Log score / NLL | `S_log(p,y) = −ln p_y` | `[0, ∞)` | 自然对数；与 cross-entropy 同形 |
| Brier（sum） | `S_B(p,y) = Σ_k (p_k − 1[k=y])^2` | `[0, 2]` | Kev、Laya eval、JevBench 用这种 |
| Brier（mean） | `(1/K) Σ_k (p_k − 1[k=y])^2` | `[0, 2/K]` | 与 binary 的 `(p−y)^2` 对齐时常用 |
| RPS | `1/(K−1) Σ_{k=1}^{K−1} (Σ_{j≤k}(p_j − 1[y≤j]))^2` | `[0,1]` | 有序；对“远离正确 level”罚得更重 |
| Spherical | `p_y / norm(p)`（越大越好） | `(0,1]` | Laya 的 reward 里用 |

**Brier 的 sum/mean 必须写明。** 同一个分布，sum 和 mean 差 K 倍；跨论文比较时这是最常见的假差异。本文以下不特别说明时，用 **sum** 并注明。

#### 5.2.2 期望 proper score + 决策效用

```
min_θ  E_{(x,q,O,y)~D} [ Σ_q w_q · S(p_θ(·|x,q,O), y) ]  −  λ · E[ u(a(p_θ), y) ]
```

第一项是校准/锐度目标；第二项是下游决策效用。`a(p)` 可以是 `argmax`、阈值策略、或带弃权的策略。`u` 是成本矩阵决定的效用。

**为什么不能只优化校准。** 一个常数预测器 `p = base rate` 在大样本上是校准的（ECE ≈ 0），但完全没有分辨力。Murphy 分解（binary）说得很清楚：

```
Brier = Reliability − Resolution + Uncertainty
```

常数预测器的 Reliability = 0、Resolution = 0，Brier 等于 Uncertainty；它“校准”但无用。**所以校准本身是退化的；proper score 通过 Resolution 项惩罚这种退化。** 这也是为什么“单纯优化 ECE”是一个坏主意，而“优化 Brier/log loss”不是同一个东西。

**为什么不能只优化效用。** 决策效用只关心 `argmax` 或某个动作的期望效用；两个模型可以做出同样的动作，但一个报 0.55、一个报 0.99。效用一样，校准差很多。如果下游要做阈值/弃权/多动作组合，概率的**大小**就重新变得重要。所以三个轴（accuracy/discrimination、calibration、utility）都要报，不能用一个替代另一个。

**一个细微但重要的点：** proper score 的“诚实是最优”是**在真实条件分布上、在期望里**成立的。如果模型类无法表示真实条件分布（misspecification），最优报告是“在模型类内最接近真实分布的那个”，不一定是真实分布；如果数据有分布漂移，有限样本的最优也不等于校准。这就是 §6 里 claim 9 的数学基础。

#### 5.2.3 为什么 log loss 和 CE 不是两个阶段

`S_log(p,y) = −ln p_y` 就是交叉熵（one-hot 目标）。所以“Phase 1: SFT + cross-entropy”与“Phase 2: log-loss optimization”在**目标函数**上是同一件事；真正的区别只可能是：

1. **目标分布**：one-hot vs soft labels（人类/teacher 的分布）。
2. **输出约束**：free-form token vs 对选项的 softmax。
3. **后处理**：是否在 held-out 上拟合 temperature。
4. **数据**：是否做 hard-example mining / 合成增强。

如果用户的 Phase 2 只是“把 CE 换成 log loss”，那它不会带来任何新东西。**Brier 才是与 CE 不同的目标**（有界、对极端置信度惩罚不同、可做 Murphy 分解）；但 Brier 也是 proper，不是“更校准”的保证。

### 5.3 为什么叫“RL”？

这是整份记录最需要精确的地方。把“RL”拆成六种可区分的含义：

| # | 含义 | 机制 | 可观察特征 | 与 Jev 公开面的相容性 |
|---|---|---|---|---|
| 1 | **直接反传 proper score**（监督，不是 RL） | `p = softmax(f_θ)`；`∇S(p,y)` 直接反传 | 无探索噪声、无 baseline；结果对 sigma/采样数不敏感；CE 与 log loss 等价 | **高**。如果模型是 pointer/softmax head，这是最省事的实现 |
| 2 | **Policy gradient / REINFORCE** | 在 logit 上加噪声，采样 `p`，用 proper score 做 reward，group-mean baseline，CE 辅助 | 对 sigma、采样数、baseline、reward normalization 敏感；高方差；需要 KL/CE 稳定 | **高**。Laya 的 `rlcd_loss` 就是这种 |
| 3 | **Expert iteration / RFT / STaR** | 模型生成 rationale/答案，用 outcome 过滤，再 SFT | 数据随模型变强而增长；没有 reward 梯度；拒绝采样 | **中**。OpenJev-RLCD 的 “calibrate, then reinforce” 有一点像，但它是 PG |
| 4 | **Contextual bandit / off-policy** | 只观测被选动作的 outcome；IPS/DR；propensity | 需要 logged outcomes、探索策略、off-policy 修正 | **低**。TypeSafe 说不用客户数据；没有公开证据 |
| 5 | **在线探索 / 延迟 outcome** | 部署中探索，回收延迟反馈，持续训练 | 非平稳；性能随部署数据变化 | **很低**。与 “not trained on customer data” 冲突 |
| 6 | **“RL” 作为任务标签** | 不承诺具体优化器；指“优化校准决策这个任务” | 没有算法签名；CEO 说 “new task / North Star” | **最高**。这与访谈原话一致 |

**关键证据：CEO 自己把 RLCD 说成 task，而不是 algorithm。**

> Swyx: Which obviously you have some secret sauces to our knowledge. You’ve never actually published a paper or anything like that on it. No, right?
>
> Diogo Almeida: No, not yet.
>
> …
>
> Diogo Almeida: So the thing that really got… the thing that I’m calling to RLHF is the task of instruction following. It’s not about the PPO. That part doesn’t matter. It’s about setting a North Star… And for us, RLCD is this new task… Just like DPO and all of its descendants also do RLHF, despite not using the algorithm in that paper.

这段话把“RLCD 是第三条路”与“RLCD 是一种新 RL 算法”分开了。**“第三条路”是优化目标的分类，不是优化器的分类。** 这与 primer 的 “RLCD optimizes for a different output contract” 完全一致。

**公开实现告诉我们什么（[SECONDARY]）：**

Laya 的 `laya/train.py`（Apache-2.0，[GitHub](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py)）默认 `loss="rlcd"`，其 docstring 是：

> `loss="rlcd"` (the default) is the notebook’s objective: a GRPO-style term over noisy logit samples rewarded by `proper_reward`, plus soft cross-entropy against the target distribution.

`rlcd_loss` 的实现（节选）：

```python
def rlcd_loss(logits, target, mask, qtype, sigma, samples=4, w_sph=0.75, w_rps=1.0):
    k = mask.sum(-1, keepdim=True).float()
    fmask = mask.float()
    eps = torch.randn((samples,) + logits.shape, device=logits.device) * sigma * fmask
    eps = (eps - eps.sum(-1, keepdim=True) / k) * fmask
    z = logits.detach().unsqueeze(0) + eps
    q = torch.softmax(z.masked_fill(~mask, _MASKED), -1)
    with torch.no_grad():
        r = proper_reward(q, target.unsqueeze(0), qtype, fmask, w_sph=w_sph, w_rps=w_rps)
        adv = r - r.mean(0, keepdim=True)
        adv = adv / (adv.std() + 1e-6)
    logp = -(((z - logits.unsqueeze(0)) ** 2) * fmask).sum(-1) / (2 * sigma ** 2)
    return -(adv * logp).mean() + soft_ce_loss(logits, target, mask)
```

`proper_reward` 的定义是：

> Strictly proper scoring rule reward: log score + spherical score + ranked probability score.

即 `r = Σ_k t_k log q_k + 0.75 · <t,q>/||q||₂ − 1.0 · RPS(q,t)·1[score]`。这不是 Brier；也不是单一 log loss；是三个 proper score 的加权组合，再加 CE 辅助。默认 `sigma_start=0.4`、`sigma_end=0.1`、`samples=4`。

**这给了我们一个可运行的参考点，但不是 TypeSafe 的算法。** Laya 的作者在 Dev.to 文章里明确说 TypeSafe “launched without technical papers, without open weights, and with zero open training datasets”，并把自己的模型定位为独立重建（[Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)）。

**实证反例：RLCD 的 PG 部分不一定赢 CE。** sbert2s1（arXiv:2610.02486）在受控网格里比较了 CE、直接 proper score（“Proper”）、Laya 的 PG（“PG (released RLCD)”）、去掉 CE 的 PG、leave-one-out 无偏 PG、以及 reparameterization：

> The released RLCD recipe trails cross-entropy. With either head, CE, Proper and Reparam are within one point of each other, while the released PG recipe trails CE by 3.0 (PFR) and 2.5 (C) points (Holm p = .005).

> After temperature scaling, no objective is clearly better calibrated than cross-entropy.

这是**对“RL 是必需的”最有力的反证之一**，但它是在 biomedical sentence encoders 上、用 Laya 的 recipe、不是 TypeSafe 的 Jev。它说明：**proper score 本身可能是有用的，PG 的实现细节可能是负担。** 另一篇 OpenJev-RLCD 在 Qwen3-1.7B 的 reasoning 任务上则报告 RLCD 在 accuracy 上匹配或超过 SFT/GRPO，并在 selective prediction 上超过它们；两篇的 setting 不同，结论不必一致。**证据是混合的，不能把任何一方当成定论。**

**结论：** 公开证据不能确定 TypeSafe 的 RLCD 是 1、2、3 还是 6。最保守的表述是：**RLCD 是一个“校准决策”的任务定义；它的实现可能是监督 proper-score 训练，也可能是 policy gradient，或者两者混合。** 如果 Stim 要复现，应该把 1 和 2 都实现，用消融而不是叙事来决定。

### 5.4 数据引擎

**公开支持的部分：**

- 数据是全合成的（CEO 确认）。
- 数据是 TypeSafe 自己造的，不用客户数据（launch FAQ、models 文档）。
- 会针对模型的 jaggedness 做“surgical”修正（CEO 访谈）。
- TypeSafe 公开承认的 failure modes 包括：字面理解、数学/计数、日期、间接推理、长 state、对抗内容、矛盾指令、**选项顺序**、生成（[Jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13)）。

**重建的循环：**

1. **任务采样。** 从任务族/领域/难度分布采样 `(x, q, O)`；state 可以是原始文本、JSON、嵌套结构、长文档、图片（1.14+）。
2. **标签。** 优先用程序化 solver/规则/可执行验证；其次用 teacher 模型生成分布，再做一致性/审计；最后才是人类标注。**合成标签必须有独立验证，否则是循环。**
3. **选项集增强。** 洗牌、插入 distractor、插入 “none of the above”、near-duplicate、改变 K、改变 level 数、把同一问题改写成不同措辞。Laya 的公开数据管线明确做 “dynamically shuffles option orders, paraphrases questions, alternates between raw text and nested JSON states, and injects random distractor questions”；Kev 也做类似增强（但 Kev 不是 TypeSafe）。
4. **难例挖掘。** 按校准误差/分辨率/置信度分层找弱区：overconfidence、underconfidence、低 accuracy 高 ECE、OOD、选项顺序翻转、长 state、多标签 Noul。对这些区域生成定向合成任务。
5. **重训 + 验证。** 在独立的任务族/数据集上验证；temperature 在单独的 calibration pool 上拟合；test 只读一次。
6. **防止循环。** 不要把模型自己的高置信输出直接当标签；不要把 test 的输入分布带进合成器；记录 provenance；去重；做污染筛查。

**一个重要的“不要”：** 如果只用 ECE 选难例，会偏向“把置信度拉回 base rate”的样本，而不是“真正难”的样本。应该用 proper score 的分解（Reliability/Resolution）或 selective risk 来选。

### 5.5 训练流水线（伪代码）

```python
# [RECONSTRUCTION] Not TypeSafe's implementation.
# D_train / D_cal / D_test are split by task family, not by random items.
theta = load_backbone_and_head()
theta = sft(theta, D_train, loss="ce")                 # stage 1: supervised
theta = temp_fit(theta, D_cal)                         # stage 1b: scalar temperature

# stage 2: direct proper-score tuning (supervised; optional)
for batch in D_train:
    p = softmax(logits(theta, batch))
    loss = proper_score(p, batch.y, kind="log+brier+rps")
    theta -= lr * grad(loss)

# stage 3: policy-gradient RLCD (optional; Laya-style)
for epoch in range(E):
    sigma = anneal(0.4, 0.1, epoch, E)
    for batch in D_train:
        z = logits(theta, batch).detach()
        eps = normal(0, sigma^2 * project_sum_zero)     # G samples
        z_g = z + eps
        q_g = softmax(z_g)
        R_g = proper_reward(q_g, batch.y)               # log + spherical + RPS
        A_g = (R_g - mean(R_g)) / (std(R_g) + eps)
        pg = -(A_g * log_normal(z_g; z, sigma^2)).mean()
        loss = pg + ce_weight * ce(logits(theta, batch), batch.y)
        theta -= lr * grad(loss)

# stage 4: hard-example / synthetic data loop
for round in range(R):
    report = evaluate(theta, D_cal)                     # calibration + selective risk
    weak = select_weak_regions(report)                  # over/under-confidence, OOD, order
    D_new = generate_targeted_tasks(weak)               # programmatic labels + audit
    D_train = dedupe(D_train + D_new)
    theta = sft(theta, D_train, loss="ce")
    theta = proper_tune(theta, D_train)
    if use_pg: theta = rlcd_pg(theta, D_train)

# stage 5: final calibration + independent evaluation
T = fit_temperature(theta, D_cal)                       # per type / per K bucket
report = evaluate(theta, D_test, temperature=T)         # never used for selection
```

**每个阶段的“为什么”：**

- Stage 1 给 backbone 一个可用的表示；没有它，PG 的方差会吞掉信号（OpenJev-RLCD 的 “calibrate, then reinforce” 就是这个意思）。
- Stage 1b 的 temperature 是**后处理**，不是训练；它能把“排序好但幅度不对”的分布拉回校准。它不改变单题 argmax，但会改变 `p_max` 的数值，因而可能改变跨问题的 `p_max` 排序（Kev 的记录里，这正是 selective 指标会随温度变化的原因）。
- Stage 2 是最“像” proper-score 训练的一步；如果它已经足够，Stage 3 可能不必要。
- Stage 3 只有在“reasoning/表示需要被 reward 塑造”时才值得；否则它只是高方差。
- Stage 4 是数据闭环；它的收益应该用**数据消融**（同样的目标，不加新数据）来证明，而不是用“加了更多数据”来证明。
- Stage 5 的 test 必须与 calibration pool 分开，且只读一次。

### 5.6 评测协议

| 指标 | 定义 | 它回答的问题 | 陷阱 |
|---|---|---|---|
| Accuracy | `argmax p = y` | 选对了吗 | 不看概率 |
| Brier（sum/mean） | `Σ(p−y)^2` 或 `(1/K)Σ` | 概率整体质量 | 必须注明 sum/mean；跨 K 不可直接比 |
| Log loss / NLL | `−ln p_y` | 对正确类的置信度 | 无界；对极端错误非常敏感 |
| ECE | 分箱后 `Σ (n_b/N)·abs(acc_b − conf_b)` | 平均校准误差 | 分箱敏感；可被常数预测器刷低；隐藏局部错误 |
| Reliability curve | `conf_b` vs `acc_b` | 哪个区间过/欠自信 | 需要样本量；小 bin 噪声大 |
| AUROC / AUPRC | 排序质量 | 正负例分得开吗 | 与校准无关；类别不平衡时 AUPRC 更重要 |
| AURC / risk–coverage | 按置信度排序后的风险-覆盖曲线 | 选择性自动化值不值 | 依赖置信度排序；跨模型比较要同 coverage |
| Selective accuracy @ error | 固定 error budget 下的 coverage | 能自动处理多少 | 需要独立 calibration set 定阈值 |
| Decision utility | `E[u(a(p),y)]` | 下游任务赚/亏多少 | 需要成本矩阵；不同业务不可比 |
| OOD calibration | 在未见任务族上的 Brier/ECE | 漂移下还准吗 | 需要真正的外部分布；不能只用随机 split |
| Order robustness | 选项置换后的 flip rate / mean max Δp | 位置偏差多大 | 准确率不变不代表概率不变 |
| Per-type / per-K | 按原语、K、level 数分层 | 哪类问题坏 | 小样本；要报 n |
| Murphy 分解 | `REL − RES + UNC` | 是校准坏还是分辨力坏 | 只对 binary/grouping 有意义 |

**分箱敏感性的具体证据：** Deußer 等用 **15** 个等宽 bin，Kev 用 **10** 个；JevBench 用 **10** 个 top-label bin。三种口径下的 ECE 不可直接比较。Deußer 还报告了 pooled 0.028 与 unweighted mean 0.061 的差异——这就是“聚合方式会改变结论”的现成例子。

**一个可信的 TypeSafe 校准声明需要公开：**

1. 模型版本（`jev-1.13.0` 而不是 `jev-latest`）。
2. eval set 的来源、冻结时间、sha256、每项的 label。
3. 每项的 `probabilities` 与 `confidence`。
4. 分箱数、分箱边界、聚合方式（micro/macro）。
5. temperature 拟合集（必须是 held-out 数据集，不是训练集的随机 split）。
6. 选项顺序、模板、K、level 数。
7. 选择性风险的阈值来源与成本矩阵。
8. 训练数据是否包含 eval 的分布（污染声明）。

**在可读范围内，TypeSafe 一项都没有公开。** 这不是说它的校准是假的；只是说它不可复现。

### 5.7 重建的可信度与边界

- **强证据支持的部分：** 输出是分布；proper score 是自然的训练目标；校准与分辨力/效用是不同轴；选项顺序是真实风险。
- **中等证据支持的部分：** TypeSafe 的数据是全合成的、会针对 jaggedness 做定向修正；public RLCD 实现用 PG + proper score + CE；temperature 是必要的后处理。
- **弱证据/纯重建的部分：** TypeSafe 具体用哪种 proper score；是否用 PG；数据闭环的每一步；`box` 的 confidence 定义；1.13/1.14 的训练差异。
- **不应该做的事：** 把 Laya 的 `rlcd_loss` 当成 TypeSafe 的算法；把 archerhume 的架构探测当成事实；把 Di Zhang 的 Plackett–Luce 假设当成结论；把“proper score”当成“校准保证”。

---

## 6. 对用户总结的逐条核对

### 6.1 核对表

状态用词：**支持** = 公开证据直接支持；**大部分支持** = 方向对，但有一个关键措辞需要降级；**部分支持** = 只有一部分成立，或只在某种解释下成立；**不支持** = 没有公开证据；**被反驳** = 有公开证据与它相反；**不可证伪** = 无法用公开材料检验。

| # | 用户主张 | 状态 | 证据（逐字/链接） | 修正 / 细微差别 |
|---|---|---|---|---|
| 1 | RLCD 展开为 “Reinforcement Learning for Calibrated Decisions”；是 TypeSafe 的后训练方法；与 RLHF（人类偏好）、RLVR（可验证结果）并列；目标是让 stated probabilities 匹配真实 outcome frequencies | **大部分支持** | primer：“**Reinforcement learning for calibrated decisions** trains TypeSafe to return decisions and calibrated probabilities instead of generated text.”；launch：“training method we call Reinforcement Learning for Calibrated Decisions (RLCD)”；launch 表：RLCD “Calibrated decisions: answers with epistemically honest probabilities on System One tasks.”；primer：“Outcomes assigned a probability of `0.2` should occur about 20% of the time.” | **“方法”要降级为“任务/目标”。** CEO：“for us, RLCD is this new task”；“Just like DPO… also do RLHF, despite not using the algorithm in that paper.” 所以 RLCD 是**优化目标/输出合同**，不一定是新算法。 |
| 2 | 算法未发表：没有论文、训练代码或技术报告 | **对 TypeSafe 支持；对 RLCD 作为方法被反驳** | CEO 被问 “You’ve never actually published a paper or anything like that on it. No, right?” 答 “No, not yet.”；FAQ：“if you want to find out more, we’d have to hire you.”；但 Laya 的 `laya/train.py` 公开了 `rlcd_loss`/`proper_reward`；OpenJev-RLCD（arXiv:2609.38850）的论文声称公开代码 | **必须区分“TypeSafe 的 RLCD”与“RLCD 这个方法”。** 前者确实未发表；后者在 2026-09-30 之后已有公开实现。用户的原句“no public implementation”作为对方法族的描述是**被反驳**的。 |
| 3 | 模型应输出选项上的分布；strictly proper scoring rule（multiclass Brier 或 log loss/NLL）是自然核心；诚实报告真实条件概率最优；0/1 正确性奖励不行 | **分布：支持；proper-score 核心：推断；“Brier 或 log loss”过窄** | Choice 文档：“The full probability distribution across every option. The sum of all values is 1.”；System One：“their probabilities are optimized against outcomes”；Laya `proper_reward`：“log score + spherical score + ranked probability score”；archerhume：“A natural objective is log loss… Another is Brier loss… Both are proper scoring rules… It does not establish which loss TypeSafe uses” | **Proper score 是一个族，不是两个选项。** 公开 RLCD 实现用的是 log + spherical + RPS，再加 CE；不是 Brier。**Log loss = cross-entropy**，所以“SFT + log loss”不是新阶段。Brier 的 sum/mean 必须写明。properness 只在“真实条件分布、期望、模型类可表示”的意义上保证诚实。 |
| 4 | 两条实现路线：(A) 有标签的监督后训练 + Brier/log loss；(B) 从环境中的 outcome feedback 做策略优化 | **部分支持 / 不完整** | Laya 的 `rlcd_loss` 是 **PG on labels + CE**，不是环境反馈；OpenJev-RLCD 也是用可验证答案/标签，不是交互环境；TypeSafe 明确不用客户数据 | **漏掉了第三条、也是公开实现走的那条：** 在 logit 上加噪声，用 proper score 做 reward，用 group-mean baseline，加 CE 辅助。**“outcome feedback”不等于“环境”**；标签就是 outcome。路线 B 是三条里公开证据最弱的。 |
| 5 | 数据闭环是差异化部分：构造任务 → 输出分布 → 获得 outcome/feedback → 算 calibration reward → 找弱区（over/underconfidence、选项顺序、OOD）→ 生成定向合成任务 → 重训 → 独立验证 | **方向部分支持；机制不可证伪** | CEO：“all your data is synthetic” — “Yep.”；“They find the jaggednesses, and then they address them surgically”；FAQ：“We make all the data ourselves.”；Jaggedness 文档承认选项顺序等 failure modes | **“obtain outcomes”与“compute calibration reward”是重建，不是公开事实。** 合成数据的标签可能来自程序化 solver/teacher，不是真实世界 outcome；“calibration reward”没有被 TypeSafe 提过。方向（合成、数据-centric、针对 jaggedness）有支持；每一步的细节没有。 |
| 6 | Accuracy/discrimination、calibration、decision utility 是三个独立轴；指标含 accuracy、AUROC、Brier、log loss、reliability、ECE；ECE 分箱敏感、会掩盖局部 miscalibration | **支持（理论）；“三轴独立”表述过强** | Murphy 分解：`Brier = Reliability − Resolution + Uncertainty`；Guo et al. 2017；Nixon et al. 2019；Vaicenavicius et al. 2019；Deußer：pooled Choice ECE 0.028、median 0.028、unweighted mean 0.061、Emotion 0.279；JevBench：ECE 10 bins、Brier sum | **校准与分辨力不是正交的“两个东西”**，而是同一个 forecast 的不同性质；proper score 会把两者放在一个标量里。**Utility 不是统计性质**，它是 `p` 加成本矩阵的函数。ECE 不是 proper score；常数预测器可以 ECE≈0 但零分辨力。可靠性图也分箱，也会掩盖局部错误。 |
| 7 | 五阶段 R&D：Phase 1 SFT+CE；Phase 2 Brier/log-loss；Phase 3 难例+合成；Phase 4 环境反馈策略优化；Phase 5 独立校准/效用验证；Baseline A–D 比较 SFT、SFT+Brier、SFT+log-loss+hard mining、后者+环境反馈 | **部分支持（合理但缺关键基线/消融）** | sbert2s1：CE、Proper、Reparam 在一个点内；PG（released RLCD）落后 CE 2.5–3.0 分；温度缩放后没有目标明显更校准；OpenJev-RLCD 在 reasoning 任务上报 RLCD 更好 | **Phase 1 与 Phase 2 不是两个目标**（log loss = CE）；**Phase 4 是证据最弱、最贵的一步**，应条件化。缺：CE+温度缩放基线、直接 proper score（无 RL）基线、常数/基础率基线、数据消融、目标消融、算力匹配、held-out calibration pool、per-type/per-K、OOD、选项顺序、置信区间。 |
| 8a | TypeSafe 把 RLCD 描述为 RLHF/RLVR 之外的第三条路 | **支持** | primer：“Pretrained language models have been adapted in two major ways. TypeSafe adds a third. RLHF and RLVR are shown here for context; TypeSafe’s training path is RLCD.”；launch 表把三者并列 | 措辞是 “third”/“training path”，但紧跟着 “RLCD optimizes for a different output contract”。**“第三条路”是目标分类，不是算法分类。** |
| 8b | TypeSafe 说阈值应在历史标注数据上校准，低置信案例交人工复核 | **部分支持（部署建议，不是 RLCD 的性质）** | Cookbook：“The band is illustrative; it is neither a calibrated guarantee nor an optimized threshold. Set production boundaries from labeled examples and from the cost of incorrect decisions and of review.”；Confidence：“The correct threshold values depend on your domain and the performance of the model for your use case. Start with conservative thresholds, test with your own data…”；Deußer：binary probabilities “poorly placed relative to a fixed 0.5 threshold” | **TypeSafe 没有说“historical labeled data”这个 exact 短语**；它说 “labeled examples”“your own data”。它明确说示例 band **不是 calibrated guarantee，也不是 optimized threshold**。这是**应用侧逻辑**，不是 RLCD 训练出来的性质。 |
| 8c | 一份独立技术分析认为 strictly proper scoring rules 是核心 | **部分支持（分析明确声明不确定）** | archerhume：“A natural objective is log loss… Another is Brier loss… Both are proper scoring rules… This explains what such training tries to achieve. **It does not establish which loss TypeSafe uses, whether its pipeline is reinforcement learning in a narrow algorithmic sense, or whether every backbone weight is updated.**”；Di Zhang 的 Plackett–Luce 假设也是博客推断 | **“argues”太重，应该是“hypothesizes”。** 分析自己说不能确定 loss、是不是窄义 RL、是否全权重更新。Di Zhang 的 “schema-conditioned Plackett–Luce” 是另一种重建，和 Brier/log loss 不是同一个假设。 |
| 9 | Proper scoring rules 解释为什么诚实概率在期望上最优，但不保证有限模型/OOD 校准 | **支持** | Gneiting & Raftery 2007：strict properness 是期望性质；Gneiting, Balabdaoui & Raftery 2007：calibration 与 sharpness 是不同性质；Guo et al. 2017；archerhume：“Properness is also not a deployment guarantee. Finite data, model limitations, optimisation error, and distribution shift can all leave calibration imperfect.”；Deußer：ECE 随数据集大幅变化 | **补一个限定：** 如果模型类 misspecified，proper score 的最优报告是“模型类内最接近真实分布的那个”，不一定是真实条件分布；所以“诚实”是相对于模型类而言的。OOD 还多一层：真实分布变了。 |

### 6.2 三个数学例子：4.605、0.223、0.195

**4.605 与 0.223 可以精确复现，前提是自然对数：**

```
−ln(0.01) = 4.605170186…
−ln(0.80) = 0.223143551…
```

所以：

- `4.605` 是“把正确类的概率报到 0.01”时的 log loss；也就是“99% 自信但错了”的代价。
- `0.223` 是“把正确类的概率报到 0.80”时的 log loss；也就是“80% 自信且对了”的代价。

这两个数都是 **natural log**。如果换成 log10，4.605 会变成 2.0；如果换成 log2，会变成 6.644。用户总结里如果写的是 4.605/0.223，那它用的就是自然对数，和 Kev 的 NLL 口径一致。

**0.195 也可以精确复现，它就是用户例子的多分类 Brier（sum）。** 用户给的预测是 `p=(0.65,0.25,0.10)`，真实类为 A，即 one-hot `y=(1,0,0)`：

```
Brier_sum = (0.65−1)^2 + 0.25^2 + 0.10^2
          = 0.1225 + 0.0625 + 0.01
          = 0.195
```

所以这个数本身没有错。需要注意的是**尺度**：0.195 是 Brier，4.605/0.223 是 log loss；它们不是同一个量，不能直接比较“哪个惩罚更重”。用户在同一节里并列展示它们，容易被误读为同一尺度。

**Brier 的 sum/mean 具体差多少：**

```
p = (0.7, 0.2, 0.1), y = 0
Brier_sum  = 0.3^2 + 0.2^2 + 0.1^2 = 0.14
Brier_mean = 0.14 / 3             = 0.0467
Log loss   = -ln(0.7)             = 0.357
```

所以“Brier 0.14”与“Brier 0.047”可以是同一个预测；不注明 sum/mean 就没有可比性。Kev 用 **sum**；Laya 的 eval 用 **sum**；JevBench 用 **sum**；Di Zhang 的文章用 **1/2 sum** 来让二分类与 `(p−y)^2` 对齐。三种口径都存在。

### 6.3 “0/1 奖励会让策略总是选更可能的选项，而不是报 70%”——精确说法

用户的直觉方向是对的，但表述不精确。分三种情况：

1. **动作是一个单一选择，奖励是 `1[argmax = y]`。** 报告的概率不进入奖励。任何 argmax 相同的分布（例如 `(0.7,0.3)` 与 `(0.99,0.01)`）得到同样的奖励，所以**校准完全不受约束**。它不一定会“报 100%”；它只是没有理由报 70%。
2. **动作是从 `p` 中采样，奖励是 `1[sample = y]`。** 期望奖励是 `Σ_k p_k·1[k=y] = p_y`，对 `p` 是线性的；风险中性最优解是把全部质量放到最好的动作上（一个顶点）。所以 0/1 奖励会把策略推向**确定性**，而不是诚实分布。
3. **动作是“报告分布”，奖励是 0/1 的正确性。** 如果奖励只看 argmax，仍然对分布的形状不敏感；如果奖励是对每个选项的 0/1 正确性求和，那它其实是多标签 accuracy，仍然不是 proper。

**准确的表述：** 0/1 奖励对“被选动作的概率大小”不敏感，因此**不提供朝向校准的梯度**；它最多塑造 argmax。它不必然“总是选更可能的选项”，因为“选”与“报”是两个不同的动作；但在任何把概率当作动作一部分的设定里，0/1 奖励的最优解通常是退化的。

**证据：**

- 2512.19920：“Standard RLVR paradigms, utilizing binary reward signals, inadvertently incentivize models as good test-takers rather than honest communicators, encouraging guessing whenever correctness probability exceeds zero.”
- OpenJev-RLCD：“a correctness reward undoes the calibrated readout.”
- 2607.00164：“Reinforcement learning with verifiable rewards can in principle train calibrated probabilistic forecasters… In practice it degrades calibration.”
- OpenJev-RLCD 的结论：“Scoring groups of samples instead rewards disagreement, and a correctness reward undoes calibration; both are visible in one identity.”

**但也要给 0/1 奖励说句公道话：** 它对**准确率/排序**是有效的；问题在于它不约束概率。正确做法不是“完全不用 0/1”，而是“用 proper score 做校准目标，用 0/1/accuracy 做辅助或评估”。Laya 的 loss 就是 proper reward + CE，CE 本身也是 proper；它并不是纯 0/1。

### 6.4 用户总结里的内部不一致

1. **“方法” vs “任务”。** 用户把 RLCD 称为“post-training method”，但 CEO 称它为 “new task / North Star”。如果 RLCD 是任务，那么“算法未发表”是自然的；如果 RLCD 是算法，那么“第三条路”就应该是算法分类，但 TypeSafe 的原文说的是 output contract。两者只能选一个作为主框架。
2. **“没有公开实现” vs “有公开实现”。** 这句话在 2026-09-30 之后已经不成立（Laya、OpenJev-RLCD）。如果用户写作时间早于 9 月 30 日，应该标注日期；否则应改为“TypeSafe 没有公开其实现”。**Stim 的 `CONTEXT.md` 目前写的是 “has no public implementation”；按本记录的发现，这句话需要加时间限定，或改成 “TypeSafe has not published its implementation”。**
3. **“两条路线” vs “五个阶段”。** 两条路线是 A（监督）和 B（环境反馈）；五个阶段里又出现了“难例+合成”（数据路线）和“环境反馈”（B）。分类维度不统一：A/B 是“优化器/信号来源”，Phase 3 是“数据”，Phase 4 是“信号来源”。应该分开成“目标 × 数据 × 优化器”三个维度。
4. **“proper score 是核心” vs “RLCD 是 RLHF/RLVR 之外的第三条路”。** Proper score 是**损失函数**层面的概念；RLHF/RLVR 是**reward 来源**层面的概念。把它们并列，是 category mistake。Di Zhang 的文章明确指出了这一点：“RLHF and RLVR primarily describe where the reward comes from… RLCD describes what the model is trained to return.” 这不是说 TypeSafe 错了；TypeSafe 的 “third path” 本来就是任务层面的分类，用户把它读成了算法层面的分类。
5. **“阈值在历史标注数据上校准” vs TypeSafe 原文。** TypeSafe 说阈值是 application-specific，示例 band “neither a calibrated guarantee nor an optimized threshold”。用户把“应用侧建议”写成了“TypeSafe 的 RLCD 说法”，这是归属错误。
6. **数学尺度混用。** 4.605/0.223 是 log loss；0.195 是 Brier。把三者放在一起比较，会得出错误的“哪个惩罚更重”的结论。
7. **Phase 1 与 Phase 2 重复。** 如果 Phase 1 是“SFT + cross-entropy”，Phase 2 是“log-loss optimization”，那它们是同一个目标；真正的区别是 soft labels、Brier、RPS、temperature、数据增强。用户没有把这层说清楚。
8. **Baseline A–D 没有“最强简单基线”。** CE + temperature scaling 是公开证据里最强的简单基线（sbert2s1：CE/Proper/Reparam 在一个点内；温度缩放后没有目标明显更校准）。没有它，任何“RLCD 更好”的结论都不可信。

**最重要的三条校正（按重要性排序）：**

1. **RLCD 是任务/North Star，不一定是算法。** 这来自 CEO 的逐字访谈，是最硬的第一方证据。它不否定“RLCD 可能有一个有趣的算法”，但它否定“RLCD 这个名字本身就定义了一个算法”。
2. **“没有公开实现”已经过时。** Laya 的 Apache-2.0 代码和 OpenJev-RLCD 的论文/代码都在 2026-09-30 前后出现；它们不是 TypeSafe 的实现，但它们是 RLCD 这个方法族的公开实现。
3. **Proper score 是“族”，不是“Brier 或 log loss 二选一”。** 公开实现用 log + spherical + RPS + CE；Brier 只是其中一个可选成员。把 proper score 等同于 Brier/log loss，会把重建限制在一个比公开证据更窄的空间里。

---

## 7. 如果 Stim 要复现：实验设计

**目标不是“证明 RLCD 是对的”，而是回答五个可失败的问题：**

- **RQ1（目标）：** 在同等数据/算力下，direct proper score（Brier/log/RPS）是否比 CE + temperature scaling 带来更好的 OOD 校准，而不损失准确率？
- **RQ2（优化器）：** 在同等目标下，policy gradient（Laya 式）是否比直接反传 proper score 更好？如果更好，是因为 reward 形状、探索，还是因为 CE 辅助？
- **RQ3（数据）：** 在同等目标/算力下，合成数据 + 选项增强 + 难例挖掘是否比“更多同分布数据”更好？
- **RQ4（泛化）：** 校准收益是否在未见任务族/数据集上仍然存在，还是只在同分布上？
- **RQ5（效用）：** 校准收益是否转化为 selective risk / decision utility 的收益，还是只是 ECE 变好？

### 7.1 阶段

| 阶段 | 做什么 | 产物 | 失败时怎么办 |
|---|---|---|---|
| 0 | 冻结 eval；按**任务族/数据集**切 train/cal/test；实现所有基线；预注册成功标准 | 一份 `protocol.md` + 冻结的 suite hash | 如果切分不干净，停止，不要开始训练 |
| 1 | SFT + CE；在 cal 上拟合 temperature；在 test 上评一次 | 强基线 A | 如果 A 已经很好，后面的增益空间很小，先报告这一点 |
| 2 | direct proper score（log / Brier / RPS / 组合）；soft labels vs one-hot | 基线 B/C | 如果 B/C 不胜 A，说明 proper score 不是瓶颈 |
| 3 | policy gradient（Laya 式）：logit 噪声、group-mean baseline、CE 辅助；ablate sigma/samples/baseline/CE 权重 | 基线 D | 如果 D 不胜 C，PG 的“RL”部分没有价值 |
| 4 | 数据引擎：选项增强、难例挖掘、定向合成；与“同算力更多同分布数据”对比 | 基线 E | 如果 E 不胜 D+更多数据，数据闭环的增益是“更多数据”而非“更聪明数据” |
| 5 | 独立校准/效用验证：OOD、selective risk、cost matrix、order robustness | 最终报告 | 如果只在同分布赢，明确写“不泛化” |
| 6（条件） | outcome-feedback bandit：只在有可靠 logged outcomes 时做；否则跳过 | 可选基线 F | 没有 outcome 就不要假装有 |

**顺序的理由：** 先把“最强简单基线”做出来，再逐步加复杂度。sbert2s1 的教训是：CE + temperature scaling 很难被超过；如果不先把它做好，后面的“RLCD 更好”可能只是基线太弱。

### 7.2 基线

| 基线 | 目标 | 数据 | 作用 |
|---|---|---|---|
| **A. SFT + CE + temp** | CE（one-hot） | 同分布 | 最强简单基线；没有它，结论不可信 |
| **B. SFT + soft-label CE** | CE（teacher/人类分布） | 同分布 | 分离“软标签”与“proper score”的贡献 |
| **C. Direct proper score** | Brier / log / RPS / 组合，直接反传 | 同分布 | 分离“proper score”与“RL”的贡献 |
| **D. RLCD-PG** | proper reward + CE，Laya 式 | 同分布 | 分离“policy gradient”的贡献 |
| **E. RLCD-PG + 数据引擎** | D + 选项增强 + 难例 + 合成 | 增强 | 分离“数据”的贡献 |
| **F. RLCD-PG + outcome feedback** | D + logged outcomes | 条件 | 只在有可靠 outcome 时做 |
| **G. 常数/基础率** | 无 | — | 校准 sanity check：ECE≈0 但零分辨力 |
| **H. SFT + CE + order augmentation** | CE | 增强 | 数据-only 消融：选项增强本身值多少 |
| **I. A + selective threshold** | CE + temp + 阈值 | 同分布 | 效用基线：校准是否转化为 coverage |

**Baseline A–D 的对照关系：**

- A vs B：软标签值多少？
- A vs C：proper score 的形状值多少？
- C vs D：PG 值多少？
- D vs E：数据引擎值多少？
- D vs H：选项增强值多少？
- A vs I：阈值/效用值多少？

### 7.3 消融

1. **目标消融**（固定数据/算力）：CE vs log vs Brier vs RPS vs log+spherical+RPS vs PG。
2. **数据消融**（固定目标/算力）：同分布更多数据 vs 合成数据 vs 难例挖掘 vs 选项增强。
3. **探索消融**（固定目标/数据）：`sigma`（0.1/0.2/0.4）、`samples`（1/4/8）、baseline（none/group-mean/leave-one-out）、CE 权重（0/0.5/1）。
4. **选项顺序**：训练时 shuffle vs 不 shuffle；测试时置换；报 flip rate 与 mean max Δp。
5. **Temperature 拟合集**：in-distribution split vs held-out dataset pool；报 out-of-fold 与 in-sample 的差距。
6. **模型规模**：0.8B/4B/9B（或 Stim 的实际成员）；看校准收益是否随规模变化。
7. **K / level 数**：K=2/5/20/255；L=3/5/10；看校准是否随 K 退化。
8. **Per-type**：choice / noul / score / box 分开报；不要只报 pooled。

### 7.4 Held-out 协议

- **按任务族/数据集切分**，不按随机 item 切。同一 state 的多个问题、同一模板的变体、同一来源的近似重复，必须落在同一侧。
- **Calibration pool 与 test 分开**：temperature/阈值只在 cal 上拟合；test 只读一次；test 结果不用于选模型。
- **预注册**：在训练前写下主指标、成功阈值、基线、消融、停止规则。可以参照 Kev 的 “commit time is the registration time”。
- **Bootstrap 按 record 聚类**：同一 state 的多个问题是一个观测，不是 N 个。
- **报告负结果**：如果 PG 不胜 CE+temp，明确写；如果数据引擎只在同分布有效，明确写。

### 7.5 成功标准与算力

**主指标（预注册）：**

1. OOD Brier/log loss 相对 A 的改进 ≥ 一个预注册的效应量（例如 5% 相对改进），且 accuracy 不低于 A 的预注册下限（例如 −1 个点）。
2. OOD ECE 改进，同时 reliability curve 不出现新的局部过自信。
3. Selective risk：在固定 error budget（例如 1%/5%）下，coverage 不低于 A。
4. Decision utility：在给定成本矩阵下，期望效用不低于 A。

**次指标：** order robustness、per-type/per-K、Noul 的 under/over-confidence、box 的 coverage。

**算力：**

- 小模型族（0.8B/4B）每个 arm 约 1–8 GPU-hours（量级假设，按 Kev 的小模型配方估算）；3 seeds；报告 GPU-hours 与峰值显存。
- PG 的采样成本要算进去：同样 wall-clock 或同样 FLOPs 下比较，不能给 D 无限多算力。
- 先在小模型上筛，只有效应量足够才上大模型；不要一开始就 27B。

**一个法律/伦理提醒：** TypeSafe 的 [Terms of Use](https://typesafe.ai/legal/terms) 含有禁止 “reverse engineer or otherwise attempt to discover the source code of the proprietary software” 和用自动化程序 “monitor, copy, or download data or other content found on or accessed through the Site” 的条款。这些文字针对 Site/专有软件；它们是否覆盖 API 的研究性探针，需要法律判断。Stim 的 clean-room 规则本来就不允许从 probing 推导兼容性；任何对 Jev API 的研究性探针都应该先对照条款，且不应把 Jev 输出直接当训练标签（Kev 的 standing rule 也是 “No Jev output in training, ever”）。本记录 §8 的探针是**研究假设**，不是兼容性推导，也不构成法律建议。

### 7.6 用户计划里缺的东西

- **CE + temperature scaling** 这个最强简单基线。
- **Direct proper score（无 RL）** 这个关键对照。
- **常数/基础率** 这个校准 sanity check。
- **数据 vs 目标 vs 优化器** 的三维消融；用户的 A–D 把三者混在一起。
- **算力匹配**；PG 的采样成本不是免费的。
- **Held-out calibration pool by dataset**；随机 split 会高估校准。
- **Per-type/per-K/OOD/order** 的分层报告。
- **置信区间/预注册**；没有这些，5 个点的差异可能只是种子噪声。
- **负结果报告**；RLCD 可能在某些 setting 下不如 CE，这本身是有价值的结果。

---

## 8. 可证伪的预测

### 8.1 假设 → 预测 → 反证

| 假设 | 如果为真，应该看到 | 如果为假，应该看到 | 怎么测 |
|---|---|---|---|
| **H1：RLCD 是“任务/North Star”，不是特定算法** | TypeSafe 继续不发表算法；不同版本可能换训练细节；公开材料只谈输出合同/校准 | 出现一篇 TypeSafe 论文，明确写出 loss/optimizer；或 CEO 改口说“RLCD 是我们的算法” | 跟踪 TypeSafe blog/arXiv/GitHub；看版本变更说明 |
| **H2：RLCD 用 proper score 训练** | 对 Brier/log/RPS 的消融会改变校准；模型对“置信度—准确率”关系敏感 | TypeSafe 说用的是纯 RLHF/DPO/0-1 reward；或校准完全来自 post-hoc temperature | 需要 TypeSafe 公开 recipe；第三方只能做行为测试 |
| **H3：RLCD 用 policy gradient** | 对 sigma/采样数/baseline 敏感；训练曲线有高方差；需要 CE/KL 稳定 | 直接反传 proper score 就够；PG 消融无差异 | 在 Stim 的复现里做 C vs D 消融；这是最干净的办法 |
| **H4：数据引擎是差异化部分** | 合成数据/难例挖掘在同等算力下优于“更多同分布数据”；OOD 更好 | 数据-only 消融无增益；增益全来自更多数据/更大模型 | E vs D+更多数据；H vs A |
| **H5：阈值是应用侧的事** | TypeSafe 继续不给通用阈值；文档说“depends on your domain” | TypeSafe 发布“推荐阈值”或“校准保证” | 跟踪 docs/confidence、cookbooks、release notes |
| **H6：box confidence 也是校准的** | box 的 confidence 与 IoU/误差单调；有 coverage 保证 | box confidence 只是启发式；没有校准定义 | 在 1.14 上做 box 误差 vs confidence 的分箱；需要图片数据 |
| **H7：选项顺序偏差来自数据/架构** | 训练时 shuffle 会降低 flip rate；不 shuffle 会保留 first-option bias | shuffle 无效；偏差是架构/位置编码固有的 | H vs A；测试时置换；报 flip rate |

### 8.2 公开 API 的行为探针（研究用，不是兼容性推导）

这些探针只需要公开 API 和正常账号；它们不 reverse engineer，也不应大规模抓取。每一条都应在 Terms 允许的范围内、以最小请求量做。

1. **Noul vs 2-option Choice 的等价性。** 同一 state、同一措辞，问一个 `noul` 和一个只有 `yes/no` 两个选项的 `choice`。如果两者共享同一个 readout，`P(yes)` 应该接近 `choice` 的 `yes` 概率；如果差很多，说明 noul 和 choice 可能是不同的头/校准。
2. **选项顺序。** 同一 `choice`，把选项顺序置换 N 次；报 flip rate、mean max Δp、以及 first-option 的胜率。TypeSafe 自己说 1.13 有 first-option bias；这个探针可以量化它。
3. **K 的影响。** 固定问题和正确选项，改变 distractor 数量（K=2/5/20/255）；看正确选项的概率是否随 K 系统性下降，以及下降是否只来自 softmax 归一化。一个校准良好的 softmax 会给无关选项很小的概率，因此正确选项只会轻微下降；如果大幅下降，说明选项之间有交互或模型对 K 敏感。
4. **Score 的 level 独立性。** 同一 `score`，在末尾加一个无关 level（例如“无法判断”），看其他 level 的概率是否按比例重归一化，还是发生重排。前者支持 per-level binary + normalize；后者支持 softmax-over-levels。
5. **Score 的数值校准。** 用已知有序标签的数据，比较 `score` 的均值与真实 mean；看 `score` 是否在两端压缩（这是 Jaggedness 文档自己警告的 “weak in numerical calibration”）。
6. **Box 的 confidence。** 在已知 bounding box 的图片上，把 confidence 分箱，报 coverage（IoU ≥ 0.5 的比例）vs confidence；如果 confidence 不是校准的，coverage 不会随 confidence 单调上升。
7. **版本差异。** 同一组输入，比较 `jev-1.13.0` 与 `jev-1.14.0` 的概率；如果差异主要是温度/缩放，说明训练目标没大变；如果排序都变了，说明模型更新更大。
8. **`output_tokens` 的含义。** 改变答案的数值（0.0 vs 0.01）和选项数，看 `output_tokens` 是否随序列化长度变化，而不是随生成步数变化。archerhume 的结论是“billing figure”；这个探针可以复核。
9. **Latency scaling。** 固定 state，增加问题数/选项数，看 latency 是否随输入 token 增长而不是随输出选项数线性增长；这支持“一次前向 + 并行 readout”，而不是“逐选项生成”。

**注意：** 这些探针会消耗 token/配额，也可能触发 rate limit；不要用它们做“逆向工程”，也不要把结果直接写成“TypeSafe 内部就是这样”。它们只能排除一些架构/行为假设，不能证明训练目标。

---

## 9. Open questions / could not verify

1. **RLCD 到底是“任务”还是“算法”？** CEO 说 “for us, RLCD is this new task”“another North Star”；launch 说 “training method”。两者可以兼容（任务定义 + 未公开的实现），但对外表述没有统一。**要解决它，只需要 TypeSafe 发一篇 position paper 或技术报告。**
2. **TypeSafe 用的是 direct proper score、policy gradient，还是两者混合？** 公开材料没有 loss/reward/optimizer 字样。Laya 用 PG + CE；OpenJev-RLCD 用 PG + CE + 两阶段；sbert2s1 发现 PG 可能不如 CE。**TypeSafe 可能和两者都不同。**
3. **Reward 到底是什么？** 是 log score、Brier、RPS、spherical，还是“人类/teacher 的分布”？是 per-question 还是 per-request？是 outcome 还是 label？**没有第一方证据。**
4. **“All your data is synthetic” 的边界。** 是程序化生成，还是 teacher 模型生成？有没有人类审计？有没有真实数据做 seed？CEO 说 “we make all the data ourselves”，但没说生成器。**这决定了“数据引擎”是否可复现。**
5. **合成标签怎么验证？** 如果标签来自 teacher，teacher 的错误会被学生继承；如果来自程序化 solver，覆盖的 task family 有限。TypeSafe 没有公开验证协议。**这决定了“calibrated against outcomes”里的 outcome 是否可信。**
6. **有没有 post-hoc temperature scaling？** TypeSafe 说 “calibrated”，但没说校准是训练出来的还是拟合出来的。Laya 明确在 held-out 上拟合 per-type temperature；Kev 的经验是 temperature 能把 ECE 从 0.059 降到 0.0085。**TypeSafe 可能也做，但没有公开。**
7. **TypeSafe 有没有任何 per-item 的校准数字？** 没有。它拒绝公开 benchmark；1.14 的 “Measuring Intelligence” 只有图，没有数字/测试集。**第三方评测（Deußer、Vals、JevBench）是唯一可读的证据。**
8. **`box` 的 confidence 是什么？** 是预测方差、IoU 预测、还是启发式？它校准吗？**TypeSafe 没有定义；这是统一 proper-score 故事的最大缺口。**
9. **1.13 与 1.14 的训练差异。** 1.14 说 “small, subtle improvements to everything”，但没有说训练目标/数据是否变了。**跨版本比较概率时，不能假设只有推理变了。**
10. **专利。** Google Patents 超时；PatentsView 返回的是门户 HTML，不是 API 结果；USPTO 公开检索需要会话。**没有找到 TypeSafe/Jev/“calibrated decisions”的专利，但这是“未验证”，不是“不存在”。**
11. **第一方论文。** arXiv 的 `all:"TypeSafe AI"` 只返回第三方论文；`au:"Diogo Almeida" AND all:"calibrat"` 返回 0；CEO 说 “No, not yet”。**截至 2026-10-17，没有第一方技术论文。**
12. **Laya 的“released RLCD recipe”是否就是 sbert2s1 用的那个？** sbert2s1 说 “PG is the released RLCD recipe”，并引用 Laya；Laya 的 `laya/train.py` 确实是 PG + proper reward + CE。**两者高度一致，但我没有找到 Laya 作者对 sbert2s1 的确认。**
13. **OpenJev-RLCD 的代码是否可获取？** 论文说 `https://github.com/ZimmyGao/openjev-rlcd`；我尝试 raw README 时连接被重置，论文 HTML 可读。**代码存在性未直接验证；论文的方法/结果可读。**
14. **用户总结里的 0.195 是什么？** 它是用户所给例子 `p=(0.65,0.25,0.10)`、`y=A` 的多分类 Brier（sum）：`0.1225+0.0625+0.01=0.195`。**数值正确；需要注意的是它与 4.605/0.223 的 log-loss 尺度不同，不能直接比较。**
15. **“historical labeled data” 是 TypeSafe 的原话吗？** 不是。TypeSafe 说 “labeled examples”“your own data”“from the cost of incorrect decisions and of review”；示例 band 明确 “neither a calibrated guarantee nor an optimized threshold”。**用户的转述把“应用侧建议”提升成了“TypeSafe 的 RLCD 说法”。**
16. **“Jev is neither small nor an LLM” 与 “pretrained language models… adds a third” 是否矛盾？** 不必然。前者可能是产品/输出合同层面的说法；后者是训练来源层面。**TypeSafe 没有给出参数量/backbone，所以无法核实。**
17. **Jev 的概率是 direct readout 还是生成后解析？** TypeSafe 说 “does not generate text”“outputs all probabilities in parallel”；archerhume 的探测支持 direct readout。**但内部机制没有第一方披露。**
18. **TypeSafe 训练时是否做选项顺序增强？** 它承认顺序会影响答案；Laya 公开做 shuffle；Kev 也做。**TypeSafe 没有说它做不做。**
19. **TypeSafe 是否用客户数据做 eval（不是训练）？** 文档只说 “not trained on customer requests or responses”；eval 策略没有说。**如果未来要用 Jev 做 teacher，这一点需要读 DPA/terms。**
20. **“第三条路”是正式分类还是营销？** CEO 的 DPO 类比说明它更像“North Star/任务”而不是严格分类。**Di Zhang 的文章也指出 RLHF/RLVR/RLCD 不在同一分类层级。** 这不影响 RLCD 的价值，但影响“它是不是一个新算法”的判断。

**“TypeSafe 没有公开 X”的证据边界：** 本记录说的“没有公开”，限定在我实际搜过的范围：`docs.typesafe.ai` 的 128 个 sitemap URL、TypeSafe 官网 blog（8 篇，逐字读了 6 篇）、`typesafe.ai/legal/terms`、GitHub 组织 `typesafe-ai` 的 12 个 repo（README/changelog/code search）、arXiv 的 TypeSafe/Jev/RLCD 查询、HN Algolia、Crossref。**我没有搜到的，不等于不存在。** 特别是：X/Twitter、LinkedIn、YouTube、播客、会议 talk、专利全文、以及需要登录的 console/DPA，都没有被完整体检。

---

## 10. Sources

**访问日期：** 除特别注明外，均为 **2026-10-17**。**版本：** `jev-1.13.0` / `jev-1.14.0`；docs sitemap 最后更新 2026-10-14。

### 10.1 TypeSafe 官方文档

- [AI primer](https://docs.typesafe.ai/introduction/machine-learning-primer) — RLCD 展开、三条路、校准定义、RLHF 的 mode dropping。
- [System One](https://docs.typesafe.ai/concepts/system-one) — “trained for calibrated decisions… optimized against outcomes”。
- [Models](https://docs.typesafe.ai/models) — 1.13/1.14、价格、context、alias、not trained on customer data。
- [Primitives](https://docs.typesafe.ai/primitives) — 问题独立、每问一个 typed answer。
- [Choice](https://docs.typesafe.ai/primitives/choice) — 255 选项、probabilities 和为 1、question id 不进模型。
- [Score](https://docs.typesafe.ai/primitives/score) — 2–10 levels、score 是加权均值、level 逐项判断、看不到编号/邻居。
- [Noul](https://docs.typesafe.ai/primitives/noul) — 没有独立 confidence。
- [Box](https://docs.typesafe.ai/primitives/box) — 1.14、无 criteria、四个边、无 “not found”。
- [Confidence](https://docs.typesafe.ai/confidence) — 三个公式、阈值是应用侧的事。
- [Jev 1.13 jaggedness](https://docs.typesafe.ai/model-jaggedness/jev-1.13) — 选项顺序偏差、字面理解、数学/日期等 failure modes。
- [Consistency noul cookbook](https://docs.typesafe.ai/cookbooks/consistency_noul_cookbook) — “neither a calibrated guarantee nor an optimized threshold”。
- [LLM guardrails cookbook](https://docs.typesafe.ai/cookbooks/llm_guardrails) — review/action thresholds 是应用逻辑。
- [Legal](https://docs.typesafe.ai/legal) — DPA/MCA/Privacy/ZDR 入口。
- [Python SDK changelog](https://docs.typesafe.ai/sdk/python/changelog) — 0.7.5（2026-10-14）支持 Box/images。

### 10.2 TypeSafe 官方 blog / 第一方人物

- [Introducing System One Models & Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)（2026-09-15，Diogo Almeida）— RLCD 命名、三条路表、FAQ、workflow evals、255 两阶段。
- [Jev: For the builders AND dreamers](https://typesafe.ai/blog/jev-1-14)（2026-10-14）— 1.14、images、box、long context、内部测量。
- [The Bitterest Lesson](https://typesafe.ai/blog/bitterest-lesson)（2026-09-10）— 任务 > 数据 > 算力 > 算法。
- [Lies, Damned Lies, and Benchmarks](https://typesafe.ai/blog/antibenchmaxxing)（2026-09-11）— 拒绝公开 benchmark 的理由。
- [TypeSafe A raises Series AI](https://typesafe.ai/blog/series-ai)（2026-10-09）— 公司/融资背景。
- [Latent Space: Jev: System One Models for Prod, Not God](https://www.latent.space/p/jev)（2026-09-21，Swyx 访谈 Diogo Almeida）— “No, not yet”、RLCD 是 new task、全合成数据、jaggedness surgical、不用客户数据。**注意：这是播客逐字稿，可能有转写错误；引用时以原文为准。**
- [Terms of Use](https://typesafe.ai/legal/terms)（最后更新 2026-09-19）— TypeSafe AI, Inc.、地址、reverse engineering/automated scraping 限制。

### 10.3 GitHub / SDK

- [typesafe-ai](https://github.com/typesafe-ai) — 12 个公开 repo；`system-one-adapter-python`（MIT）、`typesafe-sdk-python`（MIT）、`typesafe-sdk-js`（MIT）、`WorkflowEvals`（Apache-2.0）、`skills`（MIT）、`typesafe-public-examples`（MIT）。
- [WorkflowEvals](https://github.com/typesafe-ai/WorkflowEvals) — 4 个 workflow、reference-model 评分。
- [typesafe-ai/system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python) — 官方参考 adapter；confidence 公式。
- [typesafe-ai/typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python) — 官方 SDK；`Box` 类型。

### 10.4 第三方分析 / 独立评测

- [archerhume, Jev’s Architecture Unmasked](https://archerhume.com/posts/jevs-architecture-unmasked)（2026-09-17）— 10,000 次 API 探测；direct readout、共享 state、问题隔离、选项交互、`output_tokens` 是计费。**SECONDARY，架构结论高度推测。**
- [Di Zhang, What Is RLCD? The Secret Behind Jev](https://di-zhang-llm.github.io/blog/what-is-rlcd-the-secret-behind-jev/)（2026-09-21）— “RLCD = multiway preference modeling + probability calibration”；Plackett–Luce 假设。**SECONDARY，博客推断。**
- [Tobias Deußer, Lorenz Sparrenberg, Rafet Sifa, Evaluating and Benchmarking the System One Model Jev](https://arxiv.org/abs/2609.37647)（arXiv:2609.37647，2026-09-29）— 37 数据集、346,009 请求、ECE/Brier/AURC、选项旋转。
- [Vals AI, An Independent Evaluation of TypeSafe’s Jev](https://www.vals.ai/blogs/independent-evaluation-of-jev)（2026-10-06）— 400 claim verification + LegalBench；1% error budget 下 95% coverage/1.6% error。
- [JevBench / Benchmark Heaven](https://github.com/fstandhartinger/jevbench)（v1.6.1，2026-10-06）— Calibration 轴 = hard tier ECE + gold 分布 fidelity；Brier sum；ECE 10 bins；Jev 1.13.0 63.29。
- [Decision Index](https://github.com/apolinario/decision-index)（0.3）— 37 个 benchmark、chance-corrected、public/private 分榜。
- [Laya (NandhaKishorM/laya)](https://github.com/NandhaKishorM/laya)（Apache-2.0，created 2026-09-18）— `laya/train.py` 的 `rlcd_loss`/`proper_reward`；`laya/common.py`；README；`laya/calibrate.py`。**这是公开的 RLCD 实现，但不是 TypeSafe 的实现。**
- [Laya README](https://github.com/NandhaKishorM/laya/blob/main/README.md) — “trained with reinforcement learning against strictly proper scoring rules (RLCD)”；“train with RLCD (proper-scoring-rule rewards, GRPO-style policy gradient)”。
- [Laya train.py](https://github.com/NandhaKishorM/laya/blob/main/laya/train.py) — `rlcd_loss`、`proper_reward`、`sigma_start=0.4`、`sigma_end=0.1`、`samples=4`。
- [Nandakishor M, Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)（2026-09-18）— Laya 的 RLCD 数学、数据管线、对 TypeSafe “no technical papers/weights/datasets” 的评论。
- [OpenJev-RLCD](https://arxiv.org/abs/2609.38850)（arXiv:2609.38850，2026-09-30，Zhimin Gao & Pichao Wang）— “working implementation of RLCD for reasoning models”；proper score reward；calibrate-then-reinforce；明确 “we claim nothing about Jev’s internal algorithm”。
- [From Retrieval to Typed Decisions](https://arxiv.org/abs/2610.02486)（arXiv:2610.02486，2026-10-01，Pritam Deka）— sbert2s1；Laya 的 reward；CE/Proper/PG 受控比较；PG 落后 CE 2.5–3.0 分；温度缩放后无目标明显更校准。
- [JevAdvBench](https://arxiv.org/abs/2609.31142)（arXiv:2609.31142，2026-09-25）— RLCD 模型的对抗 benchmark；state 注入会翻转 12.1% 决策。
- [Calibrated Decision Models for Autonomous Penetration-Testing Harnesses](https://arxiv.org/abs/2609.28940)（arXiv:2609.28940，2026-09-24）— JEV/Laya 作为 System One decision layers；提到 RLHV 等其他训练路径。

### 10.5 理论

- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1–3. [DOI](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
- Gneiting, T., & Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102(477), 359–378. [DOI](https://doi.org/10.1198/016214506000001437)
- Gneiting, T., Balabdaoui, F., & Raftery, A. E. (2007). Probabilistic forecasts, calibration and sharpness. *JRSS-B*, 69(2), 243–268. [DOI](https://doi.org/10.1111/j.1467-9868.2007.00587.x)
- Dawid, A. P. (1982). The well-calibrated Bayesian. *JASA*, 77(379), 605–610. [DOI](https://doi.org/10.1080/01621459.1982.10477856)
- Murphy, A. H. (1973). A new vector partition of the probability score. *Journal of Applied Meteorology*, 12(4), 595–600. [DOI](https://doi.org/10.1175/1520-0450(1973)012%3C0595:ANVPOT%3E2.0.CO;2)
- Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. *ICML*. [arXiv:1706.04599](https://arxiv.org/abs/1706.04599)
- Kuleshov, V., Fenner, N., & Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated regression. *ICML*. [arXiv:1807.00263](https://arxiv.org/abs/1807.00263)
- Nixon, J., Dusenberry, M., Jerfel, G., Nguyen, T., Liu, J., Zhang, L., & Tran, D. (2019). Measuring calibration in deep learning. *NeurIPS Workshop on Uncertainty & Robustness in Deep Learning*. [arXiv:1904.01685](https://arxiv.org/abs/1904.01685)
- Vaicenavicius, J., Widmann, D., Andersson, C., Lindsten, F., Roll, J., & Schön, T. B. (2019). Evaluating model calibration in classification. *AISTATS*. [arXiv:1902.06977](https://arxiv.org/abs/1902.06977)
- Christiano, P., Leike, J., Brown, T. B., Martic, M., Legg, S., & Amodei, D. (2017). Deep reinforcement learning from human preferences. *NeurIPS*. [arXiv:1706.03741](https://arxiv.org/abs/1706.03741)
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*. [arXiv:2203.02155](https://arxiv.org/abs/2203.02155) — **Diogo Almeida 是作者之一**，与 primer 的 “co-invented by Diogo Almeida” 说法一致（作者身份不等于“co-invented”，但支持其参与）。
- DeepSeek-AI (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. [arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., et al. (2024). Tulu 3: Pushing frontiers in open language model post-training. [arXiv:2411.15124](https://arxiv.org/abs/2411.15124) — **“a novel method we call Reinforcement Learning with Verifiable Rewards (RLVR)”**；这是 RLVR 命名的直接来源。
- Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256. [DOI](https://doi.org/10.1007/BF00992696)
- Sutton, R. S., McAllester, D., Singh, S., & Mansour, Y. (2000). Policy gradient methods for reinforcement learning with function approximation. *NeurIPS 12*, 1057–1063.
- Geifman, Y., & El-Yaniv, R. (2017). Selective classification for deep neural networks. *NeurIPS*. [arXiv:1705.08500](https://arxiv.org/abs/1705.08500)
- Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. [arXiv:2107.07511](https://arxiv.org/abs/2107.07511)
- Kadavath, S., Conerly, T., Askell, A., Henighan, T., et al. (2022). Language models (mostly) know what they know. [arXiv:2207.05221](https://arxiv.org/abs/2207.05221)
- Tian, K., Mitchell, E., Zhou, A., Sharma, A., et al. (2023). Just ask for calibration: Strategies for eliciting calibrated confidence scores from language models fine-tuned with human feedback. *EMNLP*. [arXiv:2305.14975](https://arxiv.org/abs/2305.14975)
- Xiong, M., Hu, Z., Lu, X., Li, Y., et al. (2024). Can LLMs express their uncertainty? An empirical evaluation of confidence elicitation in LLMs. *ICLR*. [arXiv:2306.13063](https://arxiv.org/abs/2306.13063)
- Kuhn, L., Gal, Y., & Farquhar, S. (2023). Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. *ICLR*. [arXiv:2302.09664](https://arxiv.org/abs/2302.09664)
- Elmachtoub, A. N., & Grigas, P. (2022). Smart “predict, then optimize”. *Management Science*, 68(1), 9–26. [DOI](https://doi.org/10.1287/mnsc.2020.3922)
- Wilder, B., Dilkina, B., & Tambe, M. (2019). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. *AAAI*. [arXiv:1809.05504](https://arxiv.org/abs/1809.05504)
- “Mitigating LLM Hallucination via Behaviorally Calibrated Reinforcement Learning” (2025). [arXiv:2512.19920](https://arxiv.org/abs/2512.19920) — 0/1 reward 鼓励 guessing。
- “Verifiable Rewards for Calibrated Probabilistic Forecasting” (2026). [arXiv:2607.00164](https://arxiv.org/abs/2607.00164) — proper score 的 RLVR 在实践中会退化。
- “Taming Overconfidence in LLMs: Reward Calibration in RLHF” (2024). [arXiv:2410.09724](https://arxiv.org/abs/2410.09724) — RLHF 的 overconfidence。
- “StartLux-Decision: AutoThink Decision Models with Reinforcement Learning for Calibrated Deliberative Decisions” (2026). [arXiv:2610.17309](https://arxiv.org/abs/2610.17309) — RLCD²，proper scoring rule + reasoning。

### 10.6 访问失败 / 无法验证

| URL / 目标 | 结果（2026-10-17） | 影响 |
|---|---|---|
| `https://evals.typesafe.ai/` | HTTP **451** `{"title":"Typesafe is not available in your region."}` | 看不到 TypeSafe 官方 workflow eval 的细节/数字 |
| `https://huggingface.co/...`（Laya、kev-suites 等） | `curl` 连接被重置（code 000） | 无法直接读 Laya 权重卡/HF 页面；改用 GitHub |
| `https://patents.google.com/?q=...` | 连接超时（code 000） | 专利检索未完成；不能断言“没有专利” |
| `https://api.patentsview.org/...` | 返回 USPTO 门户 HTML，不是 API 数据 | 同上 |
| `https://x.com/typesafeai` | 连接超时 | 没有读到 TypeSafe 的 X 帖子 |
| `https://www.linkedin.com/company/typesafeai/` | HTTP **999**（LinkedIn 反爬） | 没有读到 LinkedIn 公开帖 |
| `https://www.youtube.com/results?...` | 连接超时 | 没有读到 YouTube 访谈/演讲；只通过 HN/Latent Space 间接知道存在 |
| `http://web.archive.org/cdx/...` | 连接超时 | 没有做 archive.org 的历史页面比对 |
| `WebSearch`（本环境） | HTTP **403**（订阅/网络限制） | 改用 arXiv API、HN Algolia、Crossref、直接抓取 |
| `https://typesafe.ai/research` | HTTP **404** | 没有 research 页面 |
| `https://typesafe.ai/about` | HTTP **404** | 没有 about 页面（团队在 `/team`，未展开读） |
| `https://typesafe.ai/changelog` | HTTP **404** | 没有独立 changelog；版本看 docs/models + blog |
| `https://docs.typesafe.ai/changelog` | HTTP **404** | 同上 |
| `https://raw.githubusercontent.com/ZimmyGao/openjev-rlcd/main/README.md` | `curl: (35) Recv failure: Connection reset by peer` | OpenJev-RLCD 代码仓未直接读；论文 HTML 可读 |

---

## 11. 附录：公开陈述时间线

| 日期 | 事件 | 与本记录的关系 |
|---|---|---|
| 2026-09-10 | TypeSafe blog《The Bitterest Lesson》 | “doing the right task > data > compute > algorithms”；解释为什么 TypeSafe 强调任务/数据 |
| 2026-09-11 | TypeSafe blog《Lies, Damned Lies, and Benchmarks》 | 解释为什么拒绝公开 benchmark |
| 2026-09-15 | Jev / System One launch；RLCD 命名（文档当前稳定版为 `jev-1.13.0`） | 最重要的第一方文本（三条路表、FAQ、255 两阶段） |
| 2026-09-17 | archerhume《Jev’s Architecture Unmasked》 | 第三方架构探测；direct readout/共享 state/问题隔离 |
| 2026-09-18 | Laya repo created；Dev.to 文章 | 公开的 RLCD 实现（Apache-2.0）；对 TypeSafe 无论文/权重/数据的评论 |
| 2026-09-21 | Latent Space 访谈；Di Zhang 文章 | CEO “No, not yet”“RLCD is this new task”“all data is synthetic”；Plackett–Luce 假设 |
| 2026-09-24/25 | JevAdvBench（arXiv:2609.31142） | RLCD 模型的对抗鲁棒性；state 注入翻转 12.1% 决策 |
| 2026-09-29 | Deußer 等 Jev benchmark（arXiv:2609.37647） | 37 数据集、ECE/Brier/AURC；选项旋转准确率不变 |
| 2026-09-30 | OpenJev-RLCD（arXiv:2609.38850） | “working implementation of RLCD”；proper score + PG；明确不声称是 Jev 内部算法 |
| 2026-10-01 | sbert2s1（arXiv:2610.02486） | Laya 的 RLCD recipe 受控比较；PG 落后 CE 2.5–3.0 分 |
| 2026-10-06 | Vals AI 独立评测；JevBench v1.6.1 | 选择性自动化 95%/1.6% error；Calibration 轴 90.63 |
| 2026-10-09 | TypeSafe Series AI | 公司背景；不影响技术判断 |
| 2026-10-14 | Jev 1.14.0（`jev-preview`）；images/box/long context | 版本/`box`/context 的边界 |
| 2026-10-17 | 本记录观察日 | 所有引用以此日为准 |

---

**一句话总结：** TypeSafe 公开了 RLCD 的**名字、目标、输出合同和数据哲学**，但没有公开**算法**；公开的 RLCD 实现（Laya、OpenJev-RLCD）说明 proper score + policy gradient + CE 是一条可行路线，但受控实验也说明它不一定胜过 CE + temperature scaling。对 Stim 最有价值的不是“复刻 RLCD”这个口号，而是把**目标（proper score）、优化器（direct vs PG）、数据（合成/增强/难例）、校准（held-out temperature）、效用（selective risk）**拆成可消融的实验。
