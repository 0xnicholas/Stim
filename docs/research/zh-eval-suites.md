# 中文公开评估套件清点 — research record

**观察日期：** 2026-10-18（UTC）。**范围：** 为中文评估的可选公开可比轨（`eval-zh-v1.0` 的 comparability track）清点现有公开中文评估套件，判断其是否有资格进入该轨。**不涉及：** 自建 `eval-zh-v1.0` 验证轨、`eval-v1.0` 的修改、模型训练或评估。

**怎么读的：** GitHub REST API（`gh`：repo 元数据、README、LICENSE、git tree、文件内容）；Hugging Face 镜像 `hf-mirror.com`（dataset card、文件列表、raw 数据；`huggingface.co` 直连返回 HTTP 000）；下载并检查了 CMMLU 与 FinEval 的官方 zip、SafetyBench 的 `test_zh.json` 与 `test_answers_zh.json`；SafetyBench 论文读的是 arXiv HTML（`2309.07045v2`）。**没有使用二级综述**；`WebSearch` 在本环境返回 HTTP 403，因此发现主要靠 GitHub/HF/arXiv 直接检索。所有引用日期均为 2026-10-18，除非另注。

**判定口径（来自 #12、#18）：**

1. **许可：** 允许再分发与独立重跑；排除 SA / NC / unknown；代码许可与数据许可分开看。
2. **同构性：** 能映射到 System One 的 `noul` / `choice` / `score` 中的哪一种；选项与标签是否可由第三方重建；是否只是形式同构、而非分布同构。
3. **可复现：** 公开基线、固定划分、item-level 数据、评分代码；能否做 record-clustered bootstrap；主要单元是否 ≥100 条。
4. **风险：** 翻译偏差、训练集污染、版本漂移、sealed/private 部分。

**结论先行：** 严格按这四条，**没有一套原生中文套件完全合格**。唯一通过「许可 + item-level + 固定测试集 + 公开答案 + 评分代码」这条线的是 **SafetyBench（MIT）**，但它的中文集是中英互译的产物，缺少逐条 provenance，而且只覆盖 `choice`（2–4 选项）。因此它是**有条件候选**：只能做跨语言安全维度的生态对照，不能当作原生中文决策能力的证据，也不能替代自建验证轨。若把「原生中文 + 逐条 provenance」也作为硬条件，则答案是**没有**。

---

## 1. 清点总表

| 套件 | 语言 / 题型 | 规模 | 数据许可（upstream 证据） | 测试标签 | 固定划分 | 公开基线 / 评分代码 | 判定 |
|---|---|---|---|---|---|---|---|
| **SafetyBench** | 中/英；单选 2–4 项 | 中文 test 11,435；中文 subset 2,100；7 类 | **MIT**（GitHub `LICENSE`；HF card `license: mit`） | **公开**（`opensource_data/test_answers_zh.json`） | 有 test/dev；subset 固定 | 论文 25 模型零/五 shot；`eva.py` 逐类准确率 | **有条件合格**（跨语言安全对照） |
| **CValues-Responsibility-MC** | 中文；成对二选一 | 1,712（856/856） | **Apache-2.0**（repo `LICENSE`） | 公开（文件内 `label`） | 无 train/test 划分，整套为 eval | 论文 + `cvalues_eval.py`（LLM evaluator） | **边缘**（部分公开、偏好标签、无固定划分） |
| **GAOKAO-Bench** | 中文；客观题含单选/填空，另有主观题 | 2,811（客观 1,781；主观 1,030） | repo **Apache-2.0**；但题目为 2010–2022 高考真题，**无数据级授权/上游许可证据** | 公开（`Standard Answer`） | 按年份/科目文件组织，无 train/test | 论文基线；`Bench/*` 规则抽取 | **边缘/不合格**（上游权利未知；混合题型） |
| **CFBenchmark** | 中文金融；分类/识别/生成 | 3,917 文本；8 任务 | **Apache-2.0**（HF card） | 分类标签公开；生成任务无 gold | 有 test；非 MC 固定选项 | 论文；`codes/` | **边缘/不合格**（无原生选项，非决策题） |
| **C-Eval** | 中文；四选一 | 13,948；52 学科 | 代码 **MIT**；数据 **CC BY-NC-SA 4.0**（README、`LICENSE-DATA`、HF card） | **现公开**（test parquet 有 `answer`；2025-07-27 公告） | dev/val/test | 官方 leaderboard；lm-eval-harness | **不合格（NC+SA）** |
| **CMMLU** | 中文；四选一 | 67 学科；11,528 题 | HF card `cc-by-nc-4.0`；README 写 **CC BY-NC-SA 4.0**；repo 无 LICENSE | 公开（test CSV 有 `Answer`） | dev/test | 官方 leaderboard | **不合格（NC；许可文本不一致）** |
| **FinEval** | 中文金融；四选一 | 4,661 题；34 学科（2023 版） | 数据 **CC BY-NC-SA 4.0**（HF card）；代码 Apache-2.0 | **不公开**（test CSV 无答案；card 明说 labels not released） | dev/val/test | 论文/leaderboard；提交式 | **不合格（NC-SA + sealed test）** |
| **CLUE** | 中文；多任务（NLI/分类/阅读等） | 100K–1M 级 | HF card `license: unknown`；repo 无 LICENSE | 视子集而定 | 各子集有 train/dev/test | 官方 leaderboard | **不合格（unknown）** |
| **CMRC2018** | 中文；抽取式阅读 | ~10K+ | **CC BY-SA 4.0**（HF card） | 公开 | train/dev/test | 官方基线 | **不合格（SA；非 choice）** |
| **M3Exam** | 多语言含中文；多选/多模态 | 多语言 | **CC BY-NC-SA 4.0**（SEACrowd card） | 公开 | 固定 test | 论文 | **不合格（NC-SA；翻译/多语言）** |
| **Chinese SimpleQA** | 中文；短问答（开放） | 1K–10K | **CC BY-NC-SA 4.0**（HF card） | 公开 | 固定 test | 论文/leaderboard；LLM judge | **不合格（NC-SA；非 choice）** |
| **SuperCLUE** | 中文；综合/开放/客观 | 榜单 | repo **无 LICENSE**；repo 内无测试数据 | 不公开 | 私有测试集 | 官方榜单 | **不合格（unknown + sealed）** |
| **AlignBench** | 中文；开放生成 + GPT-4 judge | 多维度 | repo **无 LICENSE** | 无固定 gold；judge 打分 | — | GPT-4-0613 judge；1–10 分 | **不合格（unknown；judge-based，不是 Stim `score`）** |
| **Flames** | 中文；价值观/安全 prompt | 10+ 类，每类 46–249 | **Apache-2.0**（HF mirror card） | **无选项/无标签** | — | 需要外部 judge | **不合格（不同构）** |
| **LawBench** | 中文法律；20 任务 | 20 × 500 | 代码 **Apache-2.0**；README §许可要求**遵循各数据集创建者许可** | 视任务而定 | 零/一 shot 文件 | 论文；`evaluation/` | **不合格（上游混合许可；仅 3 个单选任务）** |
| **CBLUE** | 中文生物医学；8 NLU 任务 | 8 任务 | 代码 **Apache-2.0**；数据经 Tianchi 下载，数据条款未在 repo 说明 | 视任务而定 | train/dev/test | 论文/基线 | **不合格（非 choice；数据条款 unknown）** |
| **CFBench** | 中/英；约束遵循生成 | — | repo **无 LICENSE** | 无固定选项 | — | 论文 | **不合格（unknown；生成式）** |
| **AGIEval（中文子集）** | 中/英；考试聚合 | 多任务 | 代码 **MIT**；数据来自各考试/基准，**上游许可未逐项说明** | 公开 | 固定 test | 论文 | **边缘/不合格（聚合、上游 unknown）** |

> 表中「不合格」均指**不能进入当前许可口径下的公开可比轨**，不表示套件本身没有研究价值。`C-Eval`、`CMMLU` 等仍是很好的自建验证轨参考，只是不能作为可再分发的公开可比轨。

---

## 2. 有条件合格：SafetyBench

### 2.1 它是什么

SafetyBench 是 ACL 2024 的中文/英文安全多选题基准，11,435 道题，7 个安全类别（Offensiveness、Unfairness and Bias、Physical Health、Mental Health、Illegal Activities、Ethics and Morality、Privacy and Property）。论文摘要与 README 均给出同一规模。数据同时提供中文、英文、中文子集（每类 300 题，共 2,100 题）和 dev few-shot 示例。

### 2.2 许可证据

| 证据 | 内容 |
|---|---|
| GitHub `LICENSE` | MIT License，Copyright (c) 2023 thu-coai；允许 use/copy/modify/merge/publish/distribute/sublicense/sell，条件是保留版权与许可声明。 |
| HF dataset card | `license: mit`（`thu-coai/SafetyBench`）。 |
| README News | `2025/07/28`：“we’ve decided to fully open-source our test answers under `opensource_data`”。 |

**判定：** 数据与代码均为 MIT；允许再分发与独立重跑。这不是「代码 MIT、数据 NC」的混合情形，和 C-Eval/FinEval 不同。

### 2.3 数据与标签的实际情况（已逐条校验）

从 HF 镜像下载 `test_zh.json` 与 GitHub `opensource_data/test_answers_zh.json` 后核对：

- `test_zh.json`：**11,435** 条，`id` 全部唯一；字段为 `question`、`options`、`category`、`id`。
- `test_answers_zh.json`：**11,435** 条，字段为 `category`、`answer`；与题目 `id` 一一对应。
- 选项数分布：**2 项 5,223 条；3 项 445 条；4 项 5,767 条**。
- 类别分布（均 ≥100，实际均 >1,000）：Ethics and Morality 1,934；Unfairness and Bias 1,904；Offensiveness 1,805；Illegal Activities 1,778；Mental Health 1,566；Privacy and Property 1,299；Physical Health 1,149。
- 答案分布：0 → 4,098；1 → 4,333；2 → 1,707；3 → 1,297；**没有越界答案，没有空选项**。
- `test_zh_subset.json`：**2,100** 条（README 说每类 downsample 300 条）；对应 `test_answers_zh_subset.json` 存在。

**注意：** 题目在 HF，答案在 GitHub `opensource_data/`；HF 主分支的 dataset tree **没有** answers 文件。第三方要重跑，必须同时钉住 HF revision 与 GitHub commit。

### 2.4 同构性

- **`choice`：完全同构。** 每条就是「给定文本 + 2/3/4 个选项 + 一个正确选项」。可以直接构造成 System One 请求：`state` = `question` 字段（或把其中的指令与待判文本拆开），`question` = 固定指令（如「选择正确答案」），`criteria` = `options` 的映射，标签 = `answer` 索引。
- **`noul`：可派生，但不是原生。** 5,223 条二选项题可以改写成 `noul`（`criteria: {true: "是。", false: "否。"}`），但原数据集没有区分「noul 题」与「choice 题」；建议在可比轨里统一用 `choice`，另报一个 K=2 子面板，避免改变原题分布。
- **`score`：没有。** 数据集没有有序 level 标签，也没有概率加权均值的目标。
- **选项/标签可由第三方重建：是。** 选项在 `options`，答案在 `answer`，类别在 `category`；没有隐式标签。

**同构性的限度：** 这是**形式同构**（typed question over a state），不是**分布同构**。SafetyBench 是安全文本分类，不是 Stim 的通用决策任务；它只能覆盖 `choice` 的一个窄领域。

### 2.5 可复现性

| 项 | 状态 |
|---|---|
| 公开基线 | 有。论文 Table 2 报告 25 个中/英模型的零 shot 与五 shot 结果；官网 `llmbench.ai/safety` 有中/英/中文子集三个榜单。 |
| 固定划分 | 有。`test_zh` / `test_en` / `test_zh_subset` 固定；`dev_zh` / `dev_en` 是 few-shot 示例。 |
| item-level 数据 | 有。题目与答案都是逐条 JSON。 |
| 评分代码 | 有。`eva.py` 按类别算准确率并给 micro `Avg.`；`code/evaluate_base.py` 等是模型侧抽取示例。 |
| record-clustered bootstrap | **可以，但退化为 item-level。** 每条记录只有一个问题，cluster = `id`；没有同一 state 的 sibling questions。按 `id` 重采样即可满足「record-clustered」的字面要求，但不能检验同一 state 多问的相关性。 |
| ≥100/unit | 满足。7 个类别均 >1,000；中文子集每类 300。 |

**建议的跑法：** 用 `test_zh` 全量（11,435）做主读数，用 `test_zh_subset`（2,100）对齐公开榜单；两者分开报告，不混算。指标按 #12：acc、Brier、NLL、ECE（10 等宽桶）、mean_conf、confidence_bias、coverage/AURC；bootstrap 2,000 次、seed 0、95% CI，cluster = `id`。机会校正用逐题 `1/K`（K=2/3/4）。

### 2.6 风险

1. **翻译偏差（最重要）。** 论文原文：

   > “Using a commercial translation API, we translate the gathered Chinese data into English, and the English data into Chinese, thereby ensuring uniformity of the questions in both languages.”

   这意味着中文 test 里有一部分是英文题目的机器翻译，英文 test 里也有一部分是中文题目的机器翻译。中文集不是纯原生中文；论文只给聚合来源分布（Figure 3），数据集本身没有逐条的「原生/翻译」标记。对于中文能力声称，这是一个实质性的构造效度风险。

2. **逐条 provenance 缺失。** 数据字段只有 `question`/`options`/`category`/`id`；没有 source、translation flag、generator、split group。无法按来源做 group-aware 切分或污染筛查，也无法只取原生中文子集。

3. **版本漂移。** HF revision `70538631943dc1517ae9bb0733251809f4445d0e` 是 2023-09-14 的题目版；答案在 GitHub 2025-07-28 的 commit `027b3ca8434e56acfb51c7e7b18a3cb2aff09d4b` 才公开；GitHub main 的 merge commit 是 `960833692a1ec82b191ff3e902daca7993c35cc1`。HF 与 GitHub 的版本语义不同，必须双钉。

4. **污染。** SafetyBench 2023 年起公开，题目与答案都在公开网络；Qwen 等基座的预训练语料很可能包含它。无法从公开信息排除。需要把 SafetyBench 加入中文语料的整源排除清单，并在报告里披露污染不可排除。

5. **无 `score`、单领域。** 只覆盖安全文本分类的 `choice`；不能代表通用决策能力，也不能替代自建轨的 `score` 覆盖。

### 2.7 判定

**有条件合格，仅限「生态对照 / 跨语言安全」用途。** 它满足许可、item-level、固定测试、公开答案、评分代码、≥100/unit；不满足「原生中文 + 逐条 provenance + 无翻译偏差」。如果可比轨的定位是「跑几个公开套件，和生态数字对照」，它可以用；如果可比轨要承担「中文能力」的实质声称，它不够格。

---

## 3. 边缘候选

### 3.1 CValues-Responsibility-MC

**是什么：** `X-PLUG/CValues` 的「负责任」多选题子集；每条给出一个问题与两个回复，问哪个回复更好。公开文件 `dataset/cvalues_responsibility_mc.jsonl` 有 **1,712** 条（文件 5,091,985 bytes；`wc -l` 1,711，最后一行无换行），标签 **856 / 856** 平衡；字段 `prompt`、`label`、`meta_info`。

**许可：** repo 根 `LICENSE` 是 **Apache-2.0**；数据文件在 repo 内，没有单独的数据许可。**但** README 明说安全相关的 multi-choice safety prompts（2.6k）「因为内容敏感，暂时不开源」；公开的只是 responsibility 子集。`100PoisonMpts` 的源头在 ModelScope，另有条款，本记录未核验。

**同构性：** 成对 `choice`（选项 = 回复1 / 回复2），标签是偏好标签（哪个更安全/更负责任），不是事实正确性；chance = 0.5。`meta_info` 提供 `pos_resp`/`neg_resp`，第三方可重建选项。没有 `score`。

**可复现：** 有 paper、有 `cvalues_eval.py`（支持 ChatGPT/ChatGLM/MOSS/Ziya 等 evaluator），有逐条标签；**没有 train/test 划分**（整套 1,712 条就是 eval），没有公开的 per-item 模型输出。可以做 item-level bootstrap，cluster = 条目。

**风险：** 只公开 responsibility 子集，safety 子集 sealed；偏好标签依赖构造规则与 evaluator；上游 `100PoisonMpts` 条款未核验；中文原生（无翻译），但领域窄；污染风险存在。

**判定：边缘。** 只有在需要「中文安全/价值观成对比较」这一维度时才考虑；不能作为通用决策可比轨的主套件。

### 3.2 GAOKAO-Bench

**是什么：** 以 2010–2022 年中国高考题为数据集的评测框架；共 **2,811** 题，其中客观题 **1,781**、主观题 **1,030**。客观题文件按科目组织（语文、英语、数学、物理、化学、生物、政治、历史、地理等），含 `Standard Answer` 与 `Analysis`；主观题有评分脚本/人工评阅。

**许可：** repo `LICENSE` 是 **Apache-2.0**。**但**题目是高考真题，属于第三方作品；README 没有数据级授权、没有考试机构许可、没有逐题来源说明。repo 的 Apache-2.0 是否有效覆盖这些题目，是一个法律问题，不是本记录能解决的。按票面「upstream 证据 + 排除 unknown」，这构成**上游权利未知**。

**同构性：** 客观题里有一部分是四选一，可直接映射 `choice`；也有填空、完形、阅读理解等，不是统一 `choice`。主观题不是 typed decision。没有 `noul`/`score` 原生题。选项/答案可由第三方重建（`Standard Answer`）。

**可复现：** 有公开基线（论文表格）、固定年份/科目文件、逐题答案、规则抽取代码；可以做 item-level bootstrap。**但**科目内题量差异大，部分科目/年份的客观题可能 <100；需要先按单元计数，才能满足 ≥100/unit。

**风险：** 上游考试版权未知；题型混合；高考题在预训练语料中极常见，污染风险高；没有逐条 provenance；版本可能有 2023+ 补充集（README 指向 GAOKAO-Bench-Updates）。

**判定：边缘/不合格。** 除非完成上游法律核验，并只取「四选一 + 每单元 ≥100」的子集，否则不进可比轨。

### 3.3 CFBenchmark

**是什么：** 中文金融助手基准（InternLM·JiShi），3,917 条金融文本，8 个任务：公司/产品识别，情感/事件/行业分类，投资建议/风险提示/摘要生成。HF card 许可 **Apache-2.0**；有 `codes/` 评测代码与论文。

**同构性：** 识别与分类任务可以转成 `choice`（标签集合可由数据重建），但数据本身没有显式选项字段；生成任务（建议/风险/摘要）不是 typed decision。没有 `score`。**不是原生多选题套件。**

**可复现：** 有公开基线、代码、固定 test；但分类任务的选项集合、标签映射、生成任务的评分都需要额外定义；没有 per-item 概率基线。

**风险：** 数据来源含 EastMoney/OpenFinData 等，HF card 的 Apache-2.0 是否覆盖全部上游文本未逐项说明；`language: en` 标签与中文内容不一致（元数据风险）；任务分布与 Stim 的通用决策任务差距大。

**判定：边缘/不合格。** 只有在需要金融分类维度、并愿意自行定义选项映射时才考虑；不作为主套件。

### 3.4 AGIEval（中文子集）

**是什么：** 聚合型考试基准，含中国高考、LogiQA 等中文子集；repo `ruixiangcui/AGIEval` 是 **MIT**。**问题：** 数据来自多个考试/基准，repo 没有逐项上游许可说明；中文子集与 GAOKAO-Bench 大量重叠；不是中文专用套件；没有逐条 provenance。

**判定：边缘/不合格。** 不作为独立候选；若使用，按 GAOKAO-Bench 的上游核验标准处理。

---

## 4. 明确不合格的套件

### 4.1 C-Eval

- **规模/题型：** 13,948 道四选一，52 个学科，dev/val/test。
- **许可：** 代码 MIT；数据 **CC BY-NC-SA 4.0**。README 原文：

  > “This work is licensed under a MIT License.”
  >
  > “The C-Eval dataset is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.”

  repo 内还有独立的 `LICENSE-DATA`，正文开头即 “Attribution-NonCommercial-ShareAlike 4.0 International”；HF card 也是 `license: cc-by-nc-sa-4.0`。
- **测试标签：** 2025-07-27 README News 说 “We have released the complete C-Eval test set to the community!”；HF `computer_network/test-00000-of-00001.parquet` 的 schema 确实含 `answer` 与 `explanation`。**但**同一 README 仍保留 “Labels on the test split are not released…” 的旧句，文档自相矛盾；这是版本漂移的典型例子。
- **判定：不合格（NC+SA）。** 即使 test labels 现在公开，数据许可仍不允许无条件再分发/商用；不能进当前口径的可比轨。

### 4.2 CMMLU

- **规模/题型：** 67 个学科，11,528 道四选一（论文/README），dev + test。
- **许可：** HF card `license: cc-by-nc-4.0`；README「许可证」节写的是 **CC BY-NC-SA 4.0**；repo `haonan-li/CMMLU` 没有 LICENSE 文件。两种说法都含 NC，且文本不一致本身就是风险。
- **测试标签：** test CSV 列 `,Question,A,B,C,D,Answer`，答案公开。
- **判定：不合格（NC；许可文本不一致）。**

### 4.3 FinEval

- **规模/题型：** HF card 说 4,661 题、34 学科；README（更新版）说 26,000+ 题型的扩展基准——版本不一致。
- **许可：** 数据 **CC BY-NC-SA 4.0**（HF card）；代码 repo `LICENSE` 是 Apache-2.0。又是「代码宽松、数据 NC」。
- **测试标签：** HF card 明说 “Labels on the test split are not released, users are required to submit their results to automatically obtain test accuracy.” 下载的 `test/*.csv` 列只有 `id,question,A,B,C,D`，没有 `Answer`；`dev`/`val` 有答案。**sealed test。**
- **判定：不合格（NC-SA + sealed test）。** 即使忽略许可，也无法独立重跑 test 读数。

### 4.4 CLUE / CMRC2018

- **CLUE：** HF card `license: unknown`；repo `CLUEbenchmark/CLUE` 的 GitHub license 为 `null`。包含 OCNLI/CMNLI/TNEWS/CHID/CLUEWSC 等可能可映射 `choice` 的子集，但整体许可是 unknown；不能逐子集推定。**不合格（unknown）。**
- **CMRC2018：** HF card `license: cc-by-sa-4.0`；抽取式阅读，不是单选；**不合格（SA + 不同构）。**

### 4.5 M3Exam / Chinese SimpleQA

- **M3Exam：** 多语言、多模态、多层级考试基准，含中文；SEACrowd HF card `license: cc-by-nc-sa-4.0`。**不合格（NC-SA；翻译/多语言）。**
- **Chinese SimpleQA：** 中文短问答事实性基准；HF card `license: cc-by-nc-sa-4.0`；开放问答 + LLM judge，不是 typed choice/score。**不合格（NC-SA + 不同构）。**

### 4.6 SuperCLUE

- repo `CLUEbenchmark/SuperCLUE` 的 GitHub license 为 `null`；repo tree 只有 README、历史榜单、图片、model cards，**没有测试数据**。README 的榜单是提交/官方测评制。**不合格（unknown + sealed/private）。**

### 4.7 AlignBench

- repo `THUDM/AlignBench` 的 GitHub license 为 `null`；README 明说使用 **GPT-4-0613 作为 judge**，对模型回复做 1–10 分打分。它不是「caller 定义有序 level、模型输出分布」的 Stim `score`，也没有固定 gold 分数；judge 版本/提示词漂移会直接改变读数。**不合格（unknown + judge-based，不是 `score` 同构）。**

### 4.8 Flames

- HF 镜像 `opencompass/flames` 的 card 是 **Apache-2.0**，但数据文件只有 `prompt`（如 `morality_chinese_values.json` 是 76 条 prompt，字段只有 `dimension`/`subcomponent`/`prompt`），**没有选项、没有标签、没有参考答案**；需要外部 judge 打分。**不合格（许可虽宽松，但不同构）。**

### 4.9 LawBench

- repo `open-compass/LawBench` 的代码许可是 **Apache-2.0**；README §许可 原文：

  > “LawBench是由创建和转换的数据集混合而成。我们要求您遵循数据集创建者的许可证。请查看任务列表以获取每个任务的原始来源。”

- 20 个任务、每任务 500 例；其中只有 3 个是单选（1-2 知识问答、2-4 问题主题识别、2-8 论点挖掘），其余是生成/抽取/多选/计算等。数据源包括 CAIL、JEC_QA、LEVEN、AIStudio 等，各自许可不同。**不合格（上游混合许可；仅部分同构）。**

### 4.10 CBLUE

- repo 代码 **Apache-2.0**；数据不在 repo 内，README 要求从 Tianchi 下载（`dataId=95414`），没有在 repo 说明数据级许可。任务为 CMedEE/CMedIE/CDN/CTC/STS/QIC/QTR/QQR 等生物医学 NLU（NER、关系抽取、STS 等），不是 typed `choice`/`score`。**不合格（数据条款 unknown；不同构）。**

### 4.11 CFBench / CFBenchmark 之外的 CFBench

- `PKU-Baichuan-MLSystemLab/CFBench` 的 GitHub license 为 `null`；任务是约束遵循（constraints-following）生成式评测，不是固定选项决策。**不合格（unknown + 生成式）。**

> 注意：`CFBench`（约束遵循）与 `CFBenchmark`（同花顺/同济金融）是两个不同项目，不要混淆。

---

## 5. 同构性与统计口径

### 5.1 题型覆盖

| 套件 | `noul` | `choice` | `score` | 说明 |
|---|---|---|---|---|
| SafetyBench | 可派生（K=2 子集） | **原生**（2–4 项） | 无 | 安全文本分类 |
| CValues-Responsibility-MC | 可派生（成对） | **原生**（成对） | 无 | 偏好选择 |
| GAOKAO-Bench | 无 | 部分（四选一） | 无 | 混合题型 |
| CFBenchmark | 无 | 可转换（分类标签） | 无 | 非原生选项 |
| C-Eval / CMMLU / FinEval | 无 | **原生**（四选一） | 无 | 许可/标签不合格 |
| CLUE 子集 | 部分（NLI/ WSC） | 部分 | 无 | 许可 unknown |
| AlignBench | 无 | 无 | 不是 Stim `score` | judge 1–10 分 |

**结论：** 公开中文套件几乎全部是 `choice`；`noul` 只能从二选一/NLI 派生；**没有一套提供原生的 `score`（有序 level + 概率加权均值）任务。** 因此，中文可比轨无法覆盖 `score`；`score` 必须由自建 `eval-zh-v1.0` 承担。

### 5.2 Record-clustered bootstrap

- SafetyBench、CValues、GAOKAO、C-Eval、CMMLU、FinEval 都是**一条记录一个问题**；cluster = 条目 `id`。按 `id` 做 2,000 次 percentile bootstrap、seed 0、95% CI，即可满足「record-clustered」的字面要求，但**不能检验同一 state 上多个问题的相关性**。
- 有 passage-level 多问结构的套件（C3、CMRC、部分 LawBench/CFBenchmark 任务）要么许可 SA/unknown，要么不是 typed decision；在当前合格/边缘集合里没有「一个 state 多个问题」的可比套件。
- 这是可比轨的结构性局限：**它不能替代自建轨的 record-clustered 统计**。自建 `eval-zh-v1.0` 应显式构造同一 state 的 sibling questions，并按 record 重采样。

### 5.3 机会校正与聚合

- `choice`：逐题 baseline = `1/K`（K=2/3/4；SafetyBench 的 K 逐题不同）；`noul` 派生：baseline = 0.5；成对偏好：baseline = 0.5。
- 聚合：suite 等权 macro；只报「逐 suite + 聚合」两级；不造单一总榜分；不跨轨平均。
- 类别/学科单元：SafetyBench 7 类均 >1,000；CValues 1,712 但按 domain 分可能 <100，需先计数；C-Eval/CMMLU/FinEval 学科约 100+，但许可/标签不合格；GAOKAO 科目题量差异大。

---

## 6. 风险清单

### 6.1 翻译偏差

- **SafetyBench：** 中英互译（Baidu 翻译 API），中文集含翻译题；无逐条 flag。**这是条件合格的核心 caveat。**
- **M3Exam：** 多语言/翻译；NC-SA，直接排除。
- **C-Eval / CMMLU / FinEval / CValues / GAOKAO：** 原生中文（不是翻译），但许可/标签/上游权利有问题。
- **通用：** 不要把「中文机器翻译的英文基准」当作中文能力证据；如果要用，必须逐条标记并分层报告。

### 6.2 训练集污染

- 所有公开基准都可能进入基座预训练语料；C-Eval、CMMLU、GAOKAO、SafetyBench 尤其常见。公开信息无法排除。
- 需要：整源排除清单（中文语料构建时）、污染筛查、label-shuffled control、完全未见的 transfer 面板；报告里披露「污染不可排除」。
- 不能把「公开榜高分」直接当作能力证据；必须和自建轨的 control 一起看。

### 6.3 版本漂移

| 套件 | 漂移证据 |
|---|---|
| C-Eval | 2025-07-27 公告发布完整 test set；HF test parquet 含 `answer`；旧 README 仍说 labels not released。 |
| CMMLU | HF card `cc-by-nc-4.0` vs README `CC BY-NC-SA 4.0`；repo 无 LICENSE。 |
| FinEval | 2023 HF card 4,661 题 vs 更新版 README 26,000+ 题型；test labels 仍 sealed。 |
| SafetyBench | HF 2023-09-14 题目版 vs GitHub 2025-07-28 答案 commit；答案不在 HF 主分支。 |
| LawBench | README 说会持续更新任务列表；上游数据源多版本。 |
| SuperCLUE | 榜单按月/年更新；测试集不公开。 |

### 6.4 Sealed / private

- **FinEval：** test labels 不公开，提交式。
- **CValues：** safety multi-choice 2.6k 不公开；只公开 responsibility 1.7k。
- **SuperCLUE：** 测试集不公开，repo 无数据。
- **C-Eval：** 即使 test labels 现在公开，官方仍是 leaderboard 提交制；历史版本可能不同。
- **AlignBench：** 分数由 GPT-4 judge 生成，不是固定 gold；judge 版本/提示词是 hidden 变量。
- **所有套件：** 没有公开的 per-item 模型输出/概率；只能重跑自己的模型，不能复核别人的逐条结果。

### 6.5 许可陷阱

- **代码许可 ≠ 数据许可：** C-Eval（MIT code / NC-SA data）、FinEval（Apache code / NC-SA data）是典型。
- **repo 许可 ≠ 上游许可：** LawBench 明确要求遵循原数据集许可；GAOKAO 的考试题、CBLUE 的医院数据、CFBenchmark 的金融文本都可能另有条款。
- **镜像许可 ≠ 原始许可：** `opencompass/flames` 是镜像，card 写 Apache-2.0；使用前应回到原始发布者确认（本记录只把它当不同构，未依赖其许可）。
- **HF card 标签可能与 README 冲突：** CMMLU 是现成例子；以更严格的文本为准，并记录冲突。

---

## 7. 建议

1. **不要把中文可比轨建成「一堆公开套件」。** 按当前许可口径，合格集合最多是 SafetyBench 一个条件候选；它只能做跨语言安全维度的生态对照，不能承担中文能力声称。
2. **SafetyBench 的使用方式（如果采用）：**
   - 钉 HF revision `70538631943dc1517ae9bb0733251809f4445d0e` 与 GitHub commit `960833692a1ec82b191ff3e902daca7993c35cc1`；答案文件 commit `027b3ca8434e56acfb51c7e7b18a3cb2aff09d4b`。
   - 对题目与答案文件分别算 sha256；发布 manifest（HF + GitHub + 文件 hash + 类别映射 + 选项数）。
   - 主读数用 `test_zh` 全量 11,435；对齐榜单用 `test_zh_subset` 2,100；分开报告。
   - 指标：acc、Brier、NLL、ECE（10 桶）、mean_conf、confidence_bias、coverage@5%/1%、AURC；bootstrap 2,000/seed 0/95% CI，cluster = `id`；机会校正逐题 `1/K`。
   - 报告里明确写：**中文集含机器翻译题、无逐条 provenance、污染不可排除、只覆盖 `choice` 安全分类**。
3. **边缘候选只在有明确维度需求时使用：** CValues-Responsibility-MC（中文安全/价值观成对比较）、GAOKAO-Bench（中文考试知识，需上游法律核验 + MCQ 子集 ≥100/unit）、CFBenchmark（金融分类，需自定义选项映射）。不要把它们和 SafetyBench 混成一个「中文总分」。
4. **自建 `eval-zh-v1.0` 必须补 `score` 与 record 结构。** 公开套件没有原生 `score`；也没有合格的多问/同 state 套件。自建轨应覆盖 `noul`/`choice`/`score`，并按 record 聚类。
5. **不要使用：** C-Eval、CMMLU、FinEval、CLUE、CMRC2018、M3Exam、Chinese SimpleQA、SuperCLUE、AlignBench、Flames、LawBench、CBLUE、CFBench、CFBench 作为可比轨来源（按当前许可/同构口径）。它们可以作为方法参考或自建轨的题目灵感，但不能作为可再分发的公开套件。
6. **未完成上游核验前，不要把 GAOKAO-Bench / CValues / CFBenchmark 写成「合格」。** 它们的 repo 许可看起来宽松，但数据上游/部分公开/格式问题没有解决。

---

## 8. Sources

### 8.1 一手数据与许可

- **SafetyBench**
  - GitHub: <https://github.com/thu-coai/SafetyBench>；`LICENSE`（MIT）；`README.md`（2025-07-28 answers news；数据描述）；`opensource_data/test_answers_zh.json`；`opensource_data/test_answers_zh_subset.json`；`eva.py`；`code/evaluate_base.py`。
  - HF: <https://hf-mirror.com/datasets/thu-coai/SafetyBench>；revision `70538631943dc1517ae9bb0733251809f4445d0e`；`test_zh.json`、`test_zh_subset.json`、`dev_zh.json`、`test_answers_zh.json`（GitHub）。
  - 论文：<https://arxiv.org/abs/2309.07045>（ACL 2024；`2309.07045v2`）；翻译与数据构造见 §3.2；Table 2 零 shot 中/英结果。
  - 榜单：<https://llmbench.ai/safety>。
- **C-Eval**
  - GitHub: <https://github.com/hkust-nlp/ceval>；`LICENSE`（MIT）；`LICENSE-DATA`（CC BY-NC-SA 4.0）；`README.md`（Licenses 节；2025-07-27 test set news；旧 labels-not-released 句）。
  - HF: <https://hf-mirror.com/datasets/ceval/ceval-exam>；revision `617524a00b307ff6f9933702f724131fe12ca7ce`；card `license: cc-by-nc-sa-4.0`；`computer_network/test-00000-of-00001.parquet` schema 含 `answer`/`explanation`。
- **CMMLU**
  - GitHub: <https://github.com/haonan-li/CMMLU>；repo license `null`；`README.md` 许可证节（CC BY-NC-SA 4.0）。
  - HF: <https://hf-mirror.com/datasets/lmlmcat/cmmlu>；revision `efcc940752ea4a1ea94d2727f11f83858d64fc8e`；card `license: cc-by-nc-4.0`；`cmmlu_v1_0_1.zip`（`test/*.csv` 含 `Answer`；`dev/*.csv`）。
  - 论文：<https://arxiv.org/abs/2306.09212>。
- **FinEval**
  - GitHub: <https://github.com/SUFE-AIFLM-Lab/FinEval>；`LICENSE`（Apache-2.0）；`README.md`（扩展版 26,000+ 题型的描述）。
  - HF: <https://hf-mirror.com/datasets/SUFE-AIFLM-Lab/FinEval>；revision `ab300028f8836f14c28e80d4e96984bca9902a85`；card `license: cc-by-nc-sa-4.0`；`FinEval.zip`（`test/*.csv` 列 `id,question,A,B,C,D`，无 `Answer`；`dev`/`val` 有答案）。
  - 论文：<https://arxiv.org/abs/2308.09975>。
- **CValues**
  - GitHub: <https://github.com/X-PLUG/CValues>；`LICENSE`（Apache-2.0）；`README.md`（safety MC 2.6k 未开源；responsibility MC 1.7k 开源）；`dataset/cvalues_responsibility_mc.jsonl`（1,712 条；856/856）；`code/cvalues_eval.py`。
  - 论文：<https://arxiv.org/abs/2307.09705>。
- **GAOKAO-Bench**
  - GitHub: <https://github.com/OpenLMLab/GAOKAO-Bench>；`LICENSE`（Apache-2.0）；`README.md`（2,811 题；1,781 客观 / 1,030 主观；2010–2022）；`Data/Objective_Questions/*.json`；`Bench/objective_bench.py`。
  - 论文：<https://arxiv.org/abs/2305.12474>。
- **LawBench**
  - GitHub: <https://github.com/open-compass/LawBench>；`LICENSE`（Apache-2.0）；`README.md` §许可（遵循原数据集创建者许可）；任务列表（20 任务；3 个单选）。
  - 论文：<https://arxiv.org/abs/2309.16289>。
- **CLUE**
  - HF: <https://hf-mirror.com/datasets/clue/clue>；revision `28178267a609dd08bdc703dd6c931dfc2c2f4431`；card `license: unknown`。
  - GitHub: <https://github.com/CLUEbenchmark/CLUE>；repo license `null`。
- **CMRC2018**
  - HF: <https://hf-mirror.com/datasets/hfl/cmrc2018>；revision `137f2c45a24275fb68f6961c4d357f46288886aa`；card `license: cc-by-sa-4.0`。
- **M3Exam**
  - HF: <https://hf-mirror.com/datasets/SEACrowd/m3exam>；revision `ef8f14c46de02df9d07bf91254cdeeb72bbc9ac3`；card `license: cc-by-nc-sa-4.0`。
  - 论文：<https://arxiv.org/abs/2306.05179>。
- **Chinese SimpleQA**
  - HF: <https://hf-mirror.com/datasets/OpenStellarTeam/Chinese-SimpleQA>；revision `336305fc5f1da8184e3f70aff4473408418383a0`；card `license: cc-by-nc-sa-4.0`。
  - 论文：<https://arxiv.org/abs/2411.07140>。
- **SuperCLUE**
  - GitHub: <https://github.com/CLUEbenchmark/SuperCLUE>；repo license `null`；tree 无测试数据；README 榜单/报告。
  - 网站：<https://www.superclueai.com/>。
- **AlignBench**
  - GitHub: <https://github.com/THUDM/AlignBench>；repo license `null`；README（GPT-4-0613 judge；1–10 分）。
  - 论文：<https://arxiv.org/abs/2311.18743>。
- **Flames**
  - HF mirror: <https://hf-mirror.com/datasets/opencompass/flames>；revision `345b0d1615d46da917bd51b6f3f0f913bc5e966e`；card `license: apache-2.0`；数据文件为 prompt-only（如 `morality_chinese_values.json` 76 条）。
- **CBLUE**
  - GitHub: <https://github.com/CBLUEbenchmark/CBLUE>；`LICENSE`（Apache-2.0）；README（数据从 Tianchi `dataId=95414` 下载；8 个 NLU 任务）。
  - 论文：<https://arxiv.org/abs/2106.08087>。
- **CFBenchmark**
  - HF: <https://hf-mirror.com/datasets/TongjiFinLab/CFBenchmark>；revision `3c825c2c9a63634736630e207c15cd27ab9633b6`；card `license: apache-2.0`；README（3,917 文本；8 任务）。
  - 论文：<https://arxiv.org/abs/2311.05812>。
- **CFBench（约束遵循）**
  - GitHub: <https://github.com/PKU-Baichuan-MLSystemLab/CFBench>；repo license `null`。
- **AGIEval**
  - GitHub: <https://github.com/ruixiangcui/AGIEval>；`LICENSE`（MIT）；数据为多考试/基准聚合。
  - 论文：<https://arxiv.org/abs/2304.06364>。

### 8.2 访问失败 / 无法验证

| 目标 | 结果（2026-10-18） | 影响 |
|---|---|---|
| `huggingface.co` 直连 | HTTP 000（连接失败） | 改用 `hf-mirror.com`；HF 页面上的评论/讨论未读。 |
| `raw.githubusercontent.com` | 连接失败/重置 | 改用 GitHub REST `contents` + `Accept: application/vnd.github.raw`。 |
| `WebSearch` | HTTP 403（订阅/网络） | 发现靠 GitHub/HF/arXiv 直接检索；没有二级综述。 |
| `modelScope.cn`（100PoisonMpts） | 未访问 | CValues 上游条款未核验。 |
| 各套件的官方 leaderboard 提交页 | 未逐一登录/提交 | 只读公开 README/论文数字。 |

---

## 9. Open questions / could not verify

1. **C-Eval 的 test label 发布范围。** HF `test` parquet 已有 `answer`；README News 说 2025-07-27 发布完整 test set；但同一 README 仍说 labels 不公开。需要官方 changelog 或 commit 来确认「完整 test set」是否包含全部 52 学科的答案。无论答案是否公开，NC-SA 许可已经 disqualify。
2. **SafetyBench 的逐条来源/翻译 flag。** 数据只有 `category`/`id`/`question`/`options`；论文只给聚合来源分布。无法只筛「原生中文」子集。需要作者补充 per-item provenance，或接受「中英互译」的整体 caveat。
3. **SafetyBench 的公开榜单是否用中文子集。** README 说有三个 leaderboard（中/英/中文子集）；论文 Table 2 是全量。对齐榜单时应确认具体 split，并在报告里写清。
4. **CValues 上游 100PoisonMpts 的条款。** repo LICENSE 是 Apache-2.0，但 100PoisonMpts 在 ModelScope 上另有 dataset card/条款；本记录未访问。若要把 CValues 从 edge 提升为 qualified，必须先核验。
5. **GAOKAO-Bench 的考试题上游权利。** repo 是 Apache-2.0，但高考真题的版权/许可没有在 README 说明；这不是技术问题，需要法律判断。
6. **CFBenchmark 的 split 与上游文本许可。** HF card 是 Apache-2.0，但数据源含 EastMoney/OpenFinData；没有逐项许可台账；分类任务的选项集合也需要自定义。
7. **C3 / CMB / 其他中文套件未逐一清点。** 本记录覆盖了有论文/榜单/公开数据的主要套件；如果要把 C3、CMB、CMMLU-Pro、C-Eval 变体等纳入候选，需要按同一口径补做 license + split + answer + format 核验。当前结论不依赖它们：已有足够证据表明主流中文套件以 NC/SA/unknown 为主。
8. **没有公开的 per-item 模型输出。** 所有套件都只公开题目/答案，不公开别家模型的逐条概率；因此无法复核 leaderboard 的逐条数字，只能重跑自己的模型。
9. **SafetyBench 的答案文件在 HF 上缺失。** HF 主分支只有题目；答案在 GitHub。若发布 Stim 的评估 manifest，必须同时记录两个源；只钉 HF revision 不够。
10. **`score` 的公开中文套件是否存在。** 本记录没有找到原生的「有序 level + 概率分布」中文套件；AlignBench 的 1–10 分是 judge 标量，不是 Stim `score`。如果后续发现，需单独核验。

---

**一句话总结：** 按 #12/#18 的严格许可口径，中文公开评估套件里只有 **SafetyBench（MIT）** 通过了「可再分发 + item-level + 固定测试 + 公开答案 + 评分代码」这条线，但它是中英互译、无逐条 provenance、只覆盖 `choice` 的安全分类集，因此只能作为**有条件的跨语言安全对照**；如果要求原生中文 + 完整 provenance，则**没有合格套件**。边缘候选是 CValues-Responsibility-MC、GAOKAO-Bench、CFBenchmark；其余主流中文套件均因 NC/SA/unknown、sealed test 或不同构而明确不合格。
