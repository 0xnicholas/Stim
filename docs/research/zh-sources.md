# 中文公开训练源清点与许可证据（data-zh-v1.0）

**票：** [#17 中文公开训练源清点与许可证据](https://github.com/0xnicholas/Stim/issues/17)
**观察日期：** 2026-10-18（UTC）
**范围：** 为 `data-zh-v1.0` 清点可用的中文公开训练源，按 [#11 训练数据与生成策略](https://github.com/0xnicholas/Stim/issues/11) 的 permissive-only、逐条 provenance、双 teacher、2k pilot 七项闸给出可用性判断与 15k 配比建议。**本记录不训练、不生成数据、不改 `data-zh-v1.0`；所有配额都是建议值，不是已生成记录。**

**怎么读的：** 直接读上游 `LICENSE`、数据集卡、官方仓库与论文；`huggingface.co` 和 `raw.githubusercontent.com` 在本环境连接被重置，因此 HF 侧通过 `hf-mirror.com` 的 API/raw 端点读取，GitHub 侧通过 `gh api` 读取，URL 仍给 canonical 地址。搜索引绑定不可用（403），发现主要靠 HF dataset API 的 license/language 过滤、GitHub 仓库搜索和论文。个别数据集的实际文件通过镜像下载后计数（`jq length`、逐行 JSON/CSV 解析）。

**证据分层：**

- **H**：直接读到上游 `LICENSE` 或数据集卡原文，且许可条款明确。
- **M**：只读到卡片/元数据/推算，或上游数据条款未逐条核验；入料前必须补上游证据。
- **X**：许可或污染条件不满足 #11，排除。

---

## 1. 结论先行

1. **可用的中文 permissive 公开源足够填满 public 6k，但结构偏科。** 最稳的九个训练源是 MASSIVE（CC-BY-4.0）、CrossWOZ / KdConv / COLD（Apache-2.0）、DuReader-robust（Apache-2.0 卡片）、CMATH（CC-BY-4.0 + 上游 MIT）、China laws（CC0-1.0 卡片）、HelpSteer3 zh（CC-BY-4.0）和 Nemotron Safety zh（CC-BY-4.0）。它们覆盖基础意图、任务型对话、知识型对话、安全/冒犯、短文档问答、法律长文档、小学数学、helpfulness score 与安全分类。
2. **最大的缺口是 devtools/code。** 没有找到达到数百条规模、且能逐条满足 permissive-only 的中文代码决策源：CommitPackFT 的 CJK 命中率极低（抽样 145,513 条中 23 条含 ≥2 个 CJK 字符，且多数是日文汉字；实际中文更少）；ToolACE 的中文样本约 20/11,300；UltraData-Code 的卡片明确要求遵守每个源仓库的许可，并禁止未经许可的原样再分发。**因此 public 6k 不给 code 配额，devtools/code 全部由合成 1,200 + 程序化 1,000 补。**
3. **传统中文训练量偏少。** MASSIVE zh-TW 提供 16,521 条公开记录（train 约 11.5K），足够抽 400 条；其余推荐训练源基本是简体。建议 15k 中 zh-TW 至少 1,500 条（public 400 + synthetic/programmatic 1,100），并在 pilot 中至少占 15%。
4. **标准中文 NLU 基准大多不可用。** CLUE 系列卡片是 `license: unknown`，CMRC2018 是 `cc-by-sa-4.0`，C-Eval 是 `cc-by-nc-sa-4.0`，CMMLU 是 `cc-by-nc-4.0`。这既排除了训练，也意味着 #18 的中文评估套件不能直接发布这些题目的 item-level 数据；可用的 permissive eval 源见第 4 节。
5. **建议的 15k 结构：** public 6k / synthetic 6k / programmatic 3k；按阶段 basic 7,200 / docs 2,600 / hard 3,000 / code 2,200。public 组合与合成缺口见第 6 节；2k pilot 见第 7 节。
6. **M-level 必须补证后再入料：** DuReader-robust 的上游数据许可、China laws 的 CC0 声明与 flk.npc.gov.cn 条款、HelpSteer3 zh 的 PII、Nemotron Safety zh 的内容/翻译链、MASSIVE 的公开 split 精确值（本环境 S3 下载受限，计数来自论文与卡片）。

---

## 2. 判定规则（沿用 #11）

- **许可白名单：** Apache-2.0 / MIT / CC-BY-3.0 / CC-BY-4.0 / CC0-1.0。SA、NC、`unknown`、`other`、无声明一律排除；GPL 也不在 #11 的白名单内。
- **逐条 provenance：** 每条记录至少带 `source`、`license`、`license_evidence_url`、`source_revision`、`generator`、`seed`、`split`、`group_id`。本记录给的是**源级**证据；入料时还要把源级许可落到每条记录。
- **记录单位：** 一个 state + 一批 noul/choice/score 问题；同一 state 的兄弟问题/变体共享 `group_id`，切分按 `group_id` 整体分配。
- **公开源上限：** 任一公开源不超过公开部分的 20%。public 6k 时上限为 1,200 条；本配比把 MASSIVE 正好压在上限。
- **双 teacher：** Qwen3.5-35B-A3B 走量、Qwen3.8-27B 走质量关键/双签；只用 open-weight；闭源模型只做 judge/filter，不做训练标签；Jev 输出永不入训。
- **pilot 七项闸：** 见第 6 节；2k pilot 过闸才放量到 15k。

---

## 3. 推荐用于训练的公开源

### 3.1 总表

| 源 | 阶段 | 许可 | 证据 | 置信 | 可用原始量 | 建议配额 | 语言变体 | PII / 敏感内容 |
|---|---|---|---|---:|---:|---:|---|---|
| MASSIVE `zh-CN` + `zh-TW` | basic | CC-BY-4.0 | [HF LICENSE](https://huggingface.co/datasets/AmazonScience/massive/blob/main/LICENSE)、[论文](https://arxiv.org/abs/2204.08582) | H | 16,521 / locale 公开（论文全局 587K train / 104K dev / 152K test + 153K held-out） | 1,200（800 简 + 400 繁） | 简体 + 繁体 | 众包本地化助理语句；丢弃 `worker_id`/`judgments`；预期无 PII |
| CrossWOZ | basic | Apache-2.0 | [LICENSE](https://github.com/thu-coai/CrossWOZ/blob/master/LICENSE)、[README](https://github.com/thu-coai/CrossWOZ) | H | 5,012 train / 500 val / 500 test 对话；102K utterances | 800 | 简体 | 模拟任务型对话；标注者 ID；无真实 PII |
| KdConv | basic | Apache-2.0 | [LICENSE](https://github.com/thu-coai/KdConv/blob/master/LICENSE)、[README](https://github.com/thu-coai/KdConv) | H | 4,500 对话 / 85,596 utterances；train 3,600 / 66,538 | 500 | 简体 | 影视/音乐/旅行公开实体；无 PII |
| COLD | basic / safety | Apache-2.0 | [LICENSE](https://github.com/thu-coai/COLDataset/blob/main/LICENSE)、[README](https://github.com/thu-coai/COLDataset) | H | 25,726 train / 6,431 dev / 5,323 test；共 37,480 | 600 | 简体 | 冒犯/仇恨内容（race/gender/region）；需内容警告与 PII 扫描 |
| Nemotron Safety Guard zh | basic / safety | CC-BY-4.0 | [HF card](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3) | H 许可 / M 内容 | 37,335 train / 2,291 valid / 2,950 test；中文 42,576 | 500 | 简体（待核） | 合成/翻译的 unsafe 内容、jailbreak、自杀/自残；含 redaction |
| DuReader-robust | docs | Apache-2.0 | [HF card](https://huggingface.co/datasets/PaddlePaddle/dureader_robust)、[loader header](https://huggingface.co/datasets/PaddlePaddle/dureader_robust/blob/main/dureader_robust.py) | H 卡片 / M 上游 | train 14,520；dev 1,417；test 50,000（4.8K real + fake） | 1,100 | 简体 | 网页片段；正则扫描到 phone/ID/email/bankcard 命中；需脱敏 |
| China laws（全国现行法律法规） | docs | CC0-1.0 | [HF card](https://huggingface.co/datasets/senry5433/china-effective-laws-regulations) | H 卡片 / M 上游 | 17,701 docs / 703,272 articles；snapshot 2026-08-26 | 600 | 简体 | 官方法律文本；预期无 PII；长文档，需 8k 截断 |
| CMATH | hard | CC-BY-4.0 | [HF card](https://huggingface.co/datasets/weitianwen/cmath)、[上游 MIT](https://github.com/XiaoMi/cmath/blob/main/MIT-LICENSE) | H | 600 dev + 1,098 test = 1,698 | 400 | 简体 | 小学应用题；虚构人名；无 PII |
| HelpSteer3 zh | hard / score | CC-BY-4.0 | [HF card](https://huggingface.co/datasets/nvidia/HelpSteer3) | H 许可 / M PII | 2,306 条中文 preference train（总 38,459） | 300 | 简体 | 人类 prompt/response；需 PII 扫描 |

**说明：** 配额是转换后的记录数，不是原始行数。一个对话/文档/题目可以产生多条记录；同一 state 的兄弟问题共享 `group_id`。公开源上限按源合并计算；MASSIVE 的 1,200 正好是 public 6k 的 20%，不再加量。

### 3.2 逐源判断

#### MASSIVE（`AmazonScience/massive`）

- **许可：** HF 仓库的 `LICENSE` 是 CC-BY-4.0；GitHub `alexa/massive` 的 `LICENSE.txt` 对**代码**写 Apache-2.0，并指向 `NOTICE.md` 说明数据条款。数据侧以 HF `LICENSE` 为准，H。
- **规模：** 论文给出 51 种语言、每种 19,521 条；全局 587K train / 104K dev / 152K test / 153K MMNLU held-out。按语言折算约 11.5K train / 2.0K dev / 3.0K test，公开 16,521 条，另有 3,000 条 held-out。`zh-CN` 与 `zh-TW` 各一套。
- **场景：** 18 个 domain、60 个 intent、55 个 slot；适合把 `choice` 构造成 intent 选择、把 `noul` 构造成 slot/约束是否出现。状态短，属于基础决策。
- **语言：** `zh-CN` 简体、`zh-TW` 繁体；这是本配比里唯一稳定提供繁体训练量的源。
- **PII：** 语句是专业翻译者本地化的助理命令；数据含 `worker_id` 和 `judgments`，入料时丢弃，避免把标注者 ID 带进语料。
- **角色：** train 用 train split；dev/test 留作 held-out。不要用 test 调参。

#### CrossWOZ（`thu-coai/CrossWOZ`）

- **许可：** 仓库 `LICENSE` 为 Apache-2.0，H。
- **规模：** 6,012 个对话（train 5,012 / val 500 / test 500），102K utterances，5 个 domain（hotel、restaurant、attraction、metro、taxi）；约 60% 对话含跨域目标。
- **场景：** 任务型对话的 next-action / next-entity / constraint-satisfaction 决策；注释含 dialogue state 与 dialogue act，适合构造 `choice`（下一个系统动作/候选实体）与 `noul`（约束是否满足）。
- **语言：** 简体。
- **PII：** 模拟对话，酒店/餐厅名做匿名化；标注者 ID 不应入语料；无真实用户 PII。
- **角色：** train 用 train；val/test held-out。注意同一对话的 turns 必须同组切分。

#### KdConv（`thu-coai/KdConv`）

- **许可：** 仓库 `LICENSE` 为 Apache-2.0，H。
- **规模：** 4,500 个对话 / 85,596 utterances；train 3,600 / 66,538，dev 450 / 9,321，test 450 / 9,737；film、music、travel 三个 domain。
- **场景：** 知识驱动的多轮对话；每条 utterance 带知识图谱 triple 注释，适合构造 `choice`（下一话题/实体/知识三元组）、`noul`（某 triple 是否相关/被引用）和 `score`（回应与知识的 groundedness）。
- **语言：** 简体。
- **PII：** 公开影视/音乐/旅行实体；无真实用户 PII。
- **角色：** train 用 train；dev/test held-out。

#### COLD（`thu-coai/COLDataset`）

- **许可：** 仓库 `LICENSE` 为 Apache-2.0，H。
- **规模：** 37,480 条评论：train 25,726 / dev 6,431 / test 5,323；二分类 safe/offensive，test 另有四类细粒度标签。
- **场景：** 中文冒犯/仇恨语言检测；适合 `choice`（safe vs offensive，或细粒度类别）与 `noul`（是否攻击个人/群体、是否 anti-bias）。
- **语言：** 简体。
- **敏感内容：** 含 race、gender、region 的冒犯/仇恨表达；必须带内容警告、限制展示，并在训练前去重/脱敏。不要把它当作“安全回复”语料；它只提供分类标签。
- **PII：** 社交媒体评论，可能有用户名/个人信息；入料前做 PII 扫描。
- **角色：** train 用 train；dev/test held-out。test 的细粒度标签不能回流到 train。

#### Nemotron Safety Guard zh（`nvidia/Nemotron-Safety-Guard-Dataset-v3`）

- **许可：** HF 卡片明确 CC-BY-4.0，H；数据是 CultureGuard 管线从 Aegis 2.0 做文化适配/翻译并合成 jailbreak 样本，内容链是 M。
- **规模：** 12 种语言共 514,617 条；中文 `zh/train.jsonl` 37,335、valid 2,291、test 2,950，共 42,576。
- **场景：** prompt/response 的 safe/unsafe、violated categories、jailbreak；适合 `choice`（风险类别）与 `noul`（是否违反某类别）。它是 guard-model 训练集，不是通用决策集。
- **语言：** 中文（简体为主，待核）。
- **敏感内容：** 含自杀/自残、暴力、仇恨等 unsafe 内容，部分 prompt 被标为 `REDACTED`；必须做内容审计与 PII 扫描。
- **角色：** train 用 train；valid/test held-out。与 COLD、SafetyBench 做模糊去重，避免同一条冒犯文本跨 split。

#### DuReader-robust（`PaddlePaddle/dureader_robust`）

- **许可：** HF 卡片 `license: apache-2.0`；loader 头部也是 Apache-2.0。卡片很薄，没有上游数据许可的逐条说明，因此 **H 卡片 / M 上游**。归档里的 `License.pdf` 在本环境无法抽出文本，需二次核验。
- **规模：** train 14,520 questions / 14,520 answers；dev 1,417 questions / 1,962 answers；test 50,000（README 说约 4.8K 真实 + 其余 fake）。本记录实际解析确认 train/dev/test 的 question 数。
- **场景：** 中文抽取式阅读理解；状态是网页段落，问题是自然语言问句。适合 `choice`（候选答案 span）与 `noul`（段落是否可回答）。**上下文平均约 147 字符，属于短文档 grounding，不是 #11 的 2k–8k 长文档阶段。**
- **语言：** 简体。
- **PII：** 对 train 的 context+question 做正则扫描，命中 phone-like 82、ID-like 18、email-like 29、bankcard-like 15；这不是 ground truth，但足以要求脱敏。网页片段还可能含姓名/地名。
- **角色：** train 用 train；dev/test held-out。test 的 4.8K real 部分可作 eval，fake 部分不要当真实分布。

#### China laws（`senry5433/china-effective-laws-regulations`）

- **许可：** HF 卡片 `license: cc0-1.0`，H 卡片；数据来自全国人大常委会办公厅「国家法律法规数据库」`flk.npc.gov.cn`，快照 2026-08-26。上传者的 CC0 声明是否覆盖官方文本的再利用，需要 M 级上游核验；中国著作权法对法律/法规/官方文件有排除，但发布前仍应让法律侧确认。
- **规模：** 17,701 部法规 / 703,272 条法条；法律 725、行政法规 612、监察法规 2、地方法规 15,801、司法解释 561；文档级和法条级两种粒度。
- **场景：** 真实长文档、法条检索、适用性判断、定义/罚则问答；适合 `choice`（适用法条/定义）、`noul`（某法条是否适用）和 `score`（相关度/严重性）。是补 #11「真实文档」阶段长 state 的主力。
- **语言：** 简体。
- **PII：** 法规文本，预期无 PII；不含案件当事人信息。
- **角色：** train 按 `doc_id` 切分；留 10% 文档做 held-out。**与 LawBench 的 1-1 法条背诵任务同源（FLK）**，必须排除 LawBench 用到的法条，或至少出 overlap report。

#### CMATH（`weitianwen/cmath`，上游 `XiaoMi/cmath`）

- **许可：** HF 卡片 CC-BY-4.0；上游 GitHub `MIT-LICENSE` 为 MIT。两者都白名单；以 CC-BY-4.0 记录，H。
- **规模：** 600 dev + 1,098 test = 1,698；小学 1–6 年级应用题，带 `golden`、`reasoning_step`、`num_digits`。
- **场景：** 可程序化验证的数学硬技能；适合 `choice`（数值答案）与 `noul`（是否需要多步/某运算）。不是开放式生成数据。
- **语言：** 简体。
- **PII：** 虚构人名（小明、小红等）；无 PII。
- **角色：** train 用 dev（600）或自行按题目 group 切分；test（1,098）held-out。不要把 test 的题面/答案放进合成 prompt。

#### HelpSteer3 zh（`nvidia/HelpSteer3`）

- **许可：** HF 卡片 CC-BY-4.0，H；语言列表含 `zh`。
- **规模：** preference train 38,459 条，其中 `language == "chinese"` 2,306 条；另有 feedback/edit/principle 配置。中文量足以抽 300 条。
- **场景：** 人类 helpfulness preference；适合 `score`（4 级 helpfulness）和 `choice`（偏好 response）。它是本配比里 public 侧唯一的 score 主力。
- **语言：** 简体（待核；卡片只标 `zh`）。
- **PII：** 人类写的 prompt/response，可能有个人信息；入料前扫描/脱敏。
- **角色：** train 用 train；validation held-out。不要用 validation 拟合 temperature。

---

## 4. 只能做 held-out / 评估的源

这些源许可 permissive，但它们是基准/评测集，**不进入训练**；用于 #12/#18 的两层评估与去污染报告。

| 源 | 评估维度 | 许可 | 证据 | 可用题量 | 语言 | 备注 |
|---|---|---|---|---:|---|---|
| TMMLU+ | 学科知识（66 科） | MIT | [HF card](https://huggingface.co/datasets/ikala/tmmluplus) | 22,203 | 繁体（zh-TW） | 台湾语境；v1.1；适合做 zh-TW 主 eval |
| XWinograd zh | 指代/逻辑 | CC-BY-4.0 | [HF card](https://huggingface.co/datasets/Muennighoff/xwinograd) | 504 | 中文 | 小但干净；适合 held-out slice |
| SafetyBench zh | 安全 | MIT | [GitHub LICENSE](https://github.com/thu-coai/SafetyBench/blob/main/LICENSE)、[HF card](https://huggingface.co/datasets/thu-coai/SafetyBench) | 11,435 | 简体 | 与 COLD/Nemotron 做模糊去重 |
| CMB | 医学 | Apache-2.0 | [HF card](https://huggingface.co/datasets/FreedomIntelligence/CMB) | 11,200 test | 简体 | 医学考试；不要训练 |
| LawBench | 法律 | Apache-2.0 | [GitHub LICENSE](https://github.com/open-compass/LawBench/blob/main/LICENSE) | 10,000（20 任务 × 500） | 简体 | 1-1 与 China laws 同源，必须去重 |
| C-VARC | 中文价值观 | CC-BY-4.0 | [HF card](https://huggingface.co/datasets/Beijing-AISI/C-VARC) | 257,609 rules | 简体 | 规则语料；适合价值/规范 eval，不直接当决策训练 |
| CodeFuse-DevOps-Eval | DevOps 知识 | MIT | [HF card](https://huggingface.co/datasets/codefuse-ai/CodeFuse-DevOps-Eval) | 5,977 | 中英 | 中文 DevOps 选择题 |
| CodeArena | 代码偏好 | Apache-2.0 | [HF card](https://huggingface.co/datasets/CSJianYang/CodeArena) | 397 | 中英 | 人偏好代码题；只做 eval |

**使用规则：**

- 这些源整源隔离，不进入 public/synthetic/programmatic 任何训练配额。
- 评估数据的发布按 #12/#15：permissive 源可发 item-level；NC/SA/unknown 源只能发 manifest + 上游指针 + hash，标「非归档」。
- LawBench / China laws、SafetyBench / COLD / Nemotron、CMB / 任何医学合成数据都要做 exact + fuzzy dedupe，并发布 overlap report。

---

## 5. 排除的源与理由

### 5.1 许可不满足

| 源 | 许可/事实 | 证据 | 判定 |
|---|---|---|---|
| CLUE 系列（TNEWS、IFLYTEK、OCNLI、CMNLI、AFQMC、CSL、C3、CHID 等） | 卡片 `license: unknown` | [clue/clue card](https://huggingface.co/datasets/clue/clue) | X：unknown |
| CMRC2018 | `cc-by-sa-4.0` | [hfl/cmrc2018 card](https://huggingface.co/datasets/hfl/cmrc2018) | X：SA |
| C-Eval | `cc-by-nc-sa-4.0` | [ceval/ceval-exam card](https://huggingface.co/datasets/ceval/ceval-exam) | X：NC + SA |
| CMMLU | `cc-by-nc-4.0` | [lmlmcat/cmmlu card](https://huggingface.co/datasets/lmlmcat/cmmlu) | X：NC |
| BAAI/COIG | 卡片 Apache-2.0，但自述含 web-crawled data under fair use，且 Leetcode 子集是 CC-BY-SA-4.0 | [HF card](https://huggingface.co/datasets/BAAI/COIG) | X：整源不可用；`translated_instructions` 若要复用，必须逐个上游（Super-NaturalInstructions Apache-2.0、Self-Instruct Apache-2.0、Unnatural Instructions MIT）重新核验，不能按整源 Apache 处理 |
| BelleGroup `train_0.5M_CN` / `train_3.5M_CN` | GPL-3.0 | [HF card](https://huggingface.co/datasets/BelleGroup/train_0.5M_CN) | X：不在白名单 |
| `shibing624/alpaca-zh` | 卡片 CC-BY-4.0，但上游 `tatsu-lab/alpaca` 是 CC-BY-NC-4.0 | [alpaca-zh card](https://huggingface.co/datasets/shibing624/alpaca-zh)、[upstream card](https://huggingface.co/datasets/tatsu-lab/alpaca) | X：许可洗白风险；不能用 |
| `m-a-p/COIG-CQIA` | 卡片无 license | [HF card](https://huggingface.co/datasets/m-a-p/COIG-CQIA) | X：unknown |
| ToxiCN | 仓库无 license | [DUT-lujunyu/ToxiCN](https://github.com/DUT-lujunyu/ToxiCN) | X：无声明 |
| CAIL2018 | 仓库无 license | [thunlp/CAIL2018](https://github.com/thunlp/CAIL2018) | X：无声明 |
| JEC-QA | 仓库无 license | [thunlp/jec-qa](https://github.com/thunlp/jec-qa) | X：无声明 |
| DRCD | 仓库无 license | [DRCKnowledgeTeam/DRCD](https://github.com/DRCKnowledgeTeam/DRCD) | X：无声明 |
| LogiQA | 仓库无 license | [lgw863/LogiQA-dataset](https://github.com/lgw863/LogiQA-dataset) | X：无声明 |
| Ape210K | 仓库无 license | [Chenny0808/ape210k](https://github.com/Chenny0808/ape210k) | X：无声明 |
| UltraData-Code | 卡片 Apache-2.0，但明确要求遵守每个源仓库许可，并禁止未经许可的原样再分发 | [HF card](https://huggingface.co/datasets/openbmb/UltraData-Code)「License and Data Sources」 | X：不是 permissive-only；除非按 repo 逐条过滤，否则不能入料 |

### 5.2 许可可用但 PII/污染风险过高

| 源 | 事实 | 证据 | 判定 |
|---|---|---|---|
| LCCC / `silver/lccc` | MIT；10M–100M 对话；PII 小节是 `[Needs More Information]`；数据来自微博对话 | [HF card](https://huggingface.co/datasets/silver/lccc)、[CDial-GPT LICENSE](https://github.com/thu-coai/CDial-GPT/blob/master/LICENSE) | X（v1）：PII 不可评估；可作 v1 后 redaction 试点，不进 15k |
| THUIR/Qilin | Apache-2.0，但含用户行为/推荐日志（user_feat、session、click） | [HF card](https://huggingface.co/datasets/THUIR/Qilin) | X：用户数据/PII，且不是决策问答 |
| `BAAI/IndustryCorpus*`、`opencsg/Fineweb-Edu-Chinese*`、`liwu/MNBVC`、`HPLT`、`OSCAR` | 卡片可能标 Apache/MIT/CC0，但文本来自 web crawl；上游 terms 不是 permissive | [IndustryCorpus](https://huggingface.co/datasets/BAAI/IndustryCorpus)、[Fineweb-Edu-Chinese](https://huggingface.co/datasets/opencsg/Fineweb-Edu-Chinese-V2.1)、[MNBVC](https://huggingface.co/datasets/liwu/MNBVC)、[HPLT](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned)、[OSCAR](https://huggingface.co/datasets/oscar-corpus/OSCAR-2201) | X：web-crawl provenance 不满足 #11；除非逐 URL 过滤 |
| CommitPackFT | MIT 卡片 + 逐行 license；但 CJK 命中极低：抽样 Python/JS/Java/Go/C++/TypeScript 共 145,513 条，仅 23 条含 ≥2 个 CJK 字符，且多数是日文汉字，实际中文更少 | [HF card](https://huggingface.co/datasets/bigcode/commitpackft)、本记录抽样 | X：不能支撑 600 条 code 配额 |
| ToolACE | Apache-2.0，但 11,300 条中按 CJK 启发式仅约 20 条中文；数据是合成 tool-use | [HF card](https://huggingface.co/datasets/Team-ACE/ToolACE)、本记录抽样 | X：中文量不足；可作英文/多语 agent 参考，不进 zh 15k |

> Wikipedia 派生语料（如 MIRACL、MLQA 等含中文 passage 的检索/QA 集）统一按 Wikipedia 的 CC-BY-SA 处理，未逐一列入；本票不把它们当候选。

### 5.3 评估/污染性排除

- C-Eval、CMMLU、CLUE、CMRC2018 既不能训练，也不能作为可再分发的 eval 数据；#18 需要另找 permissive 中文 eval（见第 4 节）。
- MMLU、SWE-bench、HumanEvalPack、ARC、CommonsenseQA、WinoGrande、BoolQ、SST-2、HellaSwag、PAWS 等英文套件按 #11 整源排除；中文扩展不复用，保持与 v1 的 head-to-head 干净。
- JevBench / Decision Index 只作对外比较，永不入训。

### 5.4 许可洗白的教训

本票最重要的许可发现不是“哪个源能用”，而是**卡片上的 permissive 标签不等于上游可再分发**：

1. `alpaca-zh` 卡片 CC-BY-4.0，上游 Alpaca 是 CC-BY-NC-4.0。
2. `COIG` 卡片 Apache-2.0，但自述 web-crawled fair use + Leetcode CC-BY-SA。
3. `UltraData-Code` 卡片 Apache-2.0，但要求遵守每个源仓库许可，并限制原样再分发。
4. `CommitPackFT` 卡片 MIT，但中文 commit message 极少；许可是真的，规模是假的。
5. `Nemotron-SFT-Multilingual-v1` 卡片是 `cc-by-4.0 + cc-by-sa-4.0` 混合，不能按纯 CC-BY 收（[HF card](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Multilingual-v1)）。

入料前的检查顺序应是：**上游 LICENSE/条款 → 数据集卡 → 逐条/逐 repo 过滤 → 记录级 provenance**，不能反过来只看卡片。

---

## 6. 15k 配比建议

### 6.1 阶段 × 来源类型

| 阶段 | Public | Synthetic | Programmatic | 合计 |
|---|---:|---:|---:|---:|
| Basic decisions | 3,600 | 2,400 | 1,200 | 7,200 |
| Real documents | 1,700 | 900 | 0 | 2,600 |
| Hard skills | 700 | 1,500 | 800 | 3,000 |
| Devtools / code | 0 | 1,200 | 1,000 | 2,200 |
| **合计** | **6,000** | **6,000** | **3,000** | **15,000** |

阶段占比：basic 48.0% / docs 17.3% / hard 20.0% / code 14.7%，与 v1 的四阶段骨架接近；docs 略高是因为中文公开文档源（DuReader、China laws）可用，而 code 公开源缺失。

### 6.2 Public 6k 组合

| 源 | 阶段 | 配额 | 占 public | 说明 |
|---|---|---:|---:|---|
| MASSIVE zh-CN + zh-TW | basic | 1,200 | 20.0% | 800 简 + 400 繁；压在上限 |
| CrossWOZ | basic | 800 | 13.3% | 任务型对话动作/实体 |
| KdConv | basic | 500 | 8.3% | 知识驱动对话 |
| COLD | basic / safety | 600 | 10.0% | safe/offensive + 细粒度 |
| Nemotron Safety zh | basic / safety | 500 | 8.3% | 风险类别/jailbreak |
| DuReader-robust | docs | 1,100 | 18.3% | 短文档 grounding |
| China laws | docs | 600 | 10.0% | 法律长文档；与 LawBench 去重 |
| CMATH | hard | 400 | 6.7% | 小学数学 |
| HelpSteer3 zh | hard / score | 300 | 5.0% | helpfulness score |
| **合计** | | **6,000** | **100%** | |

**为什么 code 是 0：** 见第 1 节与 5.2；没有找到达到数百条、逐条 permissive 的中文代码决策源。把 600 条 code 配额转给 basic/docs/hard 比硬塞一个 M 级/许可洗白的 code 源更符合 #11。

### 6.3 Synthetic 6k 缺口

| 切片 | 记录 | 内容 | 为什么必须合成 |
|---|---:|---|---|
| Basic decisions | 2,400 | 中文 state 上的 intent/action/policy 多问题记录；其中 600 条 zh-TW | 公开源只覆盖特定 domain；choice/noul 的开放状态与选项需要 teacher 造 |
| Real documents | 900 | 以 DuReader / China laws 为 state 的 QA、grounding、长文档（≤8k tokens） | 公开 QA 的题型/答案分布窄；需要覆盖 noul/score 与长文档 |
| Hard skills | 1,500 | tone 400、injection 300、PII 300、grounding 300、math/logic 200 | 中文 permissive 标注源几乎没有 tone/injection/PII；CMATH 只有 1,698 |
| Devtools/code | 1,200 | 以 permissive code 为 state 的 code review / commit / decision 合成 | 没有可用的中文 permissive code 决策源 |
| **合计** | **6,000** | | |

**合成规则（沿用 #11）：**

- 只用 Qwen3.5-35B-A3B / Qwen3.8-27B 两个 open-weight teacher；版本、量化、seed 钉死。
- 开放式决策双 teacher 独立生成，标签一致才收；分歧丢弃；逐 family 统计一致率。
- 可程序化验证的标签优先；teacher 只做自然语言状态/问题/解释，不让 teacher 决定可验证标签。
- 每 family 人工抽检 1–2%；pilot 额外 100 条，标签正确率 ≥95%。
- 不发布 teacher CoT；只发最终标签、prompt、参数、过滤规则。

### 6.4 Programmatic 3k

| 切片 | 记录 | 内容 |
|---|---:|---|
| Rule/policy hard cases | 1,200 | 规则树、政策对、约束满足/违反、优先级/严重性 score |
| Math/logic verifiable | 500 | 可程序化验证的算术/逻辑/单位换算 |
| Structured/verifiable tasks | 300 | 表格/字段/结构化输出校验 |
| Verifiable code tasks | 1,000 | 自建可执行测试的代码决策；不复用 eval 题 |
| **合计** | **3,000** | |

### 6.5 题型配比

目标沿用 #11：`choice` 45% / `noul` 35% / `score` 20%，pilot 容差 ±10 pp。

| 题型 | Public | Synthetic | Programmatic | 合计 | 占比 |
|---|---:|---:|---:|---:|---:|
| choice | 3,600 | 2,400 | 750 | 6,750 | 45% |
| noul | 2,100 | 2,100 | 1,050 | 5,250 | 35% |
| score | 300 | 1,500 | 1,200 | 3,000 | 20% |
| **合计** | **6,000** | **6,000** | **3,000** | **15,000** | **100%** |

**score 是公开侧最薄的一环：** public 只有 HelpSteer3 的 300 条；其余 2,700 条 score 必须由 synthetic/programmatic 补。pilot 要专门检查 score 的校准与标签一致性。

### 6.6 语言变体

- 简体：MASSIVE zh-CN、CrossWOZ、KdConv、COLD、Nemotron、DuReader、China laws、CMATH、HelpSteer3。
- 繁体：MASSIVE zh-TW（public 400）；synthetic basic 600 + programmatic basic 500，合计 zh-TW 约 1,500（10%）。
- 评估侧繁体：TMMLU+ 22,203、XWinograd zh 504；这是 zh-TW 的主要测量面。
- 不在 v1 引入粤语/其他变体；数据不足，且会稀释评估口径。

---

## 7. 2k pilot 建议

### 7.1 Pilot 配比

| 阶段 | Public | Synthetic | Programmatic | 合计 |
|---|---:|---:|---:|---:|
| Basic decisions | 500 | 350 | 150 | 1,000 |
| Real documents | 250 | 150 | 0 | 400 |
| Hard skills | 50 | 250 | 100 | 400 |
| Devtools / code | 0 | 50 | 150 | 200 |
| **合计** | **800** | **800** | **400** | **2,000** |

- 题型：choice 900 / noul 700 / score 400（45/35/20）。
- zh-TW：至少 300 条（15%）；示例分配 MASSIVE zh-TW 100、synthetic 繁体 120、programmatic 繁体 80。
- 公开源抽样：MASSIVE 200、CrossWOZ 150、KdConv 50、COLD 50、Nemotron 50、DuReader 150、China laws 100、CMATH 30、HelpSteer3 20。
- 每个 family 至少 50 条，便于按 family 统计双 teacher 一致率；不足 50 的 family 合并报告。

### 7.2 七项闸（沿用 #11，不放宽）

1. 开放式标签双 teacher 一致率 ≥80%（按 family 统计；分歧丢弃）。
2. 100 条人工抽检标签正确率 ≥95%，题目无系统性歧义。
3. 许可台账完整：每个入料源都有上游证据；M 级源全部补证；无 `unknown`/`other`。
4. 去污染报告干净：无 eval 源重叠，group-aware 切分无泄漏。
5. 题型分布落在目标 ±10 pp 内（choice 45 / noul 35 / score 20）。
6. 成本在估算内：pilot 云侧 ≤$50，折算 ≤$15/1k 条（混合 teacher）。
7. provenance schema 完整，语料可加载、hash 可校验。

过闸 → 放量到 15k；不过 → 修数据/修流程后重跑 pilot，不放量。

---

## 8. M 级二次核验清单

| 源 | 待核验 | 为什么是 M | 不过时的替换 |
|---|---|---|---|
| DuReader-robust | 归档 `License.pdf` / 上游数据条款 | 卡片只有一行 `apache-2.0`，没有上游数据许可说明 | 用 China laws + synthetic docs 补；或换其他 Apache 中文 QA |
| China laws | `flk.npc.gov.cn` 的再利用条款；CC0 声明是否覆盖官方文本 | 上传者 CC0 声明不等于上游条款 | 只保留法律/行政法规（官方文件著作权排除更明确），地方法规降为 eval-only |
| HelpSteer3 zh | PII 扫描、中文是否为人工原生 | 人类 prompt/response；语言标签只写 `zh` | 用 synthetic score 补；或换 permissive 中文偏好数据 |
| Nemotron Safety zh | 内容审计、翻译链、PII | 合成/翻译的 unsafe 内容，含 redaction | 用 COLD + synthetic safety 补 |
| MASSIVE | 公开 split 精确值、zh-TW 是否全繁体 | 本环境 S3 下载受限，计数来自论文与卡片 | 用 CrossWOZ/KdConv + synthetic 补；不影响许可 |
| China laws vs LawBench | 1-1 法条重叠清单 | 同源 FLK | 排除重叠法条，或把 LawBench 1-1 降为 overlap-report only |

---

## 9. 风险与改主意条件

1. **DuReader 上游许可补证失败** → 去掉 1,100 条 public docs，用 China laws（若可用）+ synthetic docs 补；public 6k 重新分配。
2. **China laws CC0 被法律侧否决** → 只保留官方法律/行政法规的短法条切片，或全部转 synthetic；docs 阶段降到 1,700 左右。
3. **HelpSteer3 PII 过高** → 去掉 300 条 score，public 6k 用 MASSIVE/CrossWOZ 补 basic；score 全部由 synthetic/programmatic 补。
4. **Nemotron 内容审计不过** → 去掉 500 条 safety，COLD 提到 800（仍 <1,200 上限），synthetic safety 加 300。
5. **MASSIVE 许可/数据漂移** → 钉 HF revision；若 zh-TW 不可用，繁体训练量转 synthetic，评估仍用 TMMLU+/XWinograd。
6. **code 缺口进一步恶化** → 把 code synthetic 从 1,200 提到 1,500、programmatic 从 1,000 提到 1,200；从 basic 的 programmatic 减 500。不要为了填配额引入许可洗白源。
7. **LawBench/China laws 污染** → 排除重叠法条；若无法干净切分，LawBench 1-1 只报 overlap，不计入主分。
8. **SafetyBench/COLD/Nemotron 重叠** → 模糊去重后按 group 切分；若重叠 >10%，SafetyBench 只报 overlap。
9. **zh-TW 训练量不足** → MASSIVE zh-TW 从 400 提到 600（MASSIVE 总量会超 1,200 上限，需从 zh-CN 挪），synthetic 繁体从 600 提到 800。
10. **公开源 20% 上限** → MASSIVE 已在 cap；任何加量都要从其他 public 源挪，不能破 cap。

---

## 10. 未决问题与交给下游

- **#18 中文公开评估套件：** C-Eval/CMMLU/CLUE/CMRC 因 NC/SA/unknown 不能作为可再分发 eval；可用的 permissive 替代是 TMMLU+、XWinograd、SafetyBench、CMB、LawBench、C-VARC、CodeFuse-DevOps-Eval、CodeArena。需要 #18 决定哪些进 `eval-zh-v1.0`、哪些只做内部诊断。
- **代码源：** 若要补 public code，下一步应扫 The Stack v2 的 permissive 子集（逐 repo license），只取中文注释/commit 的代码 state；这是程序化过滤，不是“现成中文 code 数据集”。在完成逐 repo 过滤前，不要声称有 public 中文 code 训练源。
- **PII 流水线：** DuReader、HelpSteer3、COLD、Nemotron 都需要统一 redaction（电话/身份证/邮箱/银行卡/用户名）与人工抽检；redaction 后要重跑标签一致性。
- **法律侧确认：** China laws 的 CC0 声明、DuReader 的 `License.pdf`、以及 LawBench 同源重叠，建议在 2k pilot 前完成。
- **Teacher 版本：** Qwen3.5-35B-A3B / Qwen3.8-27B 的具体 revision、量化、seed 由 #19（中文语料与 eval-zh 配方）钉死；本记录只沿用 #11 的名字。
- **数据卡：** 发布时按 #15 的 `license: other` + 机器可读台账；不要把 Apache-2.0/CC0 记录错误重许可为 CC-BY-4.0。

---

## 附录 A：源 revision 与 canonical URL

| 源 | Revision | URL |
|---|---|---|
| MASSIVE | HF `ff6bd8e4b27c3543e4f8fe2108f32bb95a6f8740` | https://huggingface.co/datasets/AmazonScience/massive |
| DuReader-robust | HF `10003b9e4d5e0afaff73fcf366f02d49880ec17a` | https://huggingface.co/datasets/PaddlePaddle/dureader_robust |
| COLD | GitHub `a0c55a945497ab58e94aef491180e1cf88ffb864` | https://github.com/thu-coai/COLDataset |
| CrossWOZ | GitHub `df82c9fdff91b9b130f2d6b89110d3870ba6260e` | https://github.com/thu-coai/CrossWOZ |
| KdConv | GitHub `653db76432de09a004ba708a68f8bbd5500e6bec` | https://github.com/thu-coai/KdConv |
| China laws | HF `11be732f1a529a46e02e8c1530bae3b84f967bc3` | https://huggingface.co/datasets/senry5433/china-effective-laws-regulations |
| CMATH | HF `6a626b7266f7c84fd2d2421ad695ffb34b8f9d69` | https://huggingface.co/datasets/weitianwen/cmath |
| HelpSteer3 | HF `f6d145777bcbde96137596340fab89793acd1031` | https://huggingface.co/datasets/nvidia/HelpSteer3 |
| Nemotron Safety | HF `a3f7ecb3433d1933701a83f18de16c36934a7f51` | https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3 |
| SafetyBench | HF `70538631943dc1517ae9bb0733251809f4445d0e`；GitHub `960833692a1ec82b191ff3e902daca7993c35cc1` | https://github.com/thu-coai/SafetyBench |
| CMB | HF `935fbc09edf1303d89872b21265ff597f426ac0d` | https://huggingface.co/datasets/FreedomIntelligence/CMB |
| LawBench | GitHub `e30981bb3ff54c41571f222e0b23e92d27375388` | https://github.com/open-compass/LawBench |
| TMMLU+ | HF `639a64db464b4661feb91817b9edfe43932bdf18` | https://huggingface.co/datasets/ikala/tmmluplus |
| XWinograd | HF `90b619ef5278605cd4f572e3e8506ab16afd44f2` | https://huggingface.co/datasets/Muennighoff/xwinograd |
| C-VARC | HF `8db54ccfdc51474d13643dff3e2bc21cf22989f7` | https://huggingface.co/datasets/Beijing-AISI/C-VARC |
| CodeFuse-DevOps-Eval | HF `380148ed13168484698658a0f916d10f9d200136` | https://huggingface.co/datasets/codefuse-ai/CodeFuse-DevOps-Eval |
| CodeArena | HF `05a5f4cae626aab85514451b7d51141b1bef3189` | https://huggingface.co/datasets/CSJianYang/CodeArena |
| CLUE | HF `28178267a609dd08bdc703dd6c931dfc2c2f4431` | https://huggingface.co/datasets/clue/clue |
| C-Eval | HF `617524a00b307ff6f9933702f724131fe12ca7ce` | https://huggingface.co/datasets/ceval/ceval-exam |
| CMMLU | HF `efcc940752ea4a1ea94d2727f11f83858d64fc8e` | https://huggingface.co/datasets/lmlmcat/cmmlu |
| COIG | HF `9f25758ec94f82762fb9c09a5c60e908cfb83632` | https://huggingface.co/datasets/BAAI/COIG |
| Belle 0.5M | HF `29c35e9b56c7fc91f04a613ed33896f7a19bad54` | https://huggingface.co/datasets/BelleGroup/train_0.5M_CN |
| LCCC | HF `5bd582fa28cd7143f2f9c852e08e23089d677c44` | https://huggingface.co/datasets/silver/lccc |
| alpaca-zh | HF `f39db019a94f8dbea48ab30d2bdc090703284559` | https://huggingface.co/datasets/shibing624/alpaca-zh |
| COIG-CQIA | HF `8b55868c6168adf86c30e7ca0f782cca1c514297` | https://huggingface.co/datasets/m-a-p/COIG-CQIA |
| Qilin | HF `2d79000cef450b99f6423357ce86e5c132d32dd5` | https://huggingface.co/datasets/THUIR/Qilin |
| UltraData-Code | HF `b20cabcf35ba2e6e75c3612916491167a33ce726` | https://huggingface.co/datasets/openbmb/UltraData-Code |
| CommitPackFT | HF `fc56fe33c030c6daa414c2b112c932b8eed085e6` | https://huggingface.co/datasets/bigcode/commitpackft |
| ToolACE | HF `6bda777c88d21e5a204703c1ee45597a8fa4f734` | https://huggingface.co/datasets/Team-ACE/ToolACE |

## 附录 B：计数与读取说明

- MASSIVE 的 split 数字来自论文 `https://arxiv.org/abs/2204.08582` 的正文与 Table 1（19,521/locale；587K/104K/152K/153K）；HF 数据归档在本环境因 S3 带宽限制只下到 ~29/39.5 MB，因此没有用本地 tar 复核 split。**这是 MASSIVE 唯一的 M 级点，但不是许可问题。**
- DuReader-robust、COLD、CrossWOZ、KdConv、CMATH、SafetyBench、CMB、C-VARC、HelpSteer3、Nemotron 的计数是下载/解析实际文件后得到的；KdConv 的 split 是按 `data/<domain>/<split>.json` 逐个 `len()` 求和。
- CommitPackFT 的“中文”用 `[\u4e00-\u9fff]` 至少 2 个字符做启发式；这个正则会把日文汉字算进来，所以 23 条是**上界**，实际中文更少。结论“不能支撑 600 条”不受影响。
- ToolACE 的“中文”用整条 JSON 的 CJK 启发式（≥10 字符），得到约 20/11,300；没有语言字段，所以这是近似值。
- 本记录没有运行任何训练、生成或 `data-zh-v1.0` 写入；所有配额都是建议。
