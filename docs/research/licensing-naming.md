# 研究:许可链、代码复用与命名空间(票 #8)

- **票号**:[#8](https://github.com/0xnicholas/Stim/issues/8) · 供决策票 [#15 开源与发布策略]、[#11 训练数据与生成策略] 使用
- **日期**:2026-10-16 · **分支**:`research/licensing-naming` · **状态**:调查完成,不做发布决策
- **方法**:一手来源直取(kev 源码浅克隆、`gh api`、`curl`、PyPI/HF/npm 官方 API 与站点文本)。所有结论附链接;不可达来源单列于 §5。
- **可复现命令**:见 §7。

## 0. 结论速览(事项 × 结论 × 依据 × 行动建议)

| # | 事项 | 结论 | 关键依据 | 行动建议 |
|---|---|---|---|---|
| L1 | 候选基座许可 | Qwen3.5-4B-Base / 0.8B / 9B、Qwen3.8-27B 均 Apache-2.0,可商用、可再分发、可衍生 | [Kev README 模型表](https://github.com/jaredpalmer/kev/blob/main/README.md)(标注 base license)、HF API `Qwen/Qwen3.5-4B-Base` → `cardData.license = apache-2.0` | 直接采用 Apache-2.0 基座即可,无额外义务(仍须随产物保留 Apache-2.0 许可文本) |
| L2 | Kev 训练集许可 | 混合:多数 CC-BY-4.0 / MIT / Apache-2.0,但含 **share-alike**(boolq CC-BY-SA-3.0、dbpedia_14 CC-BY-SA-3.0;Kev-27B 另加 ARC/HotpotQA/NQ/SNLI CC-BY-SA)与若干 `unknown` / `other` | HF API 逐个 dataset 许可(§2.2 表)、[kev-4b 模型卡](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-4b.md)、[kev-27b 模型卡](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md) | 复刻配方时**逐数据集建许可清单**,share-alike 与 `unknown` 源要么剔除、要么分开记录(Kev 自己的做法:不可公开的数据不公开) |
| L3 | Kev 代码/配方的复用边界 | Apache-2.0 允许商用、再分发、衍生;义务=附许可副本 + 标注修改过的文件 + 保留版权/署名声明;kev **无 NOTICE 文件**,故 §4(d) 未触发;§6 **不授予商标权** | [kev LICENSE](https://github.com/jaredpalmer/kev/blob/main/LICENSE)(Apache-2.0,`Copyright 2026 Jared Palmer`)、`pyproject.toml: license = "Apache-2.0"`、仓库根无 `NOTICE` | 若复制其代码/文档:保留 LICENSE、在改动文件写变更说明、README 署名;**不要**用 "Kev" 命名自家产物,也不要暗示来自 Jared Palmer 的背书 |
| L4 | 引用「配方/思路」 | 架构、训练配方、评测方法属事实/方法层面,重新实现不触发 Apache-2.0 义务(仅为礼节性署名) | Apache-2.0 §1 “Derivative Works” 定义与 §2 授权范围;Kev 自身对 Archer Hume 架构文与 TypeSafe API 设计只做致谢 | 在 README 写「受 Kev / Jev 架构文启发」+ 明确「非官方、无关联」 |
| C1 | 兼容实现的生态先例 | 生态先例充分且公开:**官方** MIT 适配器明确可指向任意后端,多个第三方 System One 兼容实现公开存活并被社区清单收录 | [typesafe-ai/system-one-adapter-python](https://github.com/typesafe-ai/system-one-adapter-python)(MIT,393★,“Drop-in TypeSafeClient replacement backed by LLM APIs”)、[razorback16/openjev](https://github.com/razorback16/openjev)(Apache-2.0,713★)、[Liuziyu77/Valen](https://github.com/Liuziyu77/Valen)(Apache-2.0,831★)、[yibie/awesome-jev](https://github.com/yibie/awesome-jev)(2367★) | 公开发布 System One 兼容实现有先例支撑;照 Kev 做法显式声明「SDK 可直接指向本地服务」 |
| C2 | TypeSafe 条款的相反信号 | **有,但作用于其客户而非全体第三方**:MCA §2.3(b) 禁止用 Services 或 **Output** 做蒸馏、训练模仿模型、或开发相似/竞争产品;§2.3(c)(d)(g) 禁止逆向、衍生、绕过;AUP §3.1–3.2 禁止逆向与抓取 | [Master Customer Agreement](https://typesafe.ai/legal/mca)(2026-09-23)、[Acceptable Use Policy](https://typesafe.ai/legal/acceptable-use-policy) | 把「不训练于 Jev 输出」写成硬规则(Kev 已在模型卡声明 “No output of Jev … was used”);不用 TypeSafe 账号跑替代服务;不抓取、不绕过 |
| C3 | Output 归属 vs 使用 | §4.2 **不主张 Output 所有权并转让给客户**,但 §2.3(b) 仍**禁止其使用方式**;§4.1 未经同意不得纳入训练数据;隐私政策承诺不训练用户输入 | 同上(MCA §4.1/§4.2、[Privacy Policy](https://typesafe.ai/legal/privacy-policy)) | 「我拥有输出」≠「我可以拿去炼替代品」;在 #11 中按 §2.3(b) 设红线 |
| C4 | 官方对兼容的**授权性**文字 | **未发现**:没有公开品牌政策页,没有「允许第三方兼容实现」的明文许可;官方姿态是 MIT 工具 + 社区收录(容忍),不是法律授予 | docs.typesafe.ai 的 `/legal`(仅 DPA/MCA/Privacy)与 AUP;`typesafe.ai/legal` 索引页 404 | 兼容按「独立实现 + 自证」路径推进;不要引用官方文档当作许可依据 |
| N1 | PyPI `stim` | **已被占用**:quantumlib 的量子稳定子库 Stim(Craig Gidney,Apache-2.0,1.16.0,2026-05-22,127 个版本)→ `pip install stim` 装到的是 Google 生态的量子库 | `https://pypi.org/pypi/stim/json` = 200;[quantumlib/Stim](https://github.com/quantumlib/Stim)(834★) | **不要**计划 `pip install stim`;这是 Kev 撞名教训的同型复发(占用者更强势) |
| N2 | PyPI 备选 | 未占用(404):`stim-ai`、`stim-core`、`stim-sdk`、`stim-cli`;已占用(200):`pystim`、`stimkit` | PyPI JSON API(§3.1) | 若要发布,选带后缀的发行名;注意 PyPI 会拒绝「与既有项目过于相似、造成混淆」的名字 |
| N3 | npm `stim` | **已被占用且活跃**:v1.21.0,30 个版本,MIT,今天仍有更新(2026-10-16T01:06Z),“Stim: fast, isolated React Native environments for coding agents” → 与 agent/coding-agent 领域贴得很近 | `https://registry.npmjs.org/stim` = 200、`/stim/latest`(§3.2) | npm 同名冲突风险高于 PyPI 量子库(受众重叠);发布 npm 包需换名并加「not affiliated」声明 |
| N4 | GitHub 命名空间 | `github.com/stim` 不可得:org `STIM` 已存在(2015-07-01 创建,2 个无关仓库);`github.com/stim/stim` 无仓库;`appandflow/stim` 已占。自有的 `0xnicholas/Stim` 已是既定事实且不冲突(仓库名按 owner 隔离) | `gh api orgs/STIM`、`gh api repos/stim/stim` = 404、`gh api repos/0xnicholas/Stim` | 项目主页只能用 `0xnicholas/Stim`;不要假设能拿到 `stim` 这个 GitHub 短名 |
| N5 | Hugging Face 命名空间 | 用户名 `stim` **已被注册**(页面标题 `stim (Max Stieber)`),该账号 0 个公开模型、0 个公开数据集 → 名字被占但近乎闲置;`stim-ai` / `stim-llm` / `getstim` 均未被占;`0xnicholas` 尚未注册 | hf-mirror 页面与 API(§3.3) | **`stim/stim-4b` 不可行**;须在自有 namespace 下发布(如 `0xnicholas/stim-4b`),发布前先注册该用户名 |
| N6 | 权重仓库命名方案 | `stim-4b` / `stim-9b` 作为**仓库名**干净(HF 全站模型搜索 `stim-9b` 0 命中;`stim-4b` 仅 1 条无关 stimuli 结果),前提是挂在自有 namespace 下;Kev 用 `jaredpalmer/kev-4b` 这种「ns + 家族前缀」结构 | hf-mirror 搜索 API(§3.4)、`jaredpalmer/kev-4b` | 采用 `<自有ns>/stim-<size>`;README 里写明与 `github.com/STIM`、`stim.appandflow.com` 无关 |

---

## 1. 许可链:基座、数据、Kev 自身

### 1.1 基座模型

Kev 全部模型建立在 Qwen 系 Apache-2.0 基座上([README 模型表](https://github.com/jaredpalmer/kev/blob/main/README.md)):

| Kev 模型 | 基座 | 标注许可 | 训练形态 |
|---|---|---|---|
| Kev-0.8B | `Qwen3.5-0.8B-Base` | Apache-2.0 | LoRA |
| Kev-4B | `Qwen3.5-4B-Base` | Apache-2.0 | LoRA(adapter + head) |
| Kev-9B | `Qwen3.5-9B-Base` | Apache-2.0 | LoRA |
| Kev-27B | `Qwen3.8-27B`(post-trained 版) | Apache-2.0 | 全权重微调(bf16) |

独立复核(HF API,经 hf-mirror):`Qwen/Qwen3.5-4B-Base` → `cardData.license: apache-2.0`(并带 `license_link`)。Apache-2.0 基座:可商用、可再分发、可衍生,义务仅为随分发保留许可与声明。**Kev-27B 值得注意的一点**(照 Kev 自己的披露):其基座是 Qwen 的 post-trained 版本,「我们不知道它后训练用了什么」——若 Stim 走 27B 档,这一点要照抄为已知未知项。

### 1.2 训练数据集许可(Kev-4B 的 13 个公开数据集,逐个核验)

`jaredpalmer/kev-4b` 的 model card 元数据列出 13 个 dataset tag;经 HF API 逐个查许可:

| 数据集 | 卡片许可 | 备注 |
|---|---|---|
| `legacy-datasets/banking77` | cc-by-4.0 | 商用/衍生 OK,需署名 |
| `google/boolq` | **cc-by-sa-3.0** | share-alike |
| `fancyzhx/ag_news` | `unknown` | 卡片未声明 |
| `nyu-mll/multi_nli` | cc-by-3.0 / **cc-by-sa-3.0** / mit / other | 混合声明,需按上游原始许可判定 |
| `SetFit/sst5` | (无 license 字段) | 未声明 |
| `Yelp/yelp_review_full` | `other` | 非标准条款 |
| `CogComp/trec` | `unknown` | 卡片未声明 |
| `fancyzhx/dbpedia_14` | **cc-by-sa-3.0** | share-alike |
| `SetFit/amazon_reviews_multi_en` | apache-2.0 | 宽松 |
| `stanfordnlp/imdb` | `other` | 非同标准的「研究用途」类条款 |
| `bigcode/commitpackft` | mit | 宽松 |
| `nvidia/Aegis-AI-Content-Safety-Dataset-2.0` | cc-by-4.0 | 商用 OK |
| `davidheineman/consumer-finance-complaints-large` | apache-2.0 | 宽松 |

Kev-27B 的卡片另有一层更明确的许可叙述([kev-27b.md](https://github.com/jaredpalmer/kev/blob/main/docs/model-cards/kev-27b.md)):

- 「24 个被封顶的公开数据集中有四个是 share-alike:ARC(CC-BY-SA-4.0)、HotpotQA(CC-BY-SA-4.0)、Natural Questions(CC-BY-SA-3.0)、SNLI(CC-BY-SA-4.0);其余是 CC-BY-4.0、MIT 或 Apache-2.0。CFPB 投诉叙事是美国政府作品。GLM-5.3 的许可是 MIT 风格并对超大型模型即服务运营商附带条件。**逐源许可、revision 与署名记录在 component manifests 里。**」
- 「119 个 task families 来自一个公开多任务集合,许可**允许商用**且有原生标签;family 列表不公开。」
- 「CodeReviewer 的 diff 只保留自 permissive 许可的项目。」

Kev 自己的数据卫生规则(可直接照抄为 Stim 的政策):

- `scripts/devtools_v1_licences.json`:220 个 CodeReviewer 项目逐个记 SPDX(Apache-2.0 77 / MIT 50 / **NOASSERTION 38** / GPL-2.0 13 / GPL-3.0 10 / BSD-3-Clause 9 / LGPL-3.0 4 / **AGPL-3.0 4** / MPL-2.0 4 / 其他),并注明方法局限:「记录的是**当前**许可,不是采集时的许可」;FlakeFlagger 24 条全部为 `None`。
- `AGENTS.md`:「SGD share-alike、Humicroedit / cfcolor 的未授权文本**不得公开**」。
- README §License:「Apache-2.0。Qwen3 / Qwen3.5 / Qwen3.8 基座也是 Apache-2.0。**训练数据集各有各的许可**,见 model cards。」

**对 Stim 的含义**:Kev 的公开产物本身是 Apache-2.0,但「权重许可 ≠ 训练数据许可」。若 Stim 复用同一批数据源,**必须自建逐数据集许可台账**(含 revision、署名、可否商用/再分发),因为 share-alike 源会污染「全部开放」的再分发承诺;`unknown` / `other` / 无声明项须单列并优先替换。这一条直接喂给 #11。

### 1.3 Kev 自身 Apache-2.0 的可复用范围与义务

事实([LICENSE 全文](https://github.com/jaredpalmer/kev/blob/main/LICENSE),`Copyright 2026 Jared Palmer`;`pyproject.toml` 声明 `license = "Apache-2.0"`、`license-files = ["LICENSE"]`,仓库根目录**不存在 `NOTICE` 文件**):

- **授权**:§2 版权许可(复制、衍生、公开展示、再授权、分发)+ §3 专利许可,永久、全球、免费、不可撤销(专利许可在发起专利诉讼时终止)。
- **分发义务**:§4(a) 附许可副本;§4(b) 改动过的文件须带显著变更说明;§4(c) 在衍生作品的源码形式中保留所有版权/专利/商标/署名声明;§4(d) 仅当作品含 `NOTICE` 文件时才需要复制其内容 —— **kev 无 NOTICE,故不触发**。
- **商标**:§6 「本许可**不授予**使用许可方的商号、商标、服务标记或产品名的许可」,仅允许「描述作品来源」的合理惯用使用与复制 NOTICE 内容。
- **贡献**:§5 未另行声明的贡献默认按同许可提交。

**边界结论**:复制 Kev 的代码/脚本/文档 → 履行 §4(a)(b)(c) 即可,商用无限制;但**不能**把 "Kev" 用作 Stim 的品牌或暗示关系。若只是在独立实现中采用相同架构与配方,Apache-2.0 的版权义务不适用(思想/方法不受版权保护),署名属礼节。

### 1.4 引用配方/代码的写法边界(可操作清单)

1. 代码层面:凡逐文件复制者,保留版权头 + 在该文件内注明「改动说明」(§4(b));项目根放 LICENSE 副本,并在 README 的「致谢/许可」节写明 Kev(Apache-2.0, Jared Palmer)。
2. 文档层面:引用其模型卡数字/表格需注明来源链接;直接大段复制模型卡正文会落入版权范围,改写 + 引用更稳。
3. 命名层面:不得以 Kev 命名衍生项目,不得让读者以为 Stim 是 Kev 的官方后继或被其背书;可写「与 Kev 无隶属关系、受其工作启发」。
4. 数据层面:Kev 的数据集是**第三方**许可,不随 Kev 的 Apache-2.0 一起授权;不可假定可整体再分发。
5. 生态礼节(非法律义务):Kev 对 Archer Hume 架构文与 TypeSafe API 设计都做了致谢 —— 这是本领域可接受的做法,建议照做。

---

## 2. 兼容姿态:System One 兼容实现可否公开发布

### 2.1 生态先例(全部为一手可核)

- **Kev 本身**:README 第 13 行「The API matches TypeSafe's System One, so you can point their Python SDK at your local server」;第 19 行「Drop-in for Jev: the TypeSafe Python SDK works against a Kev server unchanged」;并称 API 测试「运行 TypeSafe 的示例请求与官方 SDK」。
- **TypeSafe 官方仓库**(组织 `typesafe-ai`,MIT 为主):
  - [`system-one-adapter-python`](https://github.com/typesafe-ai/system-one-adapter-python) — MIT,393★,2026-08-08 创建,自述「**Drop-in replacement for `typesafe_sdk` 的 `system_one` 求值 API,由 LLM API 而不是 TypeSafe 支撑**」;README 明确支持任意 OpenAI 兼容端点(`OpenAIProvider("grok-4", base_url=...)`、`OPENAI_BASE_URL`)。
  - `typesafe-sdk-python` / `typesafe-sdk-js`(MIT,官方 SDK)、`skills`(MIT,2713★)、`typesafe-public-examples`(MIT)。
  - 含义:客户端与服务端解耦、可换后端,是**官方自己**给出的模式;兼容实现不依赖任何官方许可即可存在。
- **第三方公开实现(仍在线)**:
  - [`razorback16/openjev`](https://github.com/razorback16/openjev) — Apache-2.0,713★,「Open, Jev-compatible System One decision server on DiffusionGemma」,首页指向 codiv.ai。
  - [`Liuziyu77/Valen`](https://github.com/Liuziyu77/Valen) — Apache-2.0,831★,「Train a Jev-like multimodal model by yourself. System One Model, now with vision.」
  - `lexmount/WebJev`(见 awesome-jev)— 「open-weight Apache-2.0 decision model(a Qwen3.5-35B-A3B fine-tune),behind the same `/v1/systemone` API」。
  - `Hangzhi/diffusion-jev-sglang` — 「independent Jev-style DiffusionGemma/SGLang server」。
  - 社区清单 [`yibie/awesome-jev`](https://github.com/yibie/awesome-jev)(2367★)成体系收录这些项目,并明确声明「收录 ≠ 质量审查/推荐,也不核验许可」。
- **无抑制迹象**:未发现上述项目被下架、发函或官方否认的记录(检索范围内)。

**含义**:公开发布 System One 兼容实现,在生态中是**已被反复实践且被官方工具链间接背书**(官方 adapter 允许换后端)的做法。

### 2.2 TypeSafe 条款中的相反信号(关键:作用对象是「客户」)

[Master Customer Agreement](https://typesafe.ai/legal/mca)(Last updated **Sep 23, 2026**)§2.3 License Restrictions 原文:

> (a) sell, lease, loan, distribute, sublicense, disclose, or otherwise offer or make the Services available as a standalone service;
> (b) **use the Services or any Output to perform model distillation, train a model to imitate the output of the Services, or develop (or to facilitate the development of) a similar or competing product or service**;
> (c) reverse engineer, decompile, disassemble, or attempt to access or derive the source code or underlying data with respect to the Services, including the underlying ideas, algorithms, structure, or organization …;
> (d) modify or create derivative works of the Services; (e) remove or obscure any proprietary notices …; (g) bypass, avoid, remove, deactivate, or otherwise circumvent: (1) any access restrictions …

§4.1–§4.2:

> TypeSafe will not … include Customer Data in a dataset used to train … any artificial intelligence or machine learning models without Customer's prior consent.
> 4.2 Output. … TypeSafe does not claim ownership of Input and TypeSafe disclaims ownership of Output. **TypeSafe hereby assigns to Customer all of its right, title, and interest, if any, in the Output.**

[Acceptable Use Policy](https://typesafe.ai/legal/acceptable-use-policy)(同日更新):

> 3.1 reverse engineer, decompile, disassemble, or access or derive the source code or data used to develop the Services; 3.2 access the Services through unauthorized means (e.g., scraping or crawling); 3.3 circumvent any security measures …

[Privacy Policy](https://typesafe.ai/legal/privacy-policy):

> We will not train or fine tune any artificial intelligence or machine learning models on your prompts or other Input.

**解读边界(分析,非事实)**:上述条款约束的是 **Services 的 Customer**(通过 API key / 控制台使用 TypeSafe 的一方),不是对所有第三方实现的普遍禁止。因此:

- 从零独立实现并公开发布:条款未直接禁止,但**也没有任何条款许可**。
- 若以 TypeSafe 账号获取 Output 来评测/蒸馏/训练:§2.3(b) 明文禁止;把本地 Kev/Stim 服务包装成替代 TypeSafe 的对外服务还叠加 §2.3(a)。
- 用 Input/Output 做**自用对照评测**与「训练模仿模型或开发竞争产品」在文本上是两件事,但同一句里出现,建议按最保守读法处理,并保留评测目的的记录。
- §4.2 的 Output 所有权转让不豁免 §2.3(b) 的**使用限制**:「你的输出」不等于「可自由用于训练替代品」。
- Kev 的做法是直接切断:kev-4b 卡「No output of Jev (TypeSafe's hosted decision model) was used」、kev-27b 卡同句;README 亦有「No Jev outputs were used for training」。这是可直接沿用的合规姿态。

### 2.3 未发现的相反信号(显式清单)

- **无公开品牌/商标政策页**:docs.typesafe.ai 的 Legal 页只列 DPA、MCA、Privacy Policy(加 AUP);`typesafe.ai/legal` 索引页返回 404。
- **无「第三方兼容实现」的明文许可或禁止**;docs 的 System One 概念页(`/concepts/system-one.md`)不含任何兼容性/竞争性表述。
- **未查商标登记**:USPTO/TMview 等未访问;"System One"/"Jev" 的商标状态**未知**,按「不宣称、不使用其标识、不做官方化表述」处理。

---

## 3. 命名空间核查(逐平台)

### 3.1 PyPI

| 名称 | 状态 | 证据 |
|---|---|---|
| `stim` | **已占用** | `https://pypi.org/pypi/stim/json` → 200;`summary = "A fast library for analyzing with quantum stabilizer circuits."`,`author = Craig Gidney`,`license = Apache 2`,最新 `1.16.0`(2026-05-22 上传),共 **127 个版本**,首页 `github.com/quantumlib/stim` |
| `kev` | **已占用(对照案例)** | 200;`summary = "K.E.V. (Keys, Extra Stuff, and Values) is a Python ORM…"`,`author = Brian Jinwright`,Apache-2.0。Kev README 的警告原文:「**PyPI name collision**: `pip install kev` installs a **different package** … This project is **not** published to PyPI.」 |
| `stim-ai` / `stim-core` / `stim-sdk` / `stim-cli` | 未占用 | 均 404 |
| `pystim` / `stimkit` | 已占用 | 均 200 |

**PyPI 命名政策相关**(一手 [PyPI Help](https://pypi.org/help/)):项目名「与既有项目过于相似、可能造成混淆」会被拒绝创建;若名字被注册但从未发布过 release,可按 **PEP 541** 的 name-transfer 流程申请。→ 采用 `stim-ai` 这类「同名主体 + 后缀」在 PyPI 上**不保证通过**,发布前应先提交并观察。

### 3.2 npm

| 名称 | 状态 | 证据 |
|---|---|---|
| `stim` | **已占用且活跃** | `https://registry.npmjs.org/stim` → 200;`dist-tags.latest = 1.21.0`,30 个版本,`modified = 2026-10-16T01:06Z`;`description = "Stim: fast, isolated React Native environments for coding agents"`,`license = MIT`,repo `github.com/appandflow/stim`,维护者 `janicduplessis`、`charlesvinette` |
| `appandflow/stim` | GitHub 仓库已占用 | MIT,82★,创建 2026-04-27,pushed 2026-10-16,首页 `https://stim.appandflow.com/` |
| `stim-ai` / `stim-js` / `stimllm` | 未占用 | 均 404 |

**风险提示**:npm 的 `stim` 是 **2026 年、面向 coding agents 的活跃项目**,与 Stim 的受众(agent / 开发者工具)高度重叠,混淆风险高于 PyPI 的量子库;若发布 npm 包需换名 + 显式「not affiliated」。

### 3.3 Hugging Face(经 hf-mirror.com)

| 名称 | 状态 | 证据 |
|---|---|---|
| 用户/组织 `stim` | **已注册**(近乎闲置) | 页面标题 `stim (Max Stieber)`;`/api/models?author=stim` → 0,`/api/datasets?author=stim` → 0;`/api/users/stim` 与 `/api/organizations/stim` 均 404(镜像侧接口) |
| `stim-ai` / `stim-llm` / `getstim` | 页面 404 → 未见占用 | 镜像页面标题 `404 – Hugging Face` |
| `0xnicholas` | **尚不存在** | 页面 404;`/api/models?author=0xnicholas` → `[]` |

→ **`stim/stim-4b` 这种「以 stim 为 namespace」的方案不可行**;须先注册自有 namespace 再发布。

### 3.4 权重仓库命名方案

- 方案 `<自有ns>/stim-<size>`(如 `0xnicholas/stim-4b`):**干净**。
  - HF 模型全站搜索 `stim-9b` → 0 条;`stim-4b` → 1 条无关结果(`Rorolezozo/Qwen3-4B_finetuned_V1_stimuli_recognition`)。
  - Kev 用同一结构:`jaredpalmer/kev-4b`、`kev-9b`、`kev-27b`(核实 `license: apache-2.0`;4B 为 `peft` + safetensors 的 LoRA,27B 为 `transformers` + safetensors 的全权重微调)。
- 命名若要跨平台一致:**「模型家族名 = Stim」+「发行包名 ≠ stim」**是可复制的模式(Kev 因撞名干脆不发布到 PyPI)。这属于 #15 的决策输入,本票只陈述约束。

---

## 4. 给决策票的输入(不做决策,仅列约束与选项)

**给 #15(开源与发布策略)**:

1. 发布 `pip install stim` **不可行**(已被 quantumlib 占用);可选路径:(i) 不发布到 PyPI(照 Kev);(ii) 用带后缀发行名(如 `stim-ai`,未占用,但受 PyPI 反混淆规则约束);(iii) 换家族名。
2. npm 同名冲突更棘手且更活跃,若做 TS 工具链须另起包名。
3. HF 权重命名:必须自有 namespace;`stim-<size>` repo 名本身干净。
4. 兼容声明文案建议照 Kev 模式:说明「官方 SDK 可直接指向本服务」,同时声明与 TypeSafe、Kev、`appandflow/stim`、quantumlib/Stim **无关联**。
5. 用 Kev 代码/文档时履行 Apache-2.0 §4(a)(b)(c) + 致谢;不得使用 Kev 品牌暗示背书。

**给 #11(训练数据与生成策略)**:

1. 数据许可须逐源记录(HF 卡片的 `unknown`/`other`/无声明**不等于**可商用):建议照 Kev 建 `component manifests`(源、revision、SPDX、署名)。
2. share-alike 源(CC-BY-SA 系列)对「全部开放、可再分发」的承诺有传染风险,建议列白/黑名单。
3. **硬规则:不得用 Jev/TypeSafe 的输出训练**(Kev 已自我施加同一规则;对 TypeSafe 客户而言 MCA §2.3(b) 是合同义务)。
4. 若用 TypeSafe 账号做对照评测:仅限评测,不蒸馏、不抓取、不绕过(AUP §3),并保留用途记录。

---

## 5. 不可达 / 未验证来源清单(显式)

| 来源 | 状态 | 影响 |
|---|---|---|
| `huggingface.co` 直连 | 不可达 | 全部 HF 数据经 **hf-mirror.com** 的 API 与页面;镜像数据可能滞后/缺字段 |
| `raw.githubusercontent.com` | 不可达 | 未直接读取数据集上游 LICENSE 原文;数据集许可结论**仅基于 HF `cardData`**,不可当作上游结论 |
| `registry.npmjs.org` 全量元数据 | 首轮请求超时(>60s) | 改用 abbreviated metadata(`Accept: application/vnd.npm.install-v1+json`)+ `/stim/latest` 成功;`dist-tags`/版本数来自 abbreviated 文档 |
| `api.typesafe.ai` | 本环境 HTTP 451(地区封锁) | 未做活体 API 探测,全部结论来自文档与法律页文本 |
| WebSearch 工具 | 403 | 全部结论来自 `curl` / `gh` 直取 |
| 商标登记库(USPTO / TMview) | 未访问 | "System One" / "Jev" / "Stim" 的商标状态未知;未发现 TypeSafe 公开品牌政策页 |
| `SetFit/sst5`、`fancyzhx/ag_news`、`CogComp/trec`、`Yelp/yelp_review_full`、`stanfordnlp/imdb` | HF 卡片许可为 空 / `unknown` / `other` | 这几个源的**真实**上游许可未确认,#11 需人工核对 |
| HF 用户名 `stimai` | 页面返回 200 但未捕获到标题 | 状态**未确认**(记录为歧义项) |

---

## 6. 主要来源索引

- Kev(LICENSE / README / model cards / AGENTS.md / `scripts/devtools_v1_licences.json`):本地浅克隆 `/tmp/kev-src`,`https://github.com/jaredpalmer/kev`
- Kev 权重与数据标签:`https://hf-mirror.com/api/models/jaredpalmer/kev-4b`、`.../kev-9b`
- Qwen 基座:`https://hf-mirror.com/api/models/Qwen/Qwen3.5-4B-Base`
- TypeSafe 法律文本:`https://typesafe.ai/legal/mca`、`https://typesafe.ai/legal/acceptable-use-policy`、`https://typesafe.ai/legal/privacy-policy`、`https://docs.typesafe.ai/legal.md`
- TypeSafe 官方仓库:`https://github.com/typesafe-ai/system-one-adapter-python`、`https://github.com/typesafe-ai/typesafe-sdk-python`
- 生态第三方:`https://github.com/razorback16/openjev`、`https://github.com/Liuziyu77/Valen`、`https://github.com/yibie/awesome-jev`
- 命名空间:`https://pypi.org/pypi/stim/json`、`https://pypi.org/pypi/kev/json`、`https://pypi.org/help/`、`https://registry.npmjs.org/stim`、`https://registry.npmjs.org/stim/latest`、`https://github.com/quantumlib/Stim`、`https://github.com/appandflow/stim`、`https://github.com/STIM`

## 7. 复现命令

```bash
# 许可链
head -25 /tmp/kev-src/LICENSE; tail -15 /tmp/kev-src/LICENSE
grep -n -i "licen" /tmp/kev-src/docs/model-cards/kev-4b.md /tmp/kev-src/docs/model-cards/kev-27b.md
python3 -c "import json,collections;d=json.load(open('/tmp/kev-src/scripts/devtools_v1_licences.json'));print({k:collections.Counter(x.get('spdx') for x in v.values()) for k,v in d.items() if isinstance(v,dict) and v and isinstance(list(v.values())[0],dict)})"

# 数据集许可
for ds in legacy-datasets/banking77 google/boolq fancyzhx/ag_news nyu-mll/multi_nli SetFit/sst5 \
          Yelp/yelp_review_full CogComp/trec fancyzhx/dbpedia_14 SetFit/amazon_reviews_multi_en \
          stanfordnlp/imdb bigcode/commitpackft nvidia/Aegis-AI-Content-Safety-Dataset-2.0 \
          davidheineman/consumer-finance-complaints-large; do
  printf "%-52s " "$ds"; curl -s "https://hf-mirror.com/api/datasets/$ds" \
    | python3 -c "import json,sys;d=json.load(sys.stdin);c=d.get('cardData') or {};print(c.get('license') or [t for t in d.get('tags',[]) if t.startswith('license')])"
done

# TypeSafe 条款
curl -sL https://typesafe.ai/legal/mca | python3 -c "import sys,re,html;t=sys.stdin.read();t=re.sub(r'<script.*?</script>','',t,flags=re.S);x=re.sub(r'\s+',' ',html.unescape(re.sub(r'<[^>]+>',' ',t)));i=x.find('2.3. License Restrictions');print(x[i:i+1200])"
curl -s https://docs.typesafe.ai/legal.md

# 命名空间
for p in stim stim-ai stimllm stim-core stim-sdk stim-cli pystim stimkit; do printf "PyPI %-10s " $p; curl -s -o /dev/null -w "%{http_code}\n" https://pypi.org/pypi/$p/json; done
for p in stim stim-ai stim-js stimllm; do printf "npm  %-10s " $p; curl -s -o /dev/null -w "%{http_code}\n" https://registry.npmjs.org/$p; done
curl -s https://registry.npmjs.org/stim/latest | python3 -c "import json,sys;d=json.load(sys.stdin);print(d['name'],d['version'],d.get('license'),d.get('repository'),d.get('description'))"
gh api orgs/STIM --jq '{login,created_at,public_repos}'
for u in stim stim-ai stim-llm getstim 0xnicholas; do printf "HF   %-10s " $u; curl -s "https://hf-mirror.com/$u" | grep -o -E '<title>[^<]*</title>' | head -1; echo; done
curl -s "https://hf-mirror.com/api/models?search=stim-9b&limit=5"
```
