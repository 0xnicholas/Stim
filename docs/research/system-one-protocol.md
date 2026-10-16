# Jev 架构与 System One 协议解剖

**研究票**:[#4](https://github.com/0xnicholas/Stim/issues/4) · **用途**:支撑 #13(System One 兼容层实现策略)与 #9(基座抉择)的「逐项对齐表 + SDK 验收方式」
**日期**:2026-10-16 · **调查范围**:架构事实、线协议、SDK 行为。不做选型建议,不复现模型。

---

## 0. 一句话结论

Jev 的**线协议是公开且完整可对齐的**(端点、请求/响应 schema、错误码、confidence 公式全部有官方或参考实现来源);Jev 的**架构只有部分公开**:「prefill-only + 直接概率读出 + 共享前缀 + 问题隔离」有官方陈述与第三方黑盒实证支撑,而「backbone 用什么、MoE 与否、readout 是 slot head 还是 pointer」属于推断,其中 pointer/bilinear 与 Kev 的实际实现一致。**兼容性要分三层看:线兼容(schema)可完全达成,派生量兼容(confidence/score 公式)可精确复刻,语义兼容(校准、温度、并发、确定性)不存在可对齐的公开目标值。**

---

## 1. 来源与证据等级

| 来源 | 类型 | 证据等级 | 本文引用 |
|---|---|---|---|
| `https://docs.typesafe.ai/api` | TypeSafe 官方 API 文档 | 一手(官方) | 线协议权威 |
| `https://docs.typesafe.ai/models` | 官方 | 一手(官方) | 模型 ID、别名、上下文限制、long-context header |
| `https://docs.typesafe.ai/confidence` | 官方 | 一手(官方) | confidence 精确公式 |
| `https://docs.typesafe.ai/introduction/machine-learning-primer` | 官方 | 一手(官方) | RLCD 训练路径 |
| `https://docs.typesafe.ai/patterns/fan-out` | 官方 | 一手(官方) | 并发/多问题语义 |
| `https://archerhume.com/posts/jevs-architecture-unmasked` | 第三方黑盒调查(2026-09-17,针对 `jev-1.13.0`,约 10,000 次 API 调用) | 一手(对公开模型的实验观察),但架构结论为**推断** | 架构解剖,标注推测性 |
| `system-one-adapter` 0.2.1(PyPI,TypeSafe AI 发布) | 官方参考适配器源码 | 一手(官方) | confidence 参考实现 |
| `typesafe-sdk` 0.7.5(PyPI,TypeSafe 发布) | 官方 SDK 源码 | 一手(官方) | 客户端强依赖行为 |
| `jaredpalmer/kev` @ `1f04cfa2`(2026-10-15) | 开源实现 | 一手(实现) | Kev 偏离项 |
| `denis-pplx/autojev-27b` 模型卡 | 第三方开源实现 | 一手(实现声明) | 第三实现对照 |

**注意**:`api.typesafe.ai` 在本环境返回 **HTTP 451(地区封锁)**,因此本文所有线协议事实来自**官方文档 + 官方 SDK 源码 + 官方参考适配器源码**,**没有**对真实 Jev 端点的活体探测。

---

## 2. Jev 的公开架构

### 2.1 官方明确陈述的部分

- Jev 是 **System One 模型**的旗舰,输出「一次请求、结构化答案」而非文本生成:TypeSafe 发布声明称 *"Jev outputs all probabilities in parallel instead of autoregressively generating by token"*(经 archerhume 引用)。
- 训练方法是 **RLCD(Reinforcement Learning for Calibrated Decisions)**;ML primer 将其描述为「从预训练语言模型出发的 post-training 路径」。**具体配方未公开**。
- **Jev 不做客户数据微调**:官方明确 *"Jev is not fine-tuned or LoRA-adapted with customer data... the same weights serve every account"*([models](https://docs.typesafe.ai/models))。
- 上下文与能力(`jev-1.13.0` / `jev-1.14.0`):**per-request 64k / per-question 32k** 为默认;加 header `TypeSafe-Danger-Zone-Long-Context: true` 后两者均为 **262k**。价格 $42/Btok 输入,**输出 token 免费**。
- 别名:`jev-latest` → `jev-1.13.0`(SDK 默认);`jev-preview` → `jev-1.14.0`。别名会漂移,响应 `model` 字段回报实际版本 ID。

### 2.2 第三方黑盒实证(archerhume,针对 jev-1.13.0)

这些是**观察到的行为**,不是官方架构披露:

| 观察 | 数据 | 支撑的结论 |
|---|---|---|
| token 计数可加性 | 1 个 noul = 268 input tokens;2 个 = 276;noul+2-option Choice+2-level Score = 318 | 共享前缀 + 每题后缀的计算形态 |
| 证据位置对照实验 | 秘密放在**兄弟问题**里 → 概率 0.00;放进 **state** → 0.90–0.92 | 问题之间行为隔离,且都能读共享 state |
| 延迟随问题数 | ~100 题以内服务器时间几乎不变;之后缓升;**按 token 计,问题文本成本约为 state 的 2 倍** | state 算一次 + 问题侧批处理 |
| 上下文限制 | 每分支(state + 一题)≈ 32,768;整请求 ≈ 65,536;请求限制**只计一次 state**(23k state + 5,000 题可容纳) | 形状吻合「单条 packed 序列上限 2¹⁶,state 一次 + 全部问题在后,每分支 2¹⁵ 窗口」 |
| **IIA 违反**(选项互相作用) | 4 选项 → 追加无关第 5 选项,`log[P(customer)/P(unknown)]` 从 **+0.38 降到 +0.11**,10 个随机区块**全部下降**,均值变化 **−0.28**(95% 配对 t 区间 −0.36 ~ −0.19) | **否定**「每选项独立固定 logit + 不变 softmax」;选项在一个 listwise 表示里互相作用 |
| 选项位置敏感 | 参照卡放在**最后一个选项** → 16/16 全对(平均正确概率 ~0.88);放**第一** → 12/16;放**中间** → 11/16;放 **state** → 48/48 | 决策可读完整选项列表;存在位置敏感处理 |
| 选项顺序敏感 | 反转选项把某个技术支持分类的概率从 ~0.84–0.89 推到 **0.93–0.96** | **同一阈值 0.9 会因选项顺序改变动作** → permutation 测试是必需评估项 |
| 选项上限 255 | 255 = 2⁸−1 | 「适配固定 256 slot 头」,但该限制由**请求校验**强制,非模型 |
| 200 选项行为 | 复制答案在**每个位置都得 1.00**,错误不溢到邻近选项 | 指向 **pointer** 式打分器(非 slot head) |
| 伪造选项 | 注入伪造选项从未顶掉真实选项;选项边界**文本无法伪造** | 边界由不可伪造的标记划定 |
| `output_tokens` 语义 | noul 精确符合 **4 个共享 token + 每答案 15 个 + 每个问题标识符的 token 长度**;答案值(0.0 vs 0.01)不影响计数 | `output_tokens` 是**计费数字**,推理后从序列化响应算出,不是生成量度 |
| 无确定性 | 重复请求(含同请求内重复问题)有微小差异,响应 key 顺序也有少数几种固定模式 | **不应假设 API 级确定性**(数值 kernel / 动态批 / 路由 / 有意随机皆可能) |
| 概率精度 | 一般返回**两位小数** | 与 Kev 的 4 位小数不同 |
| tokenizer | 匹配不上测试的 192 个公开 tokenizer;最接近的是 Qwen(415 次探测中一致 348 次);数字逐位切分 | 排除「未改动的公开 tokenizer」,**不排除**公开基座(换词表/继续预训练/蒸馏均可解释) |

### 2.3 推断部分(置信度递减)

作者本人明确分层:

1. **直接数值读出**(非生成):官方陈述支撑 → 高置信。
2. **问题隔离 + 选项顺序效应**:行为可观测 → 高置信。
3. **KV 共享 / causal attention**:工程上最自然的解释(B 分支 32k / 请求 64k 只在共享下成立)→ 中高置信。
4. **readout 形态**:**两种都符合证据** —— (a) 末位 head 从 `<decide>` 表示给每个选项槽打分;(b) **pointer 式打分器**,把决策表示与每个选项自身的最终隐状态比较。200 选项实验偏向 (b),但**不决定性**。
5. **backbone 为稀疏 MoE**:纯推断。论据是 prefill-only 模型受算力限制、MoE 在无逐 token 解码场景下服务成本劣势消失、Jev 约 160 ms 处理 ~30k token(密集 70B 在 8×H100 上约需 1 秒,MoE 约 10B active 才吻合)、以及近期强基座多为 MoE。作者强调:**「其他部分不依赖这一条」**。
6. **backbone 身份/tokenizer**:未知,公开证据无法确定。

### 2.4 关键否定项

- **不是** diffusion:没有任何实验需要迭代去噪。
- **不是**「每选项独立打分再归一化」:IIA 违反实验直接否定。
- **不是**生成文本再解析:`output_tokens` 与延迟解耦(200 选项 / 1,911 output tokens 的题目与 2 选项一样快;服务器时间只随输入长度增长)。

---

## 3. Kev 的实现与对 Jev 的偏离

Kev 自称 *"based on the architecture described in Jev's Architecture Unmasked"*([README](https://github.com/jaredpalmer/kev/blob/main/README.md))。**它是上述 2.3 推断的一个具体落地选择**,不是对 Jev 内部实现的复刻。

### 3.1 逐模块映射

| 组件 | Kev 实现 | 与 Jev 公开证据的关系 |
|---|---|---|
| Backbone | Qwen3.5 / Qwen3.8 **Base**(`Qwen/Qwen3.5-4B-Base`、`Qwen3.5-9B-Base`、`Qwen3.8-27B` 等),**只取 backbone,不要 vocab head**(「我们从不生成文本」) | Jev 的 backbone 未知。Kev 用密集 Qwen,**没有**采用 MoE 推断 |
| 适配 | **LoRA rank 16,α 32**,打在 attention + MLP 投影上;hybrid 基座额外打 Gated DeltaNet 投影(Kev-9B:45.4M 参数) | 与 Jev「不用客户数据做 LoRA」不冲突(那是托管产品策略,非架构) |
| 共享前缀 | 训练侧 `kev/shared_prefix.py`:state 只跑一次,各分支从它继续,梯度回流入 state;推理侧靠 state-prefix KV cache | **直接对应** Jev 的核心公开性质(共享 state、隔离问题) |
| 问题隔离 | **block-causal 分支掩码**:`attend(i,j) iff j<=i and (seg[j]==0 or seg[j]==seg[i])` | 对应「问题不能互相看」 |
| 选项隔离(可选) | `option_isolation=True`:每个选项 span 成为独立子分支(只见 state+instruction+自身),共享 position id,`<decide>` 固定在最长 span 之后 → **排列不变性由构造保证** | Jev 的 IIA 违反说明选项**会**互相作用;Kev 提供该开关作为 permutation 不变性的可选项,默认关 |
| **Readout** | **PointerHead**:`q,k: Linear(d → 256)`,`z = (k(h_opts) @ q(h_decide)) * (1/√256)`。**双线性 pointer**,选 2.3 的 (b) | 与 200 选项实验偏向的 pointer 一致 |
| 温度校准 | 每检查点一个标量 `head.pt["temperature"]`,**只在 eval 模式除 logits**;训练时恒为 T=1;argmax 不变 | Jev **不暴露**任何温度参数 |
| 锚点(anchors) | `kev/anchors.py`:**冻结基座自己的 zero-shot 分布**(从 next-token 字母 A/B/C… logits 读出,softmax),作为 KL 目标(`--anchor --anchor_w`)。**教师=学生自己的基座,不涉及 Jev 或任何外部模型** | Jev 无对应公开概念。PLAN.md 记录:锚定**未被采纳**为 remedy |
| 软标签 | `--label_smoothing`(硬标签 CE 平滑)、开放标注者投票分布的 soft targets、`KL(base ‖ model)`。PLAN.md 立场:*"Continue training with soft targets, not hard labels."* | Jev 的 soft label 用法未公开 |
| 对比数据 | `kev/contrastive.py`:程序化生成「策略+案例」配对,标签是 facts 的**纯函数**;每对过两道**无模型**检查 —— `ablation`(去掉证据句必须变 UNDETERMINED)与 `invariance`(去掉 filler 必须不变) | Jev 无公开对应物;**这是 Kev 独有的数据构造** |
| 分隔符 | 复用 Qwen 既有冷门 special token 作 `<state>`/`<q>`/`<opt>`/`</opt>`/`<decide>`(`<|fim_prefix|>` 等),**不新增 embedding 行**;`user_tokens()` 把用户文本里的 `<\|name\|>` 改写成 `<¦name¦>`,使选项边界**不可伪造** | 对应「选项边界文本无法伪造」这一观察 |
| 掩码 vs 行形式 | hybrid 基座(Qwen3.5 Gated DeltaNet 层)**无法**满足 block-causal 掩码 → 每题一条 causal row(state + branch)。两种形式在 fp32 下一致到 ~1e-5 | 实现约束,Kev 特有 |

### 3.2 服务上下文:两者数值对照

| | Jev 1.13/1.14 | Kev |
|---|---|---|
| per-request | 64k(默认)/ 262k(long-context header) | `SERVE_MAX_STATE = 65536`(state)+ `SERVE_MAX_BRANCH = 65536+8192`(state+一题行) |
| per-question | 32k(默认)/ 262k | 同上,行上限 |
| 训练上下文 | 未公开 | `MAX_STATE=384, MAX_BRANCH=1024, MAX_PACKED=2048` |
| 超限行为 | `400` + `max_tokens_exceeded`(官方文档) | `422` + `detail`(点名 token 数与修复法);`KEV_TRUNCATE_STATES=1` 改为截断并在**每个响应**标记 `truncated` + `usage.state_tokens` / `state_tokens_used` |

### 3.3 `kev/jev.py` 是什么(重要澄清)

`kev/jev.py` **不是 Jev 模型或 Jev 协议实现**。它是「**把 Jev 当基线来评测**」的 CLI:通过 Vercel AI Gateway 临时开通一个 $1/7 天的 key,用 `JevPredictor` 跑 `evaluate_records`,把 Jev 的分数写进 `report.json` / `usage.json`。协议对齐的实现全在 `kev/api.py`(schema ↔ pointer primitive 映射)与 `kev/serve.py`(HTTP 层)。

### 3.4 API 映射(共用一个 pointer primitive)

`kev/api.py` 把三种题型塌缩到**一个指针原语**:

| 题型 | options | 返回键 |
|---|---|---|
| Noul | `[no, yes]` | `["false","true"]`,答案 = `p(true)` |
| Choice | `name` 或 `name: desc` | criteria 名 |
| Score | 有序 level 描述 | 级别下标字符串 |

- `option_text(name, desc)`:`desc` 为 `None`/`""` 时只发 `name`,否则 `"{name}: {desc}"` —— 这解释了 SDK 允许 `criteria: {"calm": None}`。
- `question_keys()` 让 labels / targets / anchors 共用同一套键。
- 序列化精度:`round_prob()` = **4 位小数**,理由写在注释里:255 个选项时 `255 × 0.00005 < 0.02`,保证 `|sum−1|` 在 TypeSafe 容差内。**Jev 实测返回两位小数**。

---

## 4. System One 线协议(精确)

### 4.1 端点与认证

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer <API_KEY>
Content-Type: application/json
```

- `GET /v1/models`:**该账号可用的模型名**,每个带 `description` 与 `release_date`。当前列出的是别名;版本化 ID(如 `jev-1.14.0`)即使不在列表里也被 `model` 字段接受。
- SDK 常量:`SYSTEM_ONE_PATH = "/v1/systemone"`,`MODELS_PATH = "/v1/models"`。
- **认证在 SDK 里是强制的**:`api_key` 缺省抛 `TypeSafeError`;key 必须是可打印 ASCII、无空白(所以 Kev 的测试用 `api_key="local"`)。

### 4.2 请求规则(全部请求通用)

| 规则 | 限制 | 违规 |
|---|---|---|
| Body 大小 | ≤ **32 MiB**(header 不计) | `413` |
| 编码 | UTF-8 JSON,**无 BOM** | `400` |
| 嵌套 | `state` / `instructions` / `criteria` 各自 ≤ **64 层**对象/数组 | `400` |
| 数字 | 仅有限数;`NaN`、`Infinity`、`1e400` 等被拒。标识符与大整数请用字符串 | `400` |

**压缩**:可用 `Content-Encoding: gzip`(或 `x-gzip`)或 `zstd`;**只允许一层**,zstd 窗口 ≤ 8 MiB;32 MiB 限制**同时**适用于压缩前后。不支持的编码或多层 → `415`;压缩体损坏 → `400`。

### 4.3 请求体

```jsonc
{
  "state": "<string | object | array>",   // 必填,被评估的内容
  "model": "jev-latest",                  // 模型或别名
  "questions": {                          // map<string, Question>,≥1
    "<你选的 id>": { ... }
  }
}
```

- **`questions` 的 key 不会发给底层模型,也不参与推理**;答案按同一 key 返回。ID 虽不进模型,**却计入 `output_tokens`**(见 2.2)。
- `instructions` 可以是 **string / object / array**;结构化写法把问题放一个字段、被引用数据放其他字段,用反引号按名引用。

#### Noul(yes/no,返回「是」的概率)

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"noul"` | |
| `instructions` | `string \| object \| array` | 要判断的 yes/no 问题 |
| `criteria` | `{true, false}`(可选) | 描述 yes(接近 1)与 no(接近 0)的含义 |

#### Choice(从你定义的集合中选一个)

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"choice"` | |
| `instructions` | `string \| object \| array` | |
| `criteria` | `map<string, string \| object \| array \| null>` | 选项 → rubric 描述;**`null` = 无需额外说明**;**最多 255 个选项** |

#### Score(按有序级别打分)

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"score"` | |
| `instructions` | `string \| object \| array` | |
| `criteria` | `array<string \| object \| array>` | **有序**级别描述数组。官方文档:*"should have at least two levels; the API accepts up to 10"* |

> ⚠️ **文档与实现的不一致(需在 #13 记录)**:官方 API 文档说 Score「至少 2 级、最多 10 级」,但 **SDK 只强制 `min_length=1`**(`Score.criteria: Sequence[Content] = Field(min_length=1)`),而 **Kev 接受 1..255** 并有一个明确断言单级 Score 可用的测试(`tests/test_api.py::test_optional_instructions_and_one_level_score` 期望 `{"score": 0.0, "legend": {"0": "today"}, "probabilities": {"0": 1.0}, "confidence": 1.0}`)。Jev 真实上限未在文档外证实。

### 4.4 响应体

```jsonc
{
  "model": "jev-1.13.0",       // 实际作答的模型
  "answers": { "<同一 id>": { ... } },
  "usage": { "input_tokens": 296, "output_tokens": 20 }
}
```

#### Noul answer

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"noul"` | |
| `noul` | `number` | 0(否)~ 1(是)的尺度。**无 `confidence` 字段** |

#### Choice answer

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"choice"` | |
| `choice` | `string` | 概率最高的选项 |
| `probabilities` | `map<string, number>` | 每个选项 → 概率,**总和约为 1** |
| `confidence` | `number` | 由 probabilities 派生,0~1 |

#### Score answer

| 字段 | 类型 | 语义 |
|---|---|---|
| `type` | `"score"` | |
| `score` | `number` | **跨级别的概率加权平均**,可以落在级别之间 |
| `legend` | `map<string, string>` | 级别号 → 原始描述 |
| `probabilities` | `map<string, number>` | 级别下标**字符串**为键,总和约为 1 |
| `confidence` | `number` | 由 probabilities 派生,0~1 |

**派生量的精确公式**(官方 [confidence](https://docs.typesafe.ai/confidence) 页 + 官方参考适配器 `system_one_adapter/_utils/confidence_metrics.py` 一致):

```python
# K = 选项数(含单选项特例返回 1.0)
def _normalize(p):                      # 总和为 0 时退化为均匀
    t = sum(p)
    return [1/len(p)]*len(p) if t == 0 else [x/t for x in p]

def choice_confidence(p):               # Choice 与 Noul(按 2 选项 Choice 处理,即 |2p−1|)
    K = len(p)
    return 1.0 if K == 1 else (max(_normalize(p)) - 1/K) / (1 - 1/K)

def score_confidence(p):                # L = 级别数;mode = 概率最大的级别下标
    L = len(p)
    if L == 1: return 1.0
    p = _normalize(p); mode = max(range(L), key=p.__getitem__)
    D = sum(abs(i - (L-1)/2) for i in range(L)) / L          # 均匀分布的 MAD
    return max(0.0, 1.0 - sum(pi*abs(i - mode) for i, pi in enumerate(p)) / D)
```

官方文档给出的自算版本:

- Noul(文档建议,非 API 返回):`confidence = |2p − 1|` ——「也是 Choice 公式作用在 yes/no Choice 上的结果」。
- Choice:`(p_max − 1/n) / (1 − 1/n)`。只取 top 概率,所以 `(0.6,0.3,0.1)` 与 `(0.6,0.2,0.2)` 的 confidence 都是 0.4。
- Score:`max(0, 1 − Σᵢpᵢ|i−m| / MAD_unif)`。3 级时 `(0, 0.5, 0.5)` → 0.25,`(0.5, 0, 0.5)` → 0(同样分布下 Choice 公式会给 0.25)。

> Kev 的 `kev/api.py` 注释显式声明镜像 `system-one-adapter 0.2.1` 的 `confidence_metrics.py`,两者公式**逐行一致**。

### 4.5 错误与边界

| 状态码 | 含义 |
|---|---|
| `400 Bad Request` | 违反请求规则(非法 UTF-8、嵌套过深、非有限数、压缩体损坏)。带 **`max_tokens_exceeded`** 时表示请求长于模型上下文 |
| `401 Unauthorized` | key 缺失或无效 |
| `413 Payload Too Large` | body > 32 MiB |
| `415 Unsupported Media Type` | 不支持的 `Content-Encoding` 或超过一层压缩 |
| `422 Unprocessable Entity` | 请求体校验失败(缺必填字段、畸形 question);**body 会指出出错字段** |
| `429 Too Many Requests` | 超出速率限制(按组织,在 console 配置;token/s 与 request/s 双维度) |
| `529 Overloaded` | 暂时过载,稍后重试 |

限流处理:官方要求用**指数退避**重试 429/529;SDK 默认自动处理。

### 4.6 边界情况汇总

| 情况 | Jev 行为 |
|---|---|
| `criteria` 选项描述为 `null` | 合法,选项只发名字 |
| `instructions` 缺失 | SDK 侧**拒绝**,除非 Noul 的 `criteria` 至少有一个非 `None` 值 |
| Choice `criteria` 为空 map | SDK 侧拒绝(`at least one choice is required`) |
| Score `criteria` 为空数组 | SDK 侧拒绝;服务端 `422` |
| Score 只有 1 级 | SDK 允许(`min_length=1`);文档说「至少 2」——**不一致** |
| `questions` 为空 map | SDK 侧拒绝(`At least one question is required`);服务端 `422` |
| 未知题型 | 服务端 `422`;响应侧 SDK 会**丢弃**未知 `type` 的答案并打 warning |
| 未知请求/响应字段 | 响应侧忽略;请求侧 SDK 的类型检查会报错,但**运行时是转义口**(见 5.3) |
| 图片 | `state`、`instructions`、`criteria` 均可带 image asset;有 asset 时 body 变为 `multipart/form-data` |
| **Box 题型** | 官方文档与 SDK 都有(`BoxAnswer{box:{top,bottom,left,right}, confidence?}`);**Kev 未实现** |

---

## 5. TypeSafe Python SDK 的实际行为与隐含假设

**包名**:`typesafe-sdk`(PyPI 最新 **0.7.5**;Kev 约束 `>=0.6.0`;`system-one-adapter 0.2.1` 要求 `>=0.7.0`)。
**依赖**:`anyio>=4`、**`httpx2>=2.0.0`**(不是 `httpx`)、`packaging`、`pydantic>=2.12.0`、`pydantic-core>=2.41.1`、`tenacity>=9`、`truststore`、`typing-extensions`;`[zstd]` extra(或 Python ≥3.14)、`[http2]`、`[pil]`。

### 5.1 默认值与常量(全部来自官方源码)

| 常量 | 值 | 环境变量 |
|---|---|---|
| `DEFAULT_BASE_URL` | `https://api.typesafe.ai` | `TYPESAFE_BASE_URL` |
| `DEFAULT_MODEL` | **`jev-latest`** | `TYPESAFE_DEFAULT_MODEL` |
| `DEFAULT_TIMEOUT` | **10.0 s** | — |
| API key | 必填 | `TYPESAFE_API_KEY` |
| 日志级别 | — | `TYPESAFE_LOG_LEVEL` |

连接池:`max_connections=100`、`max_keepalive_connections=100`、`keepalive_expiry=30.0`(httpx2 默认只保 20 条 idle 5 秒)。

### 5.2 重试策略(`RetryPolicy` 默认值)

| 项 | 默认 |
|---|---|
| `max_retries` | **2**(即最多 3 次尝试) |
| `backoff_initial` / `backoff_max` | 0.5 s / 5.0 s(指数,`ldexp`) |
| `backoff_jitter` | 0.25(每延迟随机减掉最多 25%) |
| `http_statuses` | **`{408, 429} ∪ [500, 600)`** |
| `respect_retry_after` | True(读 `retry-after` **和** `retry-after-ms`;后者单位 ms,前者支持秒数或 HTTP-date) |
| 连接错误 / 超时错误重试 | 都 True |
| `timeout`(重试预算) | 30.0 s,含首次尝试与延迟 |

**422 不在重试集合** → 校验失败立即抛出。

### 5.3 请求构造与 forward compatibility

- 发送 header:`Authorization: Bearer <key>`、`Accept: application/json`、`User-Agent: typesafe-sdk/<ver>`、`X-TypeSafe-SDK: typesafe-sdk/<ver>`、`X-TypeSafe-Runtime`;重试时额外带 **`X-TypeSafe-Retry-Count`**(构造请求前会从用户 header 里剔除,防止伪造)。
- **`system_one()` 完整签名**:`state`, `questions`, `*, model=None, assets=None, retry=None, timeout=None, extra_headers=None, extra_body=None, response_model=None`。**没有 temperature、没有 beam width、没有 batch/concurrency 参数。**
- `extra_body` 是「**浅合并、后写覆盖**」:与 `state`/`model`/`questions`/`assets` 冲突的键会**覆盖**它们,且对象值整个替换而非深合并。官方示例用 `{"beam_width": 4}` 并注明「只发 API 支持的字段」——**即 `beam_width` 只是示意,不是已支持字段**。
- `questions` 可以是 SDK 的 `Noul`/`Choice`/`Score`/`Box` 对象,**也可以是裸 dict**(带额外字段,官方称为 forward-compat 转义口)。
- SDK 侧发出前校验(`normalize_questions`):≥1 个问题;Choice criteria 非空;Score criteria 非空;Noul 需 `instructions` 或 criteria 至少一个非 `None`。

### 5.4 响应解析(客户端强依赖)

- 所有响应模型 `extra="ignore", frozen=True, strict=True` → **未知字段被忽略**(所以 Kev 的 `latency_ms`、`truncated`、`usage.state_tokens` 不影响解析)。
- **答案按 `type` 判别联合校验**;遇到不认识的 `type`:打 warning、**从 `answers` 里删掉**、继续解析其余;raw payload 仍在 `raw_http_response`。
- `ScoreAnswer` 把 `legend` / `probabilities` 的键**从 str 强转为 int**(`dict[int, ...]`)。
- **SDK 侧不校验** `probabilities` 是否求和为 1,也**不校验** `choice == argmax(probabilities)`。这两条只有 Jev 服务端 / Kev 自己的测试在管。
- `response.request_id` 取自 **`x-typesafe-request-id`** 响应头。
- `client.models.list()` 解析 `GET /v1/models` 时,**每个 card 的 `name`/`description`/`release_date` 都是必填无默认**,缺一个就抛 `TypeSafeAPIResponseValidationError`;`models` 自身是必需的 tuple。
- 错误映射:`400→BadRequest`、`401→Authentication`、`403→PermissionDenied`、`404→NotFound`、`422→UnprocessableEntity`、`429→RateLimit`(额外带 `retry_after_ms`)、`5xx→InternalServer`。`TypeSafeAPIError.__str__` 会附上 `request_id`。

### 5.5 并发与批处理语义

- 官方立场([fan-out](https://docs.typesafe.ai/patterns/fan-out)):**把所有问题放进一个请求**,「所有问题并行评估,增加问题对响应时间通常影响很小」。**不设问题数量上限**(只要满足两个上下文限制)。
- HTTP/1.1 是**默认且推荐**高请求率(并发随连接数扩展);HTTP/2 适合低-中并发想少连接(受代理/防火墙限制时)。
- 官方性能建议:复用 client;重负载下**每线程一个 client**;异步高请求率可用 `httpx-aiohttp`。
- 官方 cookbook 用**线程池**做并发(例如 re-ranking 的 1,200 次独立调用);Twitch 过滤 cookbook 明确「最多 8 条消息并发」。

---

## 6. 兼容语义:线兼容 vs 语义兼容

### 6.1 逐项对齐表

| 维度 | 线兼容(schema 可枚举) | 语义兼容(可对齐的目标值) | 证据 / 备注 |
|---|:---:|:---:|---|
| 端点路径与方法 | ✅ | — | `POST /v1/systemone`,`GET /v1/models`([constants.py](https://docs.typesafe.ai/sdk/python/api/constants)) |
| Bearer 认证 | ✅ | — | SDK 强制非空 key;本地服务可用任意占位 key |
| 请求 schema(state/model/questions) | ✅ | — | questions 的 key 不进模型 |
| Noul/Choice/Score 请求字段 | ✅ | — | |
| 答案字段集合 | ✅ | — | Choice/Score 的 `confidence` **必填无默认** |
| `choice` = argmax(probabilities) | ✅(有要求) | ⚠️ | SDK **不校验**;Kev 自己的测试才断言 |
| `probabilities` 求和≈1 | ✅ | ⚠️ | 文档「approximately 1」;Kev 取 4 位小数使 `|Σ−1|<0.02`;Jev 实测 2 位小数 |
| `score` = Σ i·pᵢ | ✅(有要求) | — | 纯派生量 |
| `legend` 原样回显 criteria | ✅ | — | |
| `confidence` 公式 | — | ✅ 可精确对齐 | 官方公式 + `system-one-adapter` 参考实现;单选项/全零/单级特例都有定义 |
| Noul 无 `confidence` | ✅ | — | 文档建议自算 `\|2p−1\|` |
| 错误码集合与语义 | ✅ | — | 400/401/413/415/422/429/529 |
| `usage.input_tokens` / `output_tokens` | ✅(字段) | ❌ | Jev:`output_tokens` 是**计费数字**(4 + 15×答案数 + Σ\|qid\|);Kev:`len(tokenize(json(answers)))`。**不可比** |
| `latency_ms` | ❌ **非 TypeSafe 字段** | ❌ | TypeSafe 官方文档全文中出现 **0 次**;Kev 扩展 |
| `x-typesafe-request-id` 响应头 | ✅ | — | 所有 TypeSafe 客户端都会读 |
| `truncated` / `usage.state_tokens*` | ❌ 扩展 | ❌ | Kev 独有;SDK 忽略未知字段 |
| **默认温度** | ❌ 无此参数 | ❌ **无可对齐目标** | Jev 不暴露温度;Kev 每检查点带拟合 T(4B 2.41 / 0.8B 2.35 / 9B 2.19 / 27B 1.32),eval 时应用,`KEV_TEMPERATURE=1.0` 关闭 |
| **校准质量** | ❌ | ❌ | 只能说「达到某 ECE」;Jev 1.13 在 1,200 题 MMLU 样本上 10-bin ECE = 0.0313 |
| **选项顺序敏感性** | ❌ | ❌ | Jev 实测 0.84–0.89 → 0.93–0.96;只能作为**评估项**,不能作为对齐目标 |
| **并发/批处理** | ❌ | ⚠️ | Jev:任意问题数、并行、无批参数;Kev:`MAX_BATCH=64`、`rows_per_pass` 16,384 token 预算,答案与批切分无关 |
| **确定性** | ❌ | ❌ | Jev 实测**非确定**;Kev 声明答案不依赖 batch 切分 |
| 上下文限制 | ⚠️ 近似 | ⚠️ | 数字不同,见 3.2 |
| Long-context header | ✅(可忽略) | ❌ | `TypeSafe-Danger-Zone-Long-Context: true` → 262k;Kev 忽略 |
| **Box 题型** | ❌ | ❌ | 官方文档与 SDK 有;**Kev 未实现** |
| 图片 / assets | ❌ | ❌ | 需要 `multipart/form-data`,part 名 `request` + `asset-N` |
| 速率限制行为 | ✅(429/529 + backoff) | ⚠️ | SDK 默认重试 408/429/5xx |

### 6.2 SDK 客户端强依赖的行为(兼容层必须满足,否则客户端直接失败)

按「不做就坏」排序:

1. **`POST {base_url}/v1/systemone` 与 `GET {base_url}/v1/models` 的精确路径** —— SDK 没有路径配置。
2. **每个响应都带 `x-typesafe-request-id`**,且非空(Kev 用 middleware 无条件补上,并回显请求方提供的值)。
3. **`Bearer` 认证** —— SDK 总是发送;服务端设了 key 就必须用 `hmac.compare_digest` 之类比较,未设则必须接受任意 key(Kev 默认开放)。
4. **答案必须携带 `type`**;`answers` 里任一元素缺 `type` 或不是 dict → 整个响应校验失败(`validation_error(response, f"answers.{name}.type")`)。
5. **Choice answer 必须有 `choice`、`confidence`、`probabilities` 三者**(都无默认值)。
6. **Score answer 必须有 `score`、`confidence`、`legend`、`probabilities` 四者**;`legend`/`probabilities` 的键是**字符串化的整数**。
7. **响应必须有 `model`(str)与 `usage`(对象)**;`usage.input_tokens`/`output_tokens` 是 `int | None`,可缺省或为 `null`。
8. **`/v1/models` 每个 card 必须有 `name`、`description`、`release_date`**,缺一即抛错 → 想被 `models.list()` 解析,不能返回空 description。
9. **422 不被重试** → 服务端把校验错误映射到 422 时,客户端会立刻抛出(带 `request_id`)。
10. **`Content-Type`**:无 asset 时为 `application/json`;有 asset 时必须是客户端构造的 `multipart/form-data`(服务端要能解析 part `request` + `asset-N`)。

**SDK 不做校验的地方**(兼容层可以「自由」但也意味着容错责任在自己):`probabilities` 是否求和为 1、`choice` 是否等于 argmax、`score` 是否等于加权平均、`legend` 键与 `probabilities` 键是否一致。

### 6.3 建议的 SDK 验收方式(可直接支撑 #13 的「SDK 验收」章节)

Kev 已经沉淀出一套可复用的模式,值得作为 Stim 的模板(`jaredpalmer/kev:tests/test_api.py`,以 `pytest -m server` + `KEV_BASE_URL` 运行):

1. **官方示例请求往返**:把 TypeSafe 文档里的 example request 逐条 POST 到本地服务,断言 shape。
2. **真实 SDK 端到端**(不是手写 HTTP):
   - 同步 `TypeSafeClient(api_key="local", base_url=<local>, model=<name>)` → `resp.nouls[...]`、`resp.choices[...]`、`resp.scores[...]`,`assert resp.request_id and resp.usage.input_tokens > 0`。
   - 异步 `AsyncTypeSafeClient` 同样跑一遍。
   - `client.models.list().models` 能解析,且包含 SDK 默认模型名 `jev-latest`。
3. **派生量自洽**:`choice == max(probabilities, key=...)`;`|Σp − 1| ≤ 0.03`;`0 ≤ confidence ≤ 1`;`score ≈ Σ i·pᵢ`(abs_tol 0.05);`legend` 与 criteria 一致。
4. **错误路径**:过长的 state → `TypeSafeUnprocessableEntityError`,断言 `status == 422`、消息含服务端 detail、`request_id` 非空,且**没有发生重试**。
5. **隔离性**:同一 state 单独问某题 vs 与兄弟题同请求,概率差 ≤ 0.011。
6. **forward-compat 反证**:服务端带扩展字段(`latency_ms`、`truncated`、`usage.state_tokens`)时 SDK 仍能解析。

> 第 4 条依赖「服务端把超限映射为 422」这一选择——这正是 Kev 的偏离点(官方 Jev 用 `400` + `max_tokens_exceeded`)。**若目标是最大兼容,超限应该用 `400`;若目标是复用 SDK 的现成错误处理,422 也已被 SDK 正确映射。这是 #13 需要明确的一处设计取舍。**

### 6.4 Jev / Kev / AutoJev 三方对照

| | Jev 1.13 | Kev | AutoJev-27B |
|---|---|---|---|
| Backbone | 未知(疑 MoE) | Qwen3.5 / Qwen3.8 Base(密集) | **Qwen3.8-27B** |
| 适配 | 未公开 | LoRA rank 16 + PointerHead | **full-weight SFT**(73,000 例,286 updates,单 H200) |
| Readout | 直接概率读出(机制未公开) | 双线性 pointer head | 「一次 forward pass per question」 |
| 温度 | 不暴露 | 每检查点标量,默认应用 | 「calibration fits a scalar temperature separately」 |
| 许可证 | 托管 | Apache-2.0 | 权重 Apache-2.0 / 代码 MIT |
| 报告指标 | MMLU-Pro 84.6% | 见各模型卡 | 84.60% / ECE 0.04282 / Brier 0.22027(vs Jev 82.79% / 0.05274 / 0.25400,自报) |

AutoJev 的「一次 forward pass per question」与 Jev 的「state 只算一次」**不是同一件事**,是另一个独立实现选择。

---

## 7. 不可达来源(显式清单)

| 来源 | 状态 | 处理方式 |
|---|---|---|
| `https://api.typesafe.ai/*`(含 `openapi.json`) | **HTTP 451 地区封锁** | 线协议全部改由官方文档 + 官方 SDK 源码 + 官方参考适配器源码确立;未做活体探测 |
| `huggingface.co` | 不可达 | 用 `hf-mirror.com` API 与 `/raw/main/` 取模型卡 |
| `raw.githubusercontent.com` | 不可达 | 用本地浅克隆 `/tmp/kev-src` 与 `gh api` |
| `registry.npmjs.org` | 不可达 | **未能读取 `@typesafe-ai/sdk`(JavaScript SDK)的实际包内容**;JS 端行为仅来自官方文档(其 `TypeSafeClient` 类文档与 Python 侧语义一致,但未逐字段核对) |
| Vercel AI Gateway / Jev 真实端点 | 无可用 API key | 未做任何真实 Jev 调用;标注为「未验证」 |
| TypeSafe 的 `system-one-adapter` 源码仓库 | 未定位到公开 Git 仓库(仅有 PyPI 分发) | 直接解包 wheel 读源码,等同于官方发布物 |

---

## 8. 置信度与未决问题

### 高置信(多来源交叉)

- 线协议端点、请求/响应 schema、错误码:官方文档 + 官方 SDK 生成 schema 双向一致。
- `confidence` / `score` 的派生公式:官方文档公式 + 官方 `system-one-adapter` 参考实现**逐行一致**。
- SDK 默认值(base_url、model、timeout、重试集合、连接池):直接读已发布 wheel 的源码。
- SDK 的强依赖行为清单(6.2):同样是读源码得出。
- Kev 的架构与偏离项:直接读源码。

### 中置信(单一来源或推断)

- Jev 的「共享前缀 + 问题隔离」:官方陈述 + 第三方实证一致,但**具体实现机制未公开**。
- 2.2 的全部量化数据:来自单一第三方调查(10k 次调用、含随机区块与对照),方法透明、证据包公开,但**未经独立复现**。
- Jev 的上下文限制语义(请求限制只计一次 state):单一来源。

### 低置信(纯推断,不应作为设计前提)

- Jev 使用稀疏 MoE:作者本人列为最不确定项。
- Jev 的 readout 是 slot head 还是 pointer:200 选项实验仅「偏向」pointer。
- Jev 的 backbone 身份与 tokenizer:公开证据无法确定。

### 未决问题(交给 #13 / #9)

1. **超限错误应映射为 `400` + `max_tokens_exceeded` 还是 `422`?** 前者是 Jev 的真实行为、最大化线兼容;后者能被 SDK 正确映射为 `TypeSafeUnprocessableEntityError` 且不被重试。需产品级取舍。
2. **Score 的级别数上限**:文档说 ≤10 且「至少 2」,SDK 只强制 ≥1,Kev 允许到 255。对齐哪个?
3. **是否需要实现 Box 题型?** 官方 SDK 与文档都有,Kev 没有。若 Stim 目标是「API 与 System One 完全兼容」,这是一个已知缺口。
4. **概率精度**:Jev 实测 2 位小数,Kev 用 4 位。取几位需要在「|Σ−1| 容差」与「信息量」之间定。
5. **`output_tokens` 怎么算?** 两种实现语义完全不同;若客户端会用这个字段做成本估算或断言,需要明确复制 Jev 的计费公式还是自定。
6. **是否要实现 `TypeSafe-Danger-Zone-Long-Context`?** 不实现则 262k 请求路径缺失;实现则需相应算力。
7. **默认定量目标缺失**:Jev 不暴露温度,也不承诺校准水平。Stim 能对齐的只有「公式」与「字段」,不能对齐「校准」。是否需要一个自定的 ECE/Brier 门槛作为验收,需在 #13 决定。
8. **JS SDK 未核对**:因 npm registry 不可达,`@typesafe-ai/sdk` 的字段级行为未验证。若 Stim 需要支持 JS 客户端,需补一轮调查。
