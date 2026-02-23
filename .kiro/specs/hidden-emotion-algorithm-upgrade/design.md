# Design: 隐藏情绪检测算法升级（Phase 1）

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    HiddenEmotionEngine v2                     │
│                                                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ PM (宏)  │  │ Pm (微)  │  │ PAU (AU) │  ← 三证据源       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                   │
│       │              │              │                         │
│  ┌────▼──────────────▼──────────────▼────┐                   │
│  │        AU 分层验证 (REQ-3)             │  ← 核心AU否决门  │
│  │  核心AU检查 → 辅助AU加权 → Z-score    │                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│  ┌────────────────▼──────────────────────┐                   │
│  │     七元掩饰矩阵查询 (REQ-1)          │  ← M[macro][micro]│
│  │  conflict_coeff = kMaskingMatrix[i][j]│                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│  ┌────────────────▼──────────────────────┐                   │
│  │     D-S 证据合成 (REQ-2)              │                   │
│  │  PM⊕Pm → intermediate                │                   │
│  │  intermediate⊕PAU → final_bpa        │                   │
│  │  K > 0.9 → Yager 修正                │                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│  ┌────────────────▼──────────────────────┐                   │
│  │     自适应阈值 (REQ-6)                │                   │
│  │  τ = α + β·(H(PM) + Var(Pm))         │                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│  ┌────────────────▼──────────────────────┐                   │
│  │     意图推理 (REQ-4)                  │                   │
│  │  Suppression | Masking | Amplification│                   │
│  │  | Congruence                         │                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│  ┌────────────────▼──────────────────────┐                   │
│  │     时序上下文 (REQ-5)                │                   │
│  │  转移概率检查 → 突变标记 → 稳定性评分 │                   │
│  └────────────────┬──────────────────────┘                   │
│                   │                                           │
│                   ▼                                           │
│            HiddenEmotionResult v2                             │
└─────────────────────────────────────────────────────────────┘
```

## 文件结构变更（已实现）

```
hidden_emotion_cpp/
├── include/
│   ├── core/
│   │   └── data_types.h          # ✅ 新增 IntentionType 枚举、扩展 HiddenEmotionResult（10个新字段）
│   └── engines/
│       ├── hidden_emotion_engine.h  # ✅ 重构：8×8 kMaskingMatrix + AuHierarchy + AuBaseline Z-score
│       ├── ds_evidence.h            # ✅ 新增：BPA + DSResult + DSEvidence（combine/combineYager/combineSmart）
│       ├── intention_engine.h       # ✅ 新增：IntentionResult + IntentionEngine（4种模式）
│       └── temporal_context.h       # ✅ 新增：TemporalContext（滑动窗口+转移概率矩阵）
├── src/engines/
│   ├── hidden_emotion_engine.cpp    # ✅ 重构：8步 analyze() 管线
│   ├── ds_evidence.cpp              # ✅ 新增：Dempster合成 + Yager修正 + BPA转换
│   ├── intention_engine.cpp         # ✅ 新增：掩饰>抑制>放大>一致 优先级判定
│   └── temporal_context.cpp         # ✅ 新增：push/dominantEmotion/stabilityScore/isAnomalousTransition
├── src/ui/
│   └── emotion_panel.cpp            # ✅ 更新：HiddenEmotionPanel 新增 D-S/意图/时序/阈值显示
├── config/
│   └── config.json                  # ✅ 更新：hidden 节新增 8 个配置项
└── CMakeLists.txt                   # ✅ 更新：ENGINE_SOURCES 新增 3 个 .cpp
```

## 核心数据结构设计

### 1. 七元掩饰矩阵（编译期常量）

```cpp
// hidden_emotion_engine.h
// 8×8 对称矩阵（含 Confusion），行=宏表情，列=微表情，值=冲突系数
// 索引顺序与 EmotionType 枚举一致：
//   0=Neutral, 1=Happiness, 2=Sadness, 3=Anger,
//   4=Fear, 5=Disgust, 6=Surprise, 7=Confusion
static constexpr size_t kMatrixSize = 8;
static constexpr std::array<std::array<float, kMatrixSize>, kMatrixSize> kMaskingMatrix = {{
    //           Neut   Hap    Sad    Ang    Fear   Disg   Surp   Conf
    /* Neut */ {{ 0.00f, 0.35f, 0.40f, 0.45f, 0.40f, 0.35f, 0.30f, 0.30f }},
    /* Hap  */ {{ 0.35f, 0.00f, 0.95f, 0.90f, 0.85f, 0.80f, 0.40f, 0.50f }},
    /* Sad  */ {{ 0.40f, 0.95f, 0.00f, 0.65f, 0.55f, 0.60f, 0.50f, 0.45f }},
    /* Ang  */ {{ 0.45f, 0.90f, 0.65f, 0.00f, 0.50f, 0.45f, 0.55f, 0.40f }},
    /* Fear */ {{ 0.40f, 0.85f, 0.55f, 0.50f, 0.00f, 0.60f, 0.45f, 0.45f }},
    /* Disg */ {{ 0.35f, 0.80f, 0.60f, 0.45f, 0.60f, 0.00f, 0.70f, 0.40f }},
    /* Surp */ {{ 0.30f, 0.40f, 0.50f, 0.55f, 0.45f, 0.70f, 0.00f, 0.35f }},
    /* Conf */ {{ 0.30f, 0.50f, 0.45f, 0.40f, 0.45f, 0.40f, 0.35f, 0.00f }},
}};
```

### 2. D-S 证据理论模块

```cpp
// ds_evidence.h
namespace hed {

/// 基本概率分配 (Basic Probability Assignment)
using BPA = std::array<float, kEmotionCount>;  // 每个情绪假设的质量

struct DSResult {
    BPA belief;       // 信度函数 Bel(A)
    BPA plausibility; // 似然函数 Pl(A)
    float conflict_k; // 冲突因子 K
};

class DSEvidence {
public:
    /// Dempster 合成规则：m1 ⊕ m2
    static DSResult combine(const BPA& m1, const BPA& m2);

    /// Yager 修正规则（高冲突时使用）
    static DSResult combineYager(const BPA& m1, const BPA& m2);

    /// 自动选择：K > threshold 用 Yager，否则用 Dempster
    static DSResult combineSmart(const BPA& m1, const BPA& m2, float yager_threshold = 0.9f);

    /// 将 EmotionResult 的概率分布转换为 BPA
    static BPA emotionToBPA(const EmotionResult& result);

    /// 将 AU 验证结果转换为 BPA
    static BPA auToBPA(const AuResult& au, EmotionType hypothesis);
};

}  // namespace hed
```

### 3. 意图推理引擎

```cpp
// intention_engine.h
namespace hed {

enum class IntentionType : uint8_t {
    Congruence = 0,   // 一致表达
    Suppression,      // 抑制
    Masking,          // 掩饰
    Amplification,    // 放大
};

struct IntentionResult {
    IntentionType type = IntentionType::Congruence;
    float confidence = 0.0f;
    std::string description;  // 中文描述
};

class IntentionEngine {
public:
    IntentionResult analyze(
        const EmotionResult& macro,
        const EmotionResult& micro,
        float masking_coeff  // 来自七元矩阵的冲突系数
    ) const;

private:
    // 阈值配置
    float suppression_threshold_ = 0.3f;   // 微表情强度低于此值视为抑制
    float masking_threshold_ = 0.7f;       // 冲突系数高于此值视为掩饰
    float amplification_ratio_ = 1.5f;     // 宏/微强度比超过此值视为放大
    float congruence_diff_ = 0.2f;         // 强度差小于此值视为一致
};

}  // namespace hed
```

### 4. 时序上下文管理器

```cpp
// temporal_context.h
namespace hed {

class TemporalContext {
public:
    explicit TemporalContext(size_t window_size = 30);

    /// 推入新的情绪观测
    void push(EmotionType emotion, float confidence);

    /// 获取主导情绪
    EmotionType dominantEmotion() const;

    /// 情绪稳定性评分 [0, 1]
    float stabilityScore() const;

    /// 检查当前情绪是否为异常转移
    bool isAnomalousTransition(EmotionType current) const;

    /// 获取从 from 到 to 的转移概率
    float transitionProbability(EmotionType from, EmotionType to) const;

private:
    size_t window_size_;
    std::deque<std::pair<EmotionType, float>> history_;

    // 转移计数矩阵（运行时统计）
    std::array<std::array<int, 7>, 7> transition_counts_{};
    int total_transitions_ = 0;

    // 异常转移阈值
    float anomaly_threshold_ = 0.1f;
};

}  // namespace hed
```

### 5. 扩展 HiddenEmotionResult

```cpp
// data_types.h 中扩展
struct HiddenEmotionResult {
    // --- 现有字段保留 ---
    EmotionType surface_emotion;
    std::optional<EmotionType> hidden_emotion;
    float surface_prob, hidden_prob;
    float conflict_score;
    bool is_hidden;
    std::vector<AuEvidence> supporting_aus;

    // --- 新增字段 ---
    float masking_coefficient = 0.0f;    // 七元矩阵冲突系数
    float ds_belief = 0.0f;              // D-S 信度
    float ds_plausibility = 0.0f;        // D-S 似然
    float ds_conflict_k = 0.0f;          // D-S 冲突因子

    IntentionType intention = IntentionType::Congruence;
    float intention_confidence = 0.0f;
    std::string intention_description;

    float stability_score = 0.0f;        // 时序稳定性
    bool is_anomalous_transition = false; // 是否异常转移
    float adaptive_threshold = 0.5f;     // 当前自适应阈值
};
```

## AU 分层验证表

```cpp
// 核心AU（必要条件）+ 辅助AU（充分条件）
struct AuHierarchy {
    std::vector<int> core_aus;      // 必须全部激活
    std::vector<int> auxiliary_aus;  // 激活则增强
};

static const std::unordered_map<EmotionType, AuHierarchy> kAuHierarchy = {
    {Happiness,   {{6, 12},           {25}}},
    {Sadness,     {{1, 15},           {4, 17}}},
    {Anger,       {{4, 7},            {5, 23, 24}}},
    {Fear,        {{1, 4, 5},         {2, 20, 26}}},
    {Disgust,     {{9, 10},           {17, 25}}},
    {Surprise,    {{1, 2, 5, 26},     {27}}},
};
```

## Testing Strategy

- 单元测试：D-S 合成规则的数学正确性（已知输入→精确输出）
- 单元测试：七元矩阵对称性、对角线为零
- 单元测试：意图推理的4种模式边界条件
- 集成测试：完整管线从 EmotionResult → HiddenEmotionResult
- 手动测试：摄像头实时运行，观察日志输出的意图判定是否合理

## Performance Considerations

- 七元矩阵用 `constexpr std::array` 编译期构建，查表 O(1)
- D-S 合成涉及 O(N²) 的焦元交集计算，N=7（情绪类别数），实际 49 次乘法，可忽略
- 时序上下文用 `std::deque` 滑动窗口，push/pop O(1)
- 整体新增计算量 < 0.1ms/帧，不影响实时性
