# NDEL Fuzzy Value Resolution Specification

Version: 0.1.0  
Status: Draft Specification

## 1. Overview

Fuzzy value resolution is the core innovation of NDEL. It enables natural language values to be interpreted into precise computational expressions based on context, domain knowledge, and learned patterns.

## 2. Fuzzy Value Types

### 2.1 Classification

Fuzzy values fall into these categories:

| Category | Description | Examples |
|----------|-------------|----------|
| **Qualitative** | Descriptive qualities | `"high"`, `"excellent"`, `"poor"` |
| **Relative** | Comparative terms | `"young"`, `"recent"`, `"large"` |
| **Approximate** | Imprecise quantities | `"about 100"`, `"few"`, `"several"` |
| **Compound** | Multi-part descriptions | `"young talented striker"` |
| **Behavioral** | Action patterns | `"improving"`, `"declining"`, `"stable"` |
| **Domain-Specific** | Context-dependent terms | `"clinical"`, `"world-class"`, `"veteran"` |

### 2.2 Identification Rules

A value is identified as fuzzy when:

1. **Quoted string in comparison**: `age < "young"`
2. **Argument to fuzzy operator**: `player is "promising"`
3. **Explicit fuzzy marker**: `fuzzy"approximate value"`
4. **Pattern matching context**: `trend shows "improvement"`
5. **Natural language phrase**: `"better than average"`

## 3. Resolution Pipeline

### 3.1 Pipeline Stages
```
Input Expression
      ↓
[1. Lexical Analysis]
      ↓
[2. Context Extraction]
      ↓
[3. Domain Resolution]
      ↓
[4. Confidence Scoring]
      ↓
[5. Alternative Generation]
      ↓
Resolved Expression
```

### 3.2 Stage Details

#### Stage 1: Lexical Analysis
```python
def lexical_analysis(fuzzy_value: str) -> TokenizedValue:
    """
    Break down fuzzy value into components
    
    Input: "young promising striker"
    Output: {
        tokens: ["young", "promising", "striker"],
        modifiers: ["young", "promising"],
        head: "striker",
        sentiment: "positive"
    }
    """
```

#### Stage 2: Context Extraction
```python
def extract_context(fuzzy_node: ASTNode) -> ResolutionContext:
    """
    Gather context for resolution
    
    Returns:
    - field_name: Variable being compared
    - field_type: Expected data type
    - operator: Comparison operator
    - domain: Active domain context
    - historical: Previous resolutions
    """
```

#### Stage 3: Domain Resolution
```python
def domain_resolution(value: str, context: ResolutionContext) -> Resolution:
    """
    Apply domain-specific rules
    
    Domain rules are checked in order:
    1. User-defined mappings
    2. Domain-specific rules
    3. Learned patterns
    4. LLM fallback
    """
```

#### Stage 4: Confidence Scoring
```python
def calculate_confidence(resolution: Resolution, context: Context) -> float:
    """
    Calculate confidence score based on:
    - Rule match strength (0.0-0.4)
    - Context relevance (0.0-0.3)
    - Historical accuracy (0.0-0.2)
    - Domain expertise (0.0-0.1)
    
    Total: 0.0-1.0
    """
```

#### Stage 5: Alternative Generation
```python
def generate_alternatives(primary: Resolution, context: Context) -> List[Resolution]:
    """
    Generate alternative interpretations
    
    Returns top 3 alternatives with lower confidence scores
    """
```

## 4. Resolution Rules

### 4.1 Rule Priority

Resolution rules are applied in this priority order:

1. **User Overrides** - Explicit user-defined mappings
2. **Domain Rules** - Domain-specific interpretations
3. **Learned Patterns** - ML-based resolutions
4. **Common Patterns** - Cross-domain patterns
5. **LLM Fallback** - AI interpretation

### 4.2 Rule Specification Format
```yaml
rule:
  pattern: "young"
  context:
    field_type: "age"
    domain: "soccer"
  resolution:
    expression: "< 23"
    confidence: 0.85
    reasoning: "U-23 is standard youth category in soccer"
  alternatives:
    - expression: "< 21"
      confidence: 0.70
      reasoning: "U-21 is also common youth category"
```

## 5. Domain-Specific Resolution

### 5.1 Soccer Domain Examples

| Fuzzy Term | Context | Resolution | Confidence |
|------------|---------|------------|------------|
| `"young"` | age | `< 23` | 0.85 |
| `"veteran"` | age | `>= 30` | 0.80 |
| `"clinical"` | finishing | `goals/shots > 0.20` | 0.75 |
| `"pacey"` | speed | `speed_percentile > 85` | 0.80 |
| `"tall"` | height | `> 185` | 0.75 |
| `"recent"` | time | `< 30 days` | 0.85 |
| `"good form"` | performance | `avg(last_5_ratings) > 7.5` | 0.80 |

### 5.2 Finance Domain Examples

| Fuzzy Term | Context | Resolution | Confidence |
|------------|---------|------------|------------|
| `"volatile"` | stock | `beta > 1.5` | 0.85 |
| `"undervalued"` | valuation | `pe_ratio < sector_avg * 0.8` | 0.75 |
| `"high volume"` | trading | `volume > avg_volume * 2` | 0.80 |
| `"bullish"` | trend | `sma_50 > sma_200` | 0.85 |
| `"risky"` | investment | `risk_score > 7` | 0.75 |

### 5.3 Healthcare Domain Examples

| Fuzzy Term | Context | Resolution | Confidence |
|------------|---------|------------|------------|
| `"elevated"` | blood_pressure | `> 140/90` | 0.90 |
| `"recent"` | diagnosis | `< 3 months` | 0.85 |
| `"chronic"` | condition | `duration > 6 months` | 0.90 |
| `"severe"` | pain_scale | `>= 8` | 0.85 |
| `"stable"` | vitals | `variation < 5%` | 0.80 |

## 6. Confidence Model

### 6.1 Confidence Components
```python
class ConfidenceModel:
    def calculate(self, resolution: Resolution) -> float:
        components = {
            'rule_match': self.rule_match_score(),      # 0-40%
            'context_fit': self.context_fit_score(),    # 0-30%
            'historical': self.historical_accuracy(),    # 0-20%
            'domain_expertise': self.domain_score()      # 0-10%
        }
        return sum(components.values())
```

### 6.2 Confidence Thresholds

| Level | Range | Action |
|-------|-------|--------|
| **High** | 0.85-1.0 | Execute without confirmation |
| **Medium** | 0.65-0.84 | Execute with logging |
| **Low** | 0.45-0.64 | Request confirmation |
| **Uncertain** | < 0.45 | Require manual resolution |

### 6.3 Confidence Propagation
```python
# AND operation: minimum confidence
conf(A && B) = min(conf(A), conf(B))

# OR operation: weighted average
conf(A || B) = (conf(A) + conf(B)) / 2

# NOT operation: preserved
conf(!A) = conf(A)

# Comparison: minimum
conf(A > B) = min(conf(A), conf(B))

# Conditional: selected branch
conf(A ? B : C) = conf(A) * conf(B) if A else conf(A) * conf(C)
```

## 7. Learning System

### 7.1 Learning Triggers

| Event | Action |
|-------|--------|
| **User Correction** | Store mapping with high weight |
| **Successful Execution** | Increase confidence by 0.05 |
| **Failed Execution** | Decrease confidence by 0.10 |
| **Alternative Selected** | Swap primary/alternative |
| **Repeated Use** | Increase confidence by 0.02 |

### 7.2 Learning Storage
```sql
CREATE TABLE resolution_history (
    id UUID PRIMARY KEY,
    fuzzy_value TEXT,
    context JSONB,
    resolution TEXT,
    confidence FLOAT,
    outcome TEXT,
    user_feedback TEXT,
    timestamp TIMESTAMP
);

CREATE TABLE learned_patterns (
    pattern TEXT,
    domain TEXT,
    context_pattern JSONB,
    resolution_template TEXT,
    success_rate FLOAT,
    use_count INTEGER
);
```

## 8. Resolution Examples

### 8.1 Simple Resolution
```ndel
Input: age < "young"
Context: {field: "age", domain: "soccer", type: "int"}

Resolution Steps:
1. Identify "young" as fuzzy value
2. Extract context: comparing age field
3. Apply soccer domain rule: "young" → "< 23"
4. Calculate confidence: 0.85
5. Generate alternatives: ["< 21" (0.70), "< 25" (0.65)]

Output: age < 23  [confidence: 0.85]
```

### 8.2 Compound Resolution
```ndel
Input: player is "promising young striker"
Context: {domain: "soccer", table: "players"}

Resolution Steps:
1. Tokenize: ["promising", "young", "striker"]
2. Resolve components:
   - "striker" → position IN ('ST', 'CF')
   - "young" → age < 23
   - "promising" → potential > 75
3. Combine: position IN ('ST', 'CF') AND age < 23 AND potential > 75
4. Calculate confidence: 0.75
5. Generate alternatives

Output: position IN ('ST', 'CF') AND age < 23 AND potential > 75  [confidence: 0.75]
```

### 8.3 Contextual Resolution
```ndel
Input: performance is "good"
Context: {recent_scores: [6.5, 7.0, 7.5, 8.0, 8.5]}

Resolution Steps:
1. Identify context: recent upward trend
2. Resolve "good" with trend context
3. Generate: avg(last_5) > 7.0 AND trend = 'increasing'
4. Calculate confidence: 0.80

Output: avg(last_5) > 7.0 AND trend = 'increasing'  [confidence: 0.80]
```

## 9. Error Handling

### 9.1 Resolution Failures

| Error | Cause | Recovery |
|-------|-------|----------|
| `UnresolvableError` | No matching rules | Request user input |
| `AmbiguousError` | Multiple equal confidence | Show alternatives |
| `ContextError` | Missing context | Use default context |
| `DomainError` | Unknown domain | Use general rules |
| `ConfidenceError` | Below threshold | Request confirmation |

### 9.2 Fallback Strategy
```python
def fallback_resolution(fuzzy_value: str, context: Context) -> Resolution:
    strategies = [
        self.try_partial_match,
        self.try_synonym_expansion,
        self.try_llm_interpretation,
        self.request_user_input
    ]
    
    for strategy in strategies:
        result = strategy(fuzzy_value, context)
        if result.confidence > 0.45:
            return result
    
    raise UnresolvableError(f"Cannot resolve: {fuzzy_value}")
```

## 10. Implementation Guidelines

### 10.1 Caching Strategy

- Cache resolved values for session
- Invalidate on context change
- Store high-confidence resolutions permanently
- LRU eviction for low-confidence resolutions

### 10.2 Performance Optimization

- Pre-compile common patterns
- Index resolution rules by context
- Batch similar resolutions
- Async LLM calls for fallback

### 10.3 Extensibility

- Plugin architecture for domains
- User-defined resolution rules
- Custom confidence models
- External resolution services

## 11. Future Enhancements

- **Multi-language Support**: Fuzzy values in different languages
- **Temporal Learning**: Time-based confidence adjustment
- **Collaborative Learning**: Share resolutions across users
- **Explanation Generation**: Natural language explanations
- **Contradiction Detection**: Identify conflicting resolutions
- **Fuzzy Composition**: Complex fuzzy expression trees

---

*This document specifies fuzzy value resolution for NDEL version 0.1.0.*
