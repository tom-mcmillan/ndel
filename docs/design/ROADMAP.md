# NDEL Roadmap: From Concept to Executable Language

## Current State
✅ Parser works (can parse NDEL syntax)
❌ No interpreter (can't execute)
❌ No LLM integration (core feature missing)
❌ No fuzzy parameter resolution

## The Build Plan

### Phase 0: Foundation (Days 1-3)
**Goal:** Get a working interpreter that can execute deterministic NDEL

**Tasks:**
1. ✅ Parser exists - extend for new syntax
2. ⬜ Build basic interpreter
3. ⬜ Add type system (int, string, bool, etc.)
4. ⬜ Support basic operations (+, -, &&, ||, etc.)
5. ⬜ Test with simple expressions

**Output:** Can execute `age > 25 && status == "active"`

---

### Phase 1: LLM Primitives (Days 4-7)
**Goal:** Add LLM operations as first-class language features

#### Step 1.1: Extend Parser for LLM Syntax
Add new token types:
```python
# In parser.py, add to TokenType enum:
LLM = auto()           # "llm" keyword
COMPLETE = auto()      # "complete" keyword
ANALYZE = auto()       # "analyze" keyword
WITH = auto()          # "with" keyword (for parameters)
REQUIRE = auto()       # "require" keyword (for quality gates)
```

Add new AST nodes:
```python
@dataclass
class LLMCall(ASTNode):
    """Represents an LLM operation"""
    operation: str  # "complete", "analyze", etc.
    prompt: ASTNode
    parameters: Dict[str, ASTNode]  # Fuzzy parameters
    requirements: List[ASTNode]  # Quality gates

@dataclass
class WithParameter(ASTNode):
    """Represents a fuzzy parameter: with creativity: "high" """
    name: str
    value: str  # Fuzzy value to be resolved

@dataclass
class Requirement(ASTNode):
    """Represents a quality gate: require confidence > 0.8"""
    condition: ASTNode
```

#### Step 1.2: Parse LLM Calls
```python
# In Parser class, add method:
def parse_llm_call(self) -> LLMCall:
    """
    Parse: llm.complete(prompt) with param: "value" require condition
    """
    self.consume(TokenType.LLM)
    self.consume(TokenType.DOT)

    operation = self.consume(TokenType.IDENTIFIER).value

    self.consume(TokenType.LPAREN)
    prompt = self.parse_expression()
    self.consume(TokenType.RPAREN)

    # Parse 'with' parameters
    parameters = {}
    while self.match(TokenType.WITH):
        self.consume(TokenType.WITH)
        param_name = self.consume(TokenType.IDENTIFIER).value
        self.consume(TokenType.COLON)
        param_value = self.parse_expression()
        parameters[param_name] = param_value

    # Parse 'require' conditions
    requirements = []
    while self.match(TokenType.REQUIRE):
        self.consume(TokenType.REQUIRE)
        condition = self.parse_expression()
        requirements.append(condition)

    return LLMCall(operation, prompt, parameters, requirements)
```

#### Step 1.3: Create LLM Runtime
```python
# Create: reference/python/llm_runtime.py

from typing import Dict, Any, Optional
from dataclasses import dataclass
import anthropic
import openai

@dataclass
class LLMResponse:
    """Response from an LLM call"""
    value: str
    confidence: float
    model: str
    tokens_used: int
    latency_ms: float

class LLMRuntime:
    """Execute LLM operations"""

    def __init__(self, provider: str = "anthropic", model: str = None):
        self.provider = provider
        self.model = model or self._default_model()
        self.client = self._init_client()

    def complete(self,
                 prompt: str,
                 parameters: Dict[str, Any] = None) -> LLMResponse:
        """
        Execute llm.complete(prompt) with parameters

        Args:
            prompt: The prompt to complete
            parameters: Resolved fuzzy parameters (e.g., {"temperature": 0.7})

        Returns:
            LLMResponse with result and metadata
        """
        import time
        start = time.time()

        # Resolve parameters to actual LLM settings
        settings = self._resolve_settings(parameters or {})

        # Call LLM
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=settings.get('max_tokens', 1024),
                temperature=settings.get('temperature', 0.7),
                messages=[{"role": "user", "content": prompt}]
            )

            result = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens

        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=settings.get('temperature', 0.7),
                max_tokens=settings.get('max_tokens', 1024)
            )

            result = response.choices[0].message.content
            tokens = response.usage.total_tokens

        latency = (time.time() - start) * 1000

        # Estimate confidence (simplified for now)
        confidence = self._estimate_confidence(result, response)

        return LLMResponse(
            value=result,
            confidence=confidence,
            model=self.model,
            tokens_used=tokens,
            latency_ms=latency
        )

    def analyze(self, data: str, parameters: Dict[str, Any] = None) -> LLMResponse:
        """Execute llm.analyze(data)"""
        prompt = f"Analyze the following:\n\n{data}"
        return self.complete(prompt, parameters)

    def summarize(self, text: str, parameters: Dict[str, Any] = None) -> LLMResponse:
        """Execute llm.summarize(text)"""
        prompt = f"Summarize the following:\n\n{text}"
        return self.complete(prompt, parameters)

    def _resolve_settings(self, parameters: Dict[str, Any]) -> Dict[str, float]:
        """Convert resolved fuzzy parameters to LLM settings"""
        # This is where fuzzy resolver output becomes actual LLM params
        return parameters

    def _estimate_confidence(self, result: str, response: Any) -> float:
        """Estimate confidence in the response"""
        # Simplified: Use response metadata
        # Future: More sophisticated confidence estimation
        return 0.85  # Placeholder

    def _default_model(self) -> str:
        if self.provider == "anthropic":
            return "claude-3-5-sonnet-20241022"
        elif self.provider == "openai":
            return "gpt-4"
        return "claude-3-5-sonnet-20241022"

    def _init_client(self):
        if self.provider == "anthropic":
            return anthropic.Anthropic()
        elif self.provider == "openai":
            return openai.OpenAI()
```

#### Step 1.4: Extend Interpreter
```python
# In interpreter.py, add to NDELInterpreter class:

from llm_runtime import LLMRuntime

class NDELInterpreter:
    def __init__(self, domain: str = "general", llm_provider: str = "anthropic"):
        self.domain = domain
        self.llm_runtime = LLMRuntime(provider=llm_provider)
        self.fuzzy_resolver = FuzzyResolver(domain)
        self.variables = {}

    def _evaluate_node(self, node: ASTNode, context: Dict) -> Any:
        """Evaluate a single AST node"""

        # ... existing code for BinaryOp, Literal, etc. ...

        if isinstance(node, LLMCall):
            return self._evaluate_llm_call(node, context)

        # ... rest of cases ...

    def _evaluate_llm_call(self, node: LLMCall, context: Dict) -> LLMResponse:
        """Execute an LLM operation"""

        # 1. Evaluate prompt
        prompt = self._evaluate_node(node.prompt, context)
        if not isinstance(prompt, str):
            prompt = str(prompt)

        # 2. Resolve fuzzy parameters
        resolved_params = {}
        for param_name, param_value_node in node.parameters.items():
            # Get fuzzy value
            fuzzy_value = self._evaluate_node(param_value_node, context)

            # Resolve it
            if isinstance(fuzzy_value, str):
                resolution = self.fuzzy_resolver.resolve(
                    fuzzy_value,
                    {'parameter': param_name, 'domain': self.domain}
                )
                resolved_params[param_name] = resolution.value
            else:
                resolved_params[param_name] = fuzzy_value

        # 3. Execute LLM operation
        if node.operation == "complete":
            response = self.llm_runtime.complete(prompt, resolved_params)
        elif node.operation == "analyze":
            response = self.llm_runtime.analyze(prompt, resolved_params)
        elif node.operation == "summarize":
            response = self.llm_runtime.summarize(prompt, resolved_params)
        else:
            raise RuntimeError(f"Unknown LLM operation: {node.operation}")

        # 4. Check requirements (quality gates)
        for requirement in node.requirements:
            # Temporarily add response to context
            context['_response'] = response

            satisfied = self._evaluate_node(requirement, context)
            if not satisfied:
                raise QualityGateError(f"Requirement not met: {requirement}")

        return response
```

---

### Phase 2: Fuzzy Parameter Resolution (Days 8-10)
**Goal:** Map fuzzy parameters to actual LLM settings

#### Step 2.1: Create Fuzzy Parameter Mappings
```python
# reference/python/fuzzy_resolver.py

class FuzzyResolver:
    """Resolve fuzzy parameters to concrete values"""

    # Universal parameter mappings (cross-domain)
    UNIVERSAL_MAPPINGS = {
        'creativity': {
            'very_low': {'temperature': 0.1, 'top_p': 0.5},
            'low': {'temperature': 0.3, 'top_p': 0.7},
            'moderate': {'temperature': 0.7, 'top_p': 0.9},
            'high': {'temperature': 0.9, 'top_p': 0.95},
            'very_high': {'temperature': 1.0, 'top_p': 1.0}
        },

        'reasoning': {
            'fast': {'max_tokens': 512, 'thinking_budget': 'low'},
            'moderate': {'max_tokens': 1024, 'thinking_budget': 'medium'},
            'careful': {'max_tokens': 2048, 'thinking_budget': 'high'},
            'deep': {'max_tokens': 4096, 'thinking_budget': 'extended'}
        },

        'tone': {
            'professional': {'system_prompt': 'You are a professional assistant...'},
            'casual': {'system_prompt': 'You are a friendly, casual assistant...'},
            'empathetic': {'system_prompt': 'You are an empathetic, caring assistant...'},
            'technical': {'system_prompt': 'You are a technical expert...'}
        },

        'verbosity': {
            'concise': {'max_tokens': 256, 'instruction': 'Be very concise.'},
            'moderate': {'max_tokens': 512, 'instruction': 'Be reasonably detailed.'},
            'detailed': {'max_tokens': 1024, 'instruction': 'Be comprehensive.'},
            'comprehensive': {'max_tokens': 2048, 'instruction': 'Be exhaustive.'}
        },

        'safety': {
            'permissive': {'guardrails': 'minimal'},
            'balanced': {'guardrails': 'standard'},
            'strict': {'guardrails': 'strict'},
            'paranoid': {'guardrails': 'maximum'}
        }
    }

    def resolve(self, fuzzy_value: str, context: Dict[str, Any]) -> Resolution:
        """
        Resolve a fuzzy parameter to concrete LLM settings

        Args:
            fuzzy_value: The fuzzy value (e.g., "high", "careful")
            context: Context including parameter name, domain, etc.

        Returns:
            Resolution with concrete value and confidence
        """
        param_name = context.get('parameter', '')

        # Try universal mappings first
        if param_name in self.UNIVERSAL_MAPPINGS:
            mapping = self.UNIVERSAL_MAPPINGS[param_name]

            # Normalize fuzzy value
            normalized = fuzzy_value.lower().replace(' ', '_')

            if normalized in mapping:
                return Resolution(
                    value=mapping[normalized],
                    confidence=0.95,
                    reasoning=f"Universal mapping for {param_name}={fuzzy_value}"
                )

        # Try domain-specific mappings
        # ... (future work)

        # Try semantic matching
        # ... (future work)

        raise ResolutionError(f"Cannot resolve {param_name}={fuzzy_value}")
```

---

### Phase 3: Working Examples (Days 11-14)
**Goal:** Demonstrate the language works end-to-end

#### Example 1: Simple Completion
```python
# examples/01_simple_completion.ndel

result = llm.complete("What is the capital of France?")
  with creativity: "low"
  with verbosity: "concise"

print(result)
```

Run it:
```bash
$ python -m ndel examples/01_simple_completion.ndel

Paris
(confidence: 0.95, latency: 234ms, tokens: 15)
```

#### Example 2: Quality Gates
```python
# examples/02_quality_gates.ndel

response = llm.complete("Write a haiku about programming")
  with creativity: "high"
  require response.confidence > 0.8
  require response.format == "haiku"

print(response)
```

#### Example 3: Multi-Step Pipeline
```python
# examples/03_pipeline.ndel

@domain("code_generation")

task review_code:
  input: code

  # Step 1: Analyze
  analysis = llm.analyze(code)
    with focus: "bugs and improvements"
    with reasoning: "careful"

  # Step 2: Generate suggestions
  if analysis.issues_found > 0:
    suggestions = llm.complete(
      "Suggest fixes for: {analysis.issues}"
    )
      with creativity: "low"
      with safety: "high"

    return {analysis, suggestions}

  return {analysis, message: "Code looks good!"}
```

---

### Phase 4: Core Language Features (Days 15-21)
**Goal:** Add essential language features for real use

1. **Variables and Assignment**
   ```ndel
   result = llm.complete(prompt)
   processed = transform(result.value)
   ```

2. **Control Flow**
   ```ndel
   if result.confidence < 0.7:
     result = retry_with_higher_quality()
   ```

3. **Functions**
   ```ndel
   function analyze_sentiment(text):
     return llm.analyze(text)
       with focus: "sentiment"
       with reasoning: "fast"
   ```

4. **Error Handling**
   ```ndel
   try:
     result = llm.complete(prompt)
       require confidence > 0.9
   catch LowConfidenceError:
     result = request_human_review(prompt)
   ```

---

### Phase 5: Advanced Features (Days 22-30)
**Goal:** Enable sophisticated LLM orchestration

1. **Parallel Execution**
   ```ndel
   results = parallel [
     llm.complete(prompt1),
     llm.complete(prompt2),
     llm.complete(prompt3)
   ]

   best = max(results, by: "confidence")
   ```

2. **Retry Logic**
   ```ndel
   result = llm.complete(prompt)
     with max_retries: 3
     with retry_strategy: "adaptive"
     require confidence > 0.85
   ```

3. **Human-in-the-Loop**
   ```ndel
   if result.confidence < 0.8:
     result = await human_review(result)
       with timeout: "5 minutes"
       with fallback: use_ai_result()
   ```

4. **Learning from Feedback**
   ```ndel
   on_human_feedback(correction):
     learn_pattern(original_input, correction)
     adjust_confidence_model()
   ```

---

## Success Criteria

### Month 1:
- ✅ Can execute simple LLM calls
- ✅ Fuzzy parameter resolution works
- ✅ Quality gates enforce requirements
- ✅ 5+ working examples

### Month 2:
- ✅ Control flow (if/else, loops)
- ✅ Functions and composition
- ✅ Error handling
- ✅ Parallel execution

### Month 3:
- ✅ Human-in-the-loop
- ✅ Learning system
- ✅ Standard library
- ✅ Documentation

### Month 6:
- ✅ Production-ready
- ✅ Multiple LLM providers
- ✅ VS Code extension
- ✅ Community using it

---

## The First Concrete Step

**What to build RIGHT NOW (next 2-3 hours):**

1. Extend parser to recognize `llm.complete()`
2. Create basic LLMRuntime with Anthropic integration
3. Create minimal FuzzyResolver with 5 parameter mappings
4. Wire it together in interpreter
5. Run one working example

**Goal:** Execute this NDEL code successfully:

```ndel
result = llm.complete("What is 2+2?")
  with creativity: "low"

print(result.value)
```

**Expected output:**
```
4

(confidence: 0.95, latency: 187ms, tokens: 12)
```

Once that works, **everything else is iteration**.

Ready to build?
