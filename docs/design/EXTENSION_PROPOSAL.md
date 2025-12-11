# NDEL Extension Proposal: Making It Production-Ready

## Executive Summary

NDEL has an elegant parser but is missing the critical fuzzy resolution layer. This proposal outlines a pragmatic path to make NDEL production-ready in 6 months.

## Phase 1: Core Implementation (Months 1-2)

### 1.1 Simple Rule-Based Fuzzy Resolver

**Goal**: 70% accuracy without ML/LLM

```python
# reference/python/fuzzy_resolver.py
import yaml
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Resolution:
    """Result of fuzzy value resolution"""
    expression: str
    confidence: float
    reasoning: str
    alternatives: List['Resolution'] = None

class RuleBasedResolver:
    """Simple pattern matching resolver"""

    def __init__(self, domain_path: str):
        self.domain = self._load_domain(domain_path)
        self.cache = {}

    def resolve(self, fuzzy_value: str, context: dict) -> Resolution:
        # 1. Check cache
        cache_key = (fuzzy_value, context['field_name'], context.get('operator'))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # 2. Try exact match
        resolution = self._exact_match(fuzzy_value, context)
        if resolution:
            self.cache[cache_key] = resolution
            return resolution

        # 3. Try partial match
        resolution = self._partial_match(fuzzy_value, context)
        if resolution:
            self.cache[cache_key] = resolution
            return resolution

        # 4. Try synonym expansion
        resolution = self._synonym_match(fuzzy_value, context)
        if resolution:
            self.cache[cache_key] = resolution
            return resolution

        # 5. Fail
        raise ResolutionError(f"Cannot resolve '{fuzzy_value}' in {context}")

    def _exact_match(self, value: str, context: dict) -> Optional[Resolution]:
        """Try exact pattern match"""
        field = context.get('field_name')
        if field in self.domain['fuzzy_mappings']:
            mappings = self.domain['fuzzy_mappings'][field]
            if value in mappings:
                return Resolution(
                    expression=mappings[value]['expression'],
                    confidence=mappings[value]['confidence'],
                    reasoning=mappings[value].get('reasoning', 'Exact rule match'),
                    alternatives=[
                        Resolution(
                            expression=alt['expression'],
                            confidence=alt['confidence'],
                            reasoning=alt.get('reasoning', '')
                        )
                        for alt in mappings[value].get('alternatives', [])
                    ]
                )
        return None

    def _partial_match(self, value: str, context: dict) -> Optional[Resolution]:
        """Try partial/fuzzy matching"""
        # Implementation: Levenshtein distance, token matching, etc.
        pass

    def _synonym_match(self, value: str, context: dict) -> Optional[Resolution]:
        """Expand synonyms and retry"""
        synonyms = self.domain.get('synonyms', {})
        if value in synonyms:
            for synonym in synonyms[value]:
                resolution = self._exact_match(synonym, context)
                if resolution:
                    resolution.confidence *= 0.9  # Slight penalty for synonym match
                    return resolution
        return None

    def _load_domain(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
```

### 1.2 Domain Definition Format

```yaml
# domains/soccer.yaml
name: soccer
version: 1.0.0
description: "Soccer/football domain for player and team analysis"

fuzzy_mappings:
  age:
    young:
      expression: "< 23"
      confidence: 0.85
      reasoning: "U-23 is standard youth category in international soccer"
      alternatives:
        - expression: "< 21"
          confidence: 0.70
          reasoning: "U-21 is also common youth category"
        - expression: "< 25"
          confidence: 0.65
          reasoning: "Some consider under 25 as young"

    veteran:
      expression: ">= 30"
      confidence: 0.80
      reasoning: "Players 30+ are typically considered veterans"
      alternatives:
        - expression: ">= 32"
          confidence: 0.70
        - expression: ">= 28"
          confidence: 0.60

    prime:
      expression: "BETWEEN 26 AND 30"
      confidence: 0.85
      reasoning: "Peak performance years for most positions"

  position:
    striker:
      expression: "IN ('ST', 'CF')"
      confidence: 0.95
      reasoning: "Standard position codes for strikers"

    midfielder:
      expression: "IN ('CM', 'CDM', 'CAM', 'LM', 'RM')"
      confidence: 0.90

    defender:
      expression: "IN ('CB', 'LB', 'RB', 'LWB', 'RWB')"
      confidence: 0.90

  potential:
    high:
      expression: "> 78"
      confidence: 0.80
      reasoning: "Based on FIFA rating scale (0-100)"
      alternatives:
        - expression: "> 75"
          confidence: 0.75
        - expression: "> 80"
          confidence: 0.70

    world-class:
      expression: "> 88"
      confidence: 0.90

  salary:
    reasonable:
      expression: "< 75000"
      confidence: 0.70
      reasoning: "Context-dependent, varies by league"
      alternatives:
        - expression: "< 50000"
          confidence: 0.65
        - expression: "< 100000"
          confidence: 0.65

# Compound patterns
compound_patterns:
  - pattern: "{quality} {age_term} {position}"
    example: "promising young striker"
    resolution:
      combine: "AND"
      components:
        - field: "potential"
          fuzzy_value: "{quality}"
        - field: "age"
          fuzzy_value: "{age_term}"
        - field: "position"
          fuzzy_value: "{position}"

# Synonyms
synonyms:
  young:
    - youthful
    - junior
    - emerging
  high:
    - strong
    - great
    - excellent
  good:
    - decent
    - solid
    - respectable

# Contextual modifiers
modifiers:
  very:
    multiplier: 1.2
    example: "very high" → adjust threshold by 20%
  somewhat:
    multiplier: 0.8
    example: "somewhat young" → relax threshold
```

### 1.3 Basic Interpreter

```python
# reference/python/interpreter.py
from typing import Any, Dict
from parser import *
from fuzzy_resolver import RuleBasedResolver

class NDELInterpreter:
    """Evaluate NDEL expressions against data"""

    def __init__(self, domain_path: str):
        self.resolver = RuleBasedResolver(domain_path)
        self.confidence_stack = []

    def evaluate(self, ast: List[ASTNode], data: Dict[str, Any]) -> Dict:
        """
        Evaluate AST against data

        Returns:
            {
                'result': bool,
                'confidence': float,
                'resolved_expression': str,
                'trace': List[str]
            }
        """
        domain = None
        expressions = []

        for node in ast:
            if isinstance(node, DomainDeclaration):
                domain = node.domain
            else:
                resolved = self._resolve_fuzzy_in_ast(node, domain)
                result = self._evaluate_node(resolved, data)
                expressions.append(result)

        return {
            'result': all(expr['result'] for expr in expressions),
            'confidence': min(expr['confidence'] for expr in expressions),
            'resolved_expression': ' AND '.join(expr['expression'] for expr in expressions),
            'trace': [expr['trace'] for expr in expressions]
        }

    def _resolve_fuzzy_in_ast(self, node: ASTNode, domain: str) -> ASTNode:
        """Walk AST and resolve all fuzzy predicates"""
        if isinstance(node, FuzzyPredicate):
            context = {
                'field_name': node.subject.name if isinstance(node.subject, Identifier) else None,
                'operator': node.operator,
                'domain': domain
            }
            resolution = self.resolver.resolve(node.fuzzy_value, context)

            # Convert resolution to deterministic AST node
            # This is simplified - real implementation would parse resolution.expression
            return resolution

        # Recursively resolve children
        # ...
        return node

    def _evaluate_node(self, node: ASTNode, data: Dict) -> Dict:
        """Evaluate a single AST node"""
        if isinstance(node, BinaryOp):
            left = self._evaluate_node(node.left, data)
            right = self._evaluate_node(node.right, data)

            if node.operator == '&&':
                return {
                    'result': left['result'] and right['result'],
                    'confidence': min(left['confidence'], right['confidence']),
                    'expression': f"({left['expression']} AND {right['expression']})",
                    'trace': f"{left['trace']} && {right['trace']}"
                }
            # ... other operators

        elif isinstance(node, Identifier):
            return {
                'result': data.get(node.name),
                'confidence': 1.0,
                'expression': node.name,
                'trace': f"{node.name}={data.get(node.name)}"
            }

        # ... other node types
```

### 1.4 End-to-End Example

```python
# example_usage.py
from parser import parse_ndel
from interpreter import NDELInterpreter

# Parse NDEL
ndel_code = """
@domain("soccer")

age < "young" && potential is "high" && salary < 50000
"""

ast = parse_ndel(ndel_code)

# Evaluate against data
interpreter = NDELInterpreter("domains/soccer.yaml")
player_data = {
    'age': 21,
    'potential': 82,
    'salary': 45000,
    'position': 'ST'
}

result = interpreter.evaluate(ast, player_data)

print(f"Match: {result['result']}")
print(f"Confidence: {result['confidence']:.0%}")
print(f"Resolved: {result['resolved_expression']}")
print(f"Trace: {result['trace']}")

# Output:
# Match: True
# Confidence: 80%
# Resolved: age < 23 AND potential > 78 AND salary < 50000
# Trace: age=21 < 23 && potential=82 > 78 && salary=45000 < 50000
```

## Phase 2: Production Features (Months 3-4)

### 2.1 Learning System

```python
class LearningResolver(RuleBasedResolver):
    """Resolver that learns from feedback"""

    def __init__(self, domain_path: str, history_db: str):
        super().__init__(domain_path)
        self.db = sqlite3.connect(history_db)
        self._init_db()

    def resolve_with_learning(self, fuzzy_value: str, context: dict) -> Resolution:
        # Get base resolution
        resolution = super().resolve(fuzzy_value, context)

        # Check historical accuracy
        historical = self._get_historical_accuracy(fuzzy_value, context)
        if historical:
            resolution.confidence = self._adjust_confidence(
                resolution.confidence,
                historical['success_rate'],
                historical['sample_size']
            )

        # Log resolution for future learning
        self._log_resolution(fuzzy_value, context, resolution)

        return resolution

    def record_feedback(self, fuzzy_value: str, context: dict, outcome: str):
        """User confirms resolution was correct/incorrect"""
        self.db.execute("""
            INSERT INTO feedback (fuzzy_value, context, outcome, timestamp)
            VALUES (?, ?, ?, ?)
        """, (fuzzy_value, json.dumps(context), outcome, datetime.now()))
        self.db.commit()

        # Update confidence for this pattern
        self._recompute_confidence(fuzzy_value, context)
```

### 2.2 SQL Code Generation

```python
class NDELToSQL:
    """Compile NDEL to executable SQL"""

    def compile(self, ndel_code: str, domain: str, table: str) -> str:
        ast = parse_ndel(ndel_code)
        interpreter = NDELInterpreter(f"domains/{domain}.yaml")

        # Resolve fuzzy values
        resolved = interpreter._resolve_fuzzy_in_ast(ast[1], domain)  # Skip domain declaration

        # Generate SQL
        sql = self._ast_to_sql(resolved, table)
        return sql

    def _ast_to_sql(self, node: ASTNode, table: str) -> str:
        where_clause = self._node_to_sql(node)
        return f"SELECT * FROM {table} WHERE {where_clause}"

    def _node_to_sql(self, node: ASTNode) -> str:
        if isinstance(node, BinaryOp):
            left = self._node_to_sql(node.left)
            right = self._node_to_sql(node.right)

            op_map = {
                '&&': 'AND',
                '||': 'OR',
                '<': '<',
                '>': '>',
                '<=': '<=',
                '>=': '>=',
                '==': '=',
                '!=': '!='
            }

            return f"({left} {op_map[node.operator]} {right})"

        elif isinstance(node, Identifier):
            return node.name

        elif isinstance(node, Literal):
            if isinstance(node.value, str):
                return f"'{node.value}'"
            return str(node.value)

        # Handle resolved fuzzy predicates
        # ...
```

### 2.3 REST API

```python
# api/server.py
from flask import Flask, request, jsonify
from ndel_to_sql import NDELToSQL
import psycopg2

app = Flask(__name__)
compiler = NDELToSQL()

@app.route('/query', methods=['POST'])
def query():
    """
    Execute NDEL query against database

    POST /query
    {
        "ndel": "age < 'young' && potential is 'high'",
        "domain": "soccer",
        "table": "players"
    }
    """
    data = request.json
    ndel = data['ndel']
    domain = data['domain']
    table = data['table']

    try:
        # Compile to SQL
        sql = compiler.compile(ndel, domain, table)

        # Execute
        conn = psycopg2.connect("dbname=soccer")
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()

        return jsonify({
            'success': True,
            'sql': sql,
            'results': results,
            'count': len(results)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/explain', methods=['POST'])
def explain():
    """
    Explain how NDEL query will be resolved

    POST /explain
    {
        "ndel": "age < 'young'",
        "domain": "soccer"
    }
    """
    data = request.json
    ast = parse_ndel(data['ndel'])
    interpreter = NDELInterpreter(f"domains/{data['domain']}.yaml")

    # Get resolution without executing
    resolution = interpreter._resolve_fuzzy_in_ast(ast[1], data['domain'])

    return jsonify({
        'original': data['ndel'],
        'resolved': resolution.expression,
        'confidence': resolution.confidence,
        'reasoning': resolution.reasoning,
        'alternatives': [
            {
                'expression': alt.expression,
                'confidence': alt.confidence
            }
            for alt in (resolution.alternatives or [])
        ]
    })
```

## Phase 3: Advanced Features (Months 5-6)

### 3.1 LLM Fallback

```python
class HybridResolver(LearningResolver):
    """Resolver with LLM fallback"""

    def __init__(self, domain_path: str, history_db: str, llm_client):
        super().__init__(domain_path, history_db)
        self.llm = llm_client

    def resolve(self, fuzzy_value: str, context: dict) -> Resolution:
        try:
            # Try rule-based first
            return super().resolve(fuzzy_value, context)
        except ResolutionError:
            # Fall back to LLM
            return self._llm_resolve(fuzzy_value, context)

    def _llm_resolve(self, fuzzy_value: str, context: dict) -> Resolution:
        prompt = f"""
        You are a fuzzy value resolver for the {context.get('domain', 'general')} domain.

        Convert this fuzzy value to a precise expression:

        Fuzzy value: "{fuzzy_value}"
        Field name: {context.get('field_name')}
        Field type: {context.get('field_type', 'unknown')}
        Operator: {context.get('operator')}

        Respond in JSON format:
        {{
            "expression": "< 23",
            "confidence": 0.85,
            "reasoning": "Explanation of why this interpretation makes sense"
        }}
        """

        response = self.llm.complete(prompt)
        result = json.loads(response)

        # LLM resolutions get confidence penalty
        result['confidence'] *= 0.9

        return Resolution(**result)
```

### 3.2 Web UI for Domain Management

```html
<!-- web/domain_builder.html -->
<div id="domain-builder">
  <h2>Build Domain: Soccer</h2>

  <div class="fuzzy-mapping">
    <h3>Field: age</h3>
    <button onclick="addMapping('age')">Add Mapping</button>

    <div class="mapping">
      <input type="text" placeholder="Fuzzy term" value="young">
      <input type="text" placeholder="Expression" value="< 23">
      <input type="number" placeholder="Confidence" value="0.85" step="0.01">
      <textarea placeholder="Reasoning">U-23 is standard youth category</textarea>
      <button onclick="testMapping()">Test</button>
    </div>
  </div>

  <div class="test-panel">
    <h3>Test Queries</h3>
    <textarea placeholder="Enter NDEL query">age < "young" && position is "striker"</textarea>
    <button onclick="testQuery()">Test</button>

    <div id="results">
      <h4>Resolution:</h4>
      <pre id="resolved-expression"></pre>
      <p>Confidence: <span id="confidence"></span></p>
    </div>
  </div>
</div>
```

### 3.3 Analytics Dashboard

Track resolution quality:
- Success rate by fuzzy term
- Confidence distribution
- Most corrected terms
- Domain coverage
- Query latency

## Success Metrics

### Month 2 (MVP):
- ✅ Parser works (done!)
- ✅ Rule-based resolver: 70% accuracy
- ✅ 3 domains with 20+ fuzzy terms each
- ✅ Basic interpreter evaluates queries

### Month 4 (Production):
- ✅ 85% accuracy with learning
- ✅ SQL generation works
- ✅ REST API deployed
- ✅ <100ms query latency (cached)
- ✅ 5 production domains

### Month 6 (Advanced):
- ✅ LLM fallback: 90%+ accuracy
- ✅ Web UI for domain management
- ✅ Analytics dashboard
- ✅ 10 production domains
- ✅ Real users in beta

## Cost Estimate

### Development:
- 1 senior engineer × 6 months = $75k-$100k
- LLM API costs: $500-$1000/month
- Infrastructure: $100/month

**Total: ~$100k-$125k**

### ROI Analysis:
If this saves 10 data analysts 2 hours/week each:
- 20 hours/week × $50/hour = $1,000/week
- $52,000/year saved
- **Break-even in 2 years**

## Conclusion

NDEL can become production-ready in 6 months with focused effort on:
1. Rule-based fuzzy resolution (good enough for most cases)
2. SQL code generation (enables immediate value)
3. Learning from corrections (improves over time)
4. LLM fallback (handles edge cases)

The key is starting simple and proving value before adding complexity.
