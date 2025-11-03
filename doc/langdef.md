# NDEL Language Definition

Version: 0.1.0  
Status: Draft Specification

## 1. Overview

NDEL (Non-Deterministic Expression Language) is an expression language that combines deterministic computational structure with non-deterministic value resolution. It enables users to write expressions mixing precise values with natural language descriptions that are interpreted based on context.

### 1.1 Design Principles

- **Deterministic Structure**: Operators, control flow, and composition rules are fixed
- **Non-Deterministic Values**: Natural language values are interpreted probabilistically
- **Confidence Tracking**: Every interpretation carries a confidence score
- **Domain Adaptability**: Interpretations vary by domain context
- **Gradual Precision**: Expressions can range from fully fuzzy to fully precise

### 1.2 Inspiration

NDEL builds upon Google's Common Expression Language (CEL) foundations while adding:
- Fuzzy value interpretation
- Confidence propagation
- Domain-specific resolution
- Learning capabilities

## 2. Lexical Elements

### 2.1 Character Set

NDEL uses UTF-8 encoding and supports Unicode identifiers.

### 2.2 Tokens
```
TOKEN           = IDENTIFIER | KEYWORD | OPERATOR | LITERAL | FUZZY_STRING
IDENTIFIER      = LETTER (LETTER | DIGIT | '_')*
KEYWORD         = 'true' | 'false' | 'null' | 'if' | 'then' | 'else' | 
                  'and' | 'or' | 'not' | 'is' | 'shows' | 'has' | 'in'
OPERATOR        = '+' | '-' | '*' | '/' | '%' | '==' | '!=' | '<' | '>' | 
                  '<=' | '>=' | '&&' | '||' | '!' | '?' | ':' | '.'
LITERAL         = NUMBER | STRING | BOOLEAN | NULL
FUZZY_STRING    = '"' (~["\r\n])* '"'  // Interpreted contextually
```

### 2.3 Reserved Words

The following words are reserved and cannot be used as identifiers:
```
and         else        function    in          let         or
approximately  false       has         is          not         shows
confidence  for         if          null        return      then
domain      fuzzy       import      of          roughly     true
```

### 2.4 Comments
```
// Single line comment
/* Multi-line 
   comment */
```

## 3. Types

### 3.1 Primitive Types

| Type | Description | Examples |
|------|-------------|----------|
| `bool` | Boolean | `true`, `false` |
| `int` | 64-bit signed integer | `42`, `-17`, `0` |
| `double` | 64-bit floating point | `3.14`, `2.5e10` |
| `string` | UTF-8 string | `'hello'`, `"world"` |
| `bytes` | Byte sequence | `b'\\x01\\x02'` |
| `null` | Null value | `null` |

### 3.2 Composite Types

| Type | Description | Examples |
|------|-------------|----------|
| `list` | Ordered collection | `[1, 2, 3]`, `['a', 'b']` |
| `map` | Key-value pairs | `{'x': 1, 'y': 2}` |
| `struct` | Named fields | `Point{x: 1.0, y: 2.0}` |

### 3.3 Special Types

| Type | Description | Examples |
|------|-------------|----------|
| `fuzzy` | Value requiring interpretation | `"young"`, `"high quality"` |
| `confidence` | Confidence score [0.0, 1.0] | `0.85` |
| `resolution` | Interpretation result | `{value: 25, confidence: 0.9}` |

### 3.4 Type Coercion

NDEL performs minimal automatic type coercion:
- Fuzzy values are resolved to their target type based on context
- No automatic numeric conversions (must be explicit)
- String concatenation with `+` converts operands to strings

## 4. Expressions

### 4.1 Primary Expressions
```ndel
// Literals
42                  // int
3.14                // double  
"hello"             // string
true                // bool
null                // null

// Identifiers
player_age          // variable reference
team.name           // field access

// Fuzzy values
"young player"      // fuzzy string
"high performance"  // requires interpretation
```

### 4.2 Operators

#### 4.2.1 Arithmetic Operators

| Operator | Description | Types |
|----------|-------------|-------|
| `+` | Addition | numeric, string |
| `-` | Subtraction | numeric |
| `*` | Multiplication | numeric |
| `/` | Division | numeric |
| `%` | Modulo | int |
| `-` | Unary negation | numeric |

#### 4.2.2 Comparison Operators

| Operator | Description | Result |
|----------|-------------|--------|
| `==` | Equal | bool |
| `!=` | Not equal | bool |
| `<` | Less than | bool |
| `<=` | Less than or equal | bool |
| `>` | Greater than | bool |
| `>=` | Greater than or equal | bool |

#### 4.2.3 Logical Operators

| Operator | Description | Short-circuit |
|----------|-------------|---------------|
| `&&` | Logical AND | Yes |
| `||` | Logical OR | Yes |
| `!` | Logical NOT | No |

#### 4.2.4 Fuzzy Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `is` | Fuzzy predicate | `player is "promising"` |
| `shows` | Pattern matching | `trend shows "improvement"` |
| `approximately` | Fuzzy equality | `value approximately 100` |

### 4.3 Conditional Expression
```ndel
condition ? true_value : false_value
```

### 4.4 Member Access
```ndel
object.field           // Field access
map['key']            // Map lookup
list[0]               // List index
```

### 4.5 Function Calls
```ndel
function_name(arg1, arg2, ...)
object.method(args)   // Method call syntax
```

## 5. Fuzzy Value Resolution

### 5.1 Fuzzy Value Identification

Fuzzy values are identified as:
1. Quoted strings in comparison contexts
2. Arguments to fuzzy operators (`is`, `shows`)
3. Explicitly marked with `fuzzy` prefix

### 5.2 Resolution Process
```
Input: "young player"
Context: {field: "age", domain: "soccer", type: int}
Steps:
  1. Identify as fuzzy value
  2. Apply domain rules
  3. Generate interpretation
  4. Calculate confidence
Output: {value: "age < 23", confidence: 0.85}
```

### 5.3 Confidence Propagation

Confidence propagates through expressions:

| Operation | Confidence Calculation |
|-----------|----------------------|
| `A && B` | `min(conf(A), conf(B))` |
| `A || B` | `weighted_avg(conf(A), conf(B))` |
| `!A` | `conf(A)` |
| `A op B` | `min(conf(A), conf(B))` for comparison ops |

## 6. Evaluation

### 6.1 Evaluation Order

1. Parse expression to AST
2. Identify fuzzy values
3. Resolve fuzzy values using domain context
4. Type check resolved expression
5. Evaluate deterministic expression
6. Return result with confidence

### 6.2 Short-Circuit Evaluation

Logical operators `&&` and `||` short-circuit:
- `&&`: If left operand is false, right is not evaluated
- `||`: If left operand is true, right is not evaluated

### 6.3 Error Handling

NDEL defines these runtime errors:
- `resolution_error`: Fuzzy value cannot be resolved
- `type_error`: Type mismatch after resolution
- `confidence_error`: Confidence below required threshold
- `domain_error`: Domain context not available

## 7. Domain Context

### 7.1 Domain Declaration
```ndel
@domain("soccer")
where player is "young"  // Resolves using soccer context
```

### 7.2 Domain Interface

Each domain must implement:
```
interface DomainContext {
  resolve(fuzzyValue: string, fieldContext: Context): Resolution
  getConfidenceThreshold(): double
  getFunctions(): FunctionRegistry
}
```

## 8. Standard Library

### 8.1 Type Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `type()` | Get type of value | `(any) → string` |
| `has()` | Check field existence | `(any, string) → bool` |
| `default()` | Provide default value | `(any, any) → any` |
| `cast()` | Type conversion | `(any, string) → any` |

### 8.2 Confidence Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `confidence()` | Get current confidence | `() → double` |
| `with_confidence()` | Set minimum confidence | `(double, any) → any` |
| `alternatives()` | Get alternative interpretations | `() → list` |

### 8.3 List Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `size()` | Get size | `(list) → int` |
| `all()` | All elements match | `(list, pred) → bool` |
| `exists()` | Any element matches | `(list, pred) → bool` |
| `filter()` | Filter elements | `(list, pred) → list` |
| `map()` | Transform elements | `(list, func) → list` |

### 8.4 String Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `contains()` | Contains substring | `(string, string) → bool` |
| `startsWith()` | Starts with prefix | `(string, string) → bool` |
| `endsWith()` | Ends with suffix | `(string, string) → bool` |
| `matches()` | Regex match | `(string, string) → bool` |
| `lower()` | Convert to lowercase | `(string) → string` |
| `upper()` | Convert to uppercase | `(string) → string` |

## 9. Grammar Summary
```ebnf
expression     ::= logical_or
logical_or     ::= logical_and ('||' logical_and)*
logical_and    ::= equality ('&&' equality)*
equality       ::= relational (('==' | '!=') relational)*
relational     ::= additive (('<' | '>' | '<=' | '>=') additive)*
additive       ::= multiplicative (('+' | '-') multiplicative)*
multiplicative ::= unary (('*' | '/' | '%') unary)*
unary          ::= ('!' | '-')? fuzzy
fuzzy          ::= postfix ('is' STRING | 'shows' STRING)?
postfix        ::= primary ('.' IDENT | '[' expression ']' | '(' args? ')')*
primary        ::= IDENT | LITERAL | '(' expression ')'
```

## 10. Examples

### 10.1 Basic Expressions
```ndel
// Deterministic
age > 25 && goals >= 10

// Fuzzy
player is "young" && performance is "excellent"

// Mixed
salary < 100000 && potential is "high"
```

### 10.2 Domain-Specific
```ndel
// Soccer domain
where player is "clinical finisher" 
  and age < "veteran"
  and salary is "reasonable for talent"

// Finance domain  
where stock shows "bullish pattern"
  and volume is "above average"
  and risk is "acceptable"
```

### 10.3 Confidence Control
```ndel
// Require high confidence
with_confidence(0.9, player is "world class")

// Check confidence
if confidence() > 0.8 then
  execute_trade()
else
  request_human_review()
```

## 11. Implementation Notes

### 11.1 Parsing
- Use ANTLR4 or similar parser generator
- Maintain source position for error reporting
- Preserve fuzzy values as special AST nodes

### 11.2 Resolution
- Cache resolved fuzzy values
- Track resolution history for learning
- Provide explanation generation

### 11.3 Optimization
- Compile frequently-used expressions
- Pre-resolve static fuzzy values
- Implement incremental confidence updates

## 12. Future Extensions

- Pattern matching syntax
- Async function calls
- Streaming evaluation
- Multi-domain expressions
- Federated learning
- Natural language generation from expressions

---

*This document defines NDEL version 0.1.0. The language is under active development and this specification may change.*
