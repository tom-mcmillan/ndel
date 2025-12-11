# CEL (Common Expression Language) - Deep Dive

## What is CEL?

**CEL is Google's lightweight, fast expression language designed for policy evaluation and validation.**

Created by Google, used internally in:
- Firebase Security Rules
- Google Cloud IAM conditions
- Kubernetes validation
- gRPC services

## The Problem CEL Solves

### Before CEL:
```yaml
# Firebase had to invent custom expression syntax
allow read: if request.auth.uid == resource.owner_id

# Kubernetes had to use complex YAML logic
# IAM had different expression syntax
# Every system reinvented the wheel
```

### With CEL:
```cel
// Universal expression language across all Google services
request.auth.uid == resource.owner_id
  && resource.visibility == "public"
  && timestamp(request.time) < timestamp(resource.expires_at)
```

**One language, many use cases.**

---

## CEL's Design Principles

### 1. **Keep it Small and Fast**

**Key constraint: Linear time evaluation**

```cel
// CEL evaluates in O(n) time, where n = expression size
// This is guaranteed!

list.all(x, x > 0)  // O(n) - iterates list once
list.map(x, x * 2)  // O(n) - single pass

// Not possible in CEL (would be O(n²)):
// nested_list.all(x, x.all(y, y > 0))  ❌ Deliberately limited
```

**Why?**
- Security: No denial-of-service via expensive expressions
- Predictability: Know maximum evaluation time
- Efficiency: Can evaluate millions of expressions/second

**Trade-off:** Not Turing-complete (can't write infinite loops)

### 2. **Not Turing-Complete (Intentional!)**

CEL **cannot**:
- ❌ Loop indefinitely
- ❌ Mutate state
- ❌ Have side effects
- ❌ Call arbitrary functions
- ❌ Do recursion

CEL **can**:
- ✅ Evaluate expressions
- ✅ Make decisions
- ✅ Validate data
- ✅ Transform data (without mutation)

**Example of what CEL cannot do:**
```javascript
// This is JavaScript (Turing-complete):
while (true) {  // Infinite loop possible
  data.push(x); // Mutation possible
}

// CEL equivalent (safe):
data.filter(x, x > 0)  // No loops, no mutation, guaranteed to finish
```

**Why limit power?**
- Safety in untrusted contexts
- Guaranteed termination
- Predictable resource usage

### 3. **Developer-Friendly Syntax**

CEL looks like C/Java/JavaScript:

```cel
// Familiar operators
x + y * z
a && b || c
list[0]
map["key"]
obj.field

// Familiar functions
size(list)
has(obj.field)
startsWith(text, "prefix")

// Ternary operator
condition ? true_value : false_value
```

### 4. **Make it Extensible**

CEL allows custom:
- **Functions**: Add domain-specific operations
- **Types**: Use protocol buffer types
- **Contexts**: Provide data to expressions

```cel
// Built-in CEL:
size(list) > 0

// Custom extension (you define):
isValidEmail(user.email)
customDomainCheck(resource)
```

### 5. **Protocol Buffer Native**

CEL is built on **Protocol Buffers** (Google's serialization format).

```protobuf
// Define types in .proto files
message Account {
  string user_id = 1;
  int64 balance = 2;
  repeated string emails = 3;
}
```

```cel
// Use them directly in CEL
account.balance >= 1000
&& size(account.emails) > 0
&& has(account.user_id)
```

**Benefits:**
- Type safety
- Schema validation
- Cross-language compatibility
- Serialization built-in

---

## CEL's Architecture

### The 4 Core Components:

```
1. Text Source
   ↓
2. Parser → AST (Abstract Syntax Tree)
   ↓
3. Type Checker → Typed AST
   ↓
4. Evaluator → Result
```

### 1. Text Representation
```cel
account.balance >= transaction.amount
```

### 2. AST (Abstract Syntax Tree)
```
        >=
       /  \
      .    .
     / \  / \
  account  transaction
  balance  amount
```

### 3. Type Checking
```
account.balance : int64
transaction.amount : int64
>= : (int64, int64) → bool
Result type: bool ✓
```

### 4. Evaluation
```
Context: {account: {balance: 1000}, transaction: {amount: 500}}
Result: true
```

---

## CEL Type System

### Primitive Types

| Type | Description | Example |
|------|-------------|---------|
| `int` | 64-bit signed integer | `42`, `-17` |
| `uint` | 64-bit unsigned integer | `42u` |
| `double` | 64-bit float | `3.14`, `2.5e10` |
| `bool` | Boolean | `true`, `false` |
| `string` | Unicode string | `"hello"` |
| `bytes` | Byte sequence | `b"data"` |
| `null` | Null value | `null` |

### Collection Types

```cel
// List
[1, 2, 3, 4, 5]
["a", "b", "c"]

// Map
{"key1": value1, "key2": value2}
{1: "one", 2: "two"}

// Nested
[[1, 2], [3, 4]]
{"users": [user1, user2]}
```

### Protocol Buffer Types

```cel
// Message type (from .proto definition)
Account{
  user_id: "user123",
  balance: 1000,
  emails: ["a@example.com"]
}
```

### Dynamic Type (`dyn`)

```cel
// For heterogeneous collections
[1, "hello", true, null]  // List of dyn

// Or when type is unknown at compile-time
type(value) == string ? value : string(value)
```

---

## CEL Operators (Precedence Order)

### 1. Member Access (Highest)
```cel
obj.field
list[index]
map["key"]
```

### 2. Unary
```cel
!condition
-number
```

### 3. Multiplicative
```cel
a * b
a / b
a % b
```

### 4. Additive
```cel
a + b
a - b
```

### 5. Relational
```cel
a < b
a <= b
a > b
a >= b
```

### 6. Equality
```cel
a == b
a != b
a in list
```

### 7. Logical AND
```cel
a && b
```

### 8. Logical OR
```cel
a || b
```

### 9. Conditional (Lowest)
```cel
condition ? true_value : false_value
```

---

## CEL Built-in Functions

### Type Checking
```cel
type(value)              // Get type as string
has(obj.field)           // Check field exists
dyn(value)              // Convert to dynamic type
```

### Size Operations
```cel
size(list)              // List length
size(map)               // Map size
size(string)            // String length (bytes)
size(bytes)             // Byte sequence length
```

### String Operations
```cel
contains(str, substr)
startsWith(str, prefix)
endsWith(str, suffix)
matches(str, regex)
```

### Timestamp/Duration
```cel
timestamp(str)          // Parse timestamp
duration(str)           // Parse duration
```

### Collection Operations
```cel
// Macros (special syntax)
list.all(x, x > 0)              // All elements match
list.exists(x, x > 10)          // Any element matches
list.exists_one(x, x == 5)      // Exactly one matches
list.map(x, x * 2)              // Transform elements
list.filter(x, x > 0)           // Filter elements
```

---

## CEL Key Features

### 1. Short-Circuit Evaluation
```cel
// If left side is false, right side is NOT evaluated
false && expensive_function()  // expensive_function() never called

// If left side is true, right side is NOT evaluated
true || expensive_function()   // expensive_function() never called
```

### 2. Null-Safe Navigation
```cel
// Traditional (crashes if obj is null):
obj.field

// Safe (returns null if obj is null):
has(obj.field) ? obj.field : default_value
```

### 3. Macros (Comprehensions)
```cel
// Check all elements
numbers.all(x, x > 0)

// Find any element
numbers.exists(x, x > 100)

// Exactly one
numbers.exists_one(x, x == 42)

// Transform
numbers.map(x, x * 2)

// Filter
numbers.filter(x, x % 2 == 0)
```

### 4. Optional Indices (Safe Access)
```cel
// Returns null if index doesn't exist instead of error
list[?10]    // Safe index access
map[?"key"]  // Safe key access
```

---

## Example Use Cases

### 1. Security Policy (Firebase)
```cel
// Only owners can delete, anyone can read public posts
resource.owner == request.auth.uid  // Delete rule
|| resource.visibility == "public"   // Read rule
```

### 2. Validation (Kubernetes)
```cel
// Validate replica count is reasonable
object.spec.replicas >= 1
&& object.spec.replicas <= 100
```

### 3. IAM Conditions (Google Cloud)
```cel
// Allow access during business hours from specific IPs
request.time.getHours("America/Los_Angeles") >= 9
&& request.time.getHours("America/Los_Angeles") <= 17
&& request.ip in ["203.0.113.0/24", "198.51.100.0/24"]
```

### 4. Data Validation
```cel
// Validate email and age
has(user.email)
&& user.email.matches("[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}")
&& user.age >= 18
&& user.age < 120
```

---

## CEL Limitations (By Design)

### Cannot Do:
1. **Loops** - No `while`, `for`
2. **Mutation** - All operations are pure
3. **Recursion** - Limited depth
4. **Side Effects** - No I/O, no state changes
5. **Variable Assignment** - No `x = 5`
6. **Complex Algorithms** - Not Turing-complete

### Why These Limitations?
- **Safety** - Can run untrusted code
- **Performance** - Guaranteed linear time
- **Predictability** - Always terminates
- **Simplicity** - Easy to understand and verify

---

## What NDEL Inherits from CEL

### ✅ Keep from CEL:
1. **Safe evaluation** - No infinite loops
2. **Familiar syntax** - C-like expressions
3. **Type system** - Strong typing with gradual typing
4. **Operators** - Standard mathematical/logical ops
5. **Pure functions** - No side effects
6. **AST-based** - Parse → AST → Evaluate

### 🆕 What NDEL Adds:
1. **Non-determinism** - LLM calls are fuzzy
2. **Confidence tracking** - Every result has confidence
3. **Fuzzy parameters** - `creativity: "high"` instead of `temperature: 0.9`
4. **LLM operations** - `llm.complete()`, `llm.analyze()`
5. **Quality gates** - `require confidence > 0.8`
6. **Human-in-the-loop** - `await human_review()`
7. **Learning** - System improves from feedback

---

## The Key Difference: CEL vs NDEL

### CEL Philosophy:
**"Deterministic evaluation of expressions on structured data"**

```cel
// Input → Deterministic evaluation → Output
account.balance >= 1000  // Always same result for same input
```

### NDEL Philosophy:
**"Controlled non-deterministic computation with confidence tracking"**

```ndel
// Input → Non-deterministic evaluation → Output + Confidence
llm.analyze(text) with reasoning: "careful"
// Different results each time, but with confidence scores
```

---

## CEL Implementation Details

### Written In:
- **Go** - Reference implementation (google/cel-go)
- **C++** - (google/cel-cpp)
- **Java** - (google/cel-java)

### Performance:
- Evaluates **millions of expressions per second**
- Linear time guarantee: O(n) where n = expression complexity
- Memory-safe, no allocations during evaluation (in some implementations)

### Tooling:
- Online playground: https://playcel.undistro.io/
- Official spec: https://github.com/google/cel-spec
- Conformance tests across implementations

---

## What We Should Learn from CEL for NDEL

### 1. **Simplicity is a Feature**
CEL is intentionally limited. NDEL should be too. Don't try to be Python.

### 2. **Safety First**
CEL prioritizes safety over power. NDEL should do the same with LLM operations.

### 3. **Performance Matters**
Linear-time guarantee makes CEL usable at scale. NDEL needs similar constraints.

### 4. **Extensibility Through Functions**
CEL allows custom functions. NDEL should allow custom LLM operations.

### 5. **Types Provide Safety**
Strong typing caught errors. NDEL needs types for LLM responses, confidence, etc.

### 6. **Standard Syntax Reduces Friction**
CEL looks familiar. NDEL should too.

---

## NDEL's Position Relative to CEL

```
Simple                                           Complex
  ↓                                                 ↓
CEL ────────────── NDEL ────────────── Python
  ↑                   ↑                     ↑
  |                   |                     |
Pure expressions   LLM orchestration    General programming
Deterministic      Controlled fuzzy     Turing-complete
Always same result Confidence scores    Side effects allowed
```

**NDEL occupies the sweet spot:**
- More expressive than CEL (LLM operations)
- Safer/simpler than Python (not Turing-complete)
- Purpose-built for LLM orchestration

---

## Summary: CEL's Lessons for NDEL

1. **Keep it focused** - CEL does one thing well (expression evaluation)
2. **Embrace limitations** - Not Turing-complete is a feature
3. **Familiar syntax** - Developers shouldn't need to learn new syntax
4. **Type safety** - Types catch errors early
5. **Linear time** - Performance guarantees matter
6. **Extensible** - Allow custom functions/operations
7. **Well-specified** - Clear language definition
8. **Multi-language** - Reference implementations in multiple languages

**NDEL should follow these principles while adding:**
- Non-determinism as a first-class concept
- Confidence tracking throughout
- LLM-specific operations
- Fuzzy parameter resolution

This makes NDEL **"CEL for the LLM era"** - the same philosophy applied to probabilistic computation.
