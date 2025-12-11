# Anatomy of a Programming Language

## Overview: The 7 Essential Components

Every programming language, from Python to C++ to our NDEL, needs these parts:

```
Source Code (text file)
       ↓
1. LEXER (Tokenizer) - Break text into words/symbols
       ↓
   Tokens (stream of words)
       ↓
2. PARSER - Organize tokens into structure
       ↓
   AST (Abstract Syntax Tree)
       ↓
3. SEMANTIC ANALYZER - Check if it makes sense
       ↓
   Validated AST
       ↓
4. INTERPRETER/COMPILER - Execute or translate
       ↓
5. RUNTIME - Provide built-in operations
       ↓
   Results/Effects
       ↓
6. STANDARD LIBRARY - Common functions/features
       ↓
7. LANGUAGE SPECIFICATION - The rules/documentation
```

Let me explain each in detail:

---

## 1. LEXER (Tokenizer)

**What it does:** Breaks source code text into meaningful chunks called "tokens"

**Analogy:** Like reading English - you break text into words, punctuation, numbers.

**Example:**
```
Source code:  age < "young" && status == "active"

Lexer output:
  IDENTIFIER(age)
  LESS(<)
  STRING("young")
  AND(&&)
  IDENTIFIER(status)
  EQUAL(==)
  STRING("active")
  EOF
```

**Components:**
- **Character stream** - Read source code character by character
- **Token types** - Define what kinds of tokens exist (IDENTIFIER, NUMBER, STRING, OPERATOR, etc.)
- **Pattern matching** - Recognize patterns (e.g., numbers, strings in quotes, keywords)
- **Error handling** - Report illegal characters

**What NDEL has:** ✅ **COMPLETE** - [parser.py lines 88-313](reference/python/parser.py)

**Example from NDEL:**
```python
class TokenType(Enum):
    NUMBER = auto()
    STRING = auto()
    IDENTIFIER = auto()
    # ... etc

class Lexer:
    def next_token(self) -> Token:
        # Read characters, return tokens
        pass
```

---

## 2. PARSER

**What it does:** Organizes tokens into a tree structure (AST) that represents the program's structure

**Analogy:** Like diagramming sentences in English class - subject, verb, object, modifiers

**Example:**
```
Tokens: age < 23 && status == "active"

AST:
         &&
        /  \
       /    \
      <      ==
     / \    /  \
   age 23 status "active"
```

**Components:**
- **Grammar rules** - Define valid syntax (e.g., expression = term operator term)
- **Parsing algorithm** - Usually "recursive descent" or use a parser generator
- **AST node types** - Classes for different constructs (BinaryOp, Literal, FunctionCall, etc.)
- **Error reporting** - Tell user where syntax is wrong

**What NDEL has:** ✅ **MOSTLY COMPLETE** - [parser.py lines 403-811](reference/python/parser.py)

**Example from NDEL:**
```python
@dataclass
class BinaryOp(ASTNode):
    left: ASTNode
    operator: str
    right: ASTNode

class Parser:
    def parse_expression(self):
        # Build AST from tokens
        pass
```

**What NDEL needs:**
- ⬜ Add LLM operation syntax (`llm.complete()`)
- ⬜ Add `with` parameter syntax
- ⬜ Add `require` quality gate syntax
- ⬜ Add function definition syntax
- ⬜ Add control flow (if/else, while, for)

---

## 3. SEMANTIC ANALYZER

**What it does:** Checks if the program makes logical sense (even if syntax is correct)

**Analogy:** Checking if an English sentence is grammatically correct AND meaningful
- "The dog ate the bone" ✅ (correct and meaningful)
- "The bone ate the dog" ✅ (correct grammar but weird)
- "The idea ate the justice" ❌ (nonsense)

**What it checks:**
- **Type checking** - Is `"hello" + 5` valid? Depends on language
- **Variable scope** - Is this variable defined? Can we access it here?
- **Function signatures** - Right number/type of arguments?
- **Semantic errors** - Using undefined variables, type mismatches, etc.

**Example:**
```ndel
// Syntax is fine, but semantically wrong:
result = llm.complete(123)  // ❌ prompt should be string, not number

age = "hello"
total = age + 5  // ❌ can't add string to number
```

**Components:**
- **Symbol table** - Track defined variables, functions, types
- **Type system** - Rules for what types exist and how they interact
- **Scope management** - Track what's accessible where
- **Error messages** - Helpful explanations of what's wrong

**What NDEL has:** ❌ **MISSING**

**What NDEL needs:**
- ⬜ Type system (int, string, bool, llm_response, confidence, etc.)
- ⬜ Type checking algorithm
- ⬜ Symbol table for variables and functions
- ⬜ Scope management (global, function, block scopes)

---

## 4. INTERPRETER or COMPILER

This is where the two main approaches diverge:

### Option A: INTERPRETER (What we want for NDEL)

**What it does:** Walks the AST and executes each node immediately

**Analogy:** Like a simultaneous translator - translate and speak at the same time

**How it works:**
```python
def evaluate(node: ASTNode, context: dict) -> Any:
    if isinstance(node, Literal):
        return node.value

    elif isinstance(node, BinaryOp):
        left = evaluate(node.left, context)
        right = evaluate(node.right, context)

        if node.operator == '+':
            return left + right
        elif node.operator == '<':
            return left < right
        # ... etc

    elif isinstance(node, LLMCall):
        # Execute LLM operation!
        return call_llm(node.prompt, node.parameters)
```

**Components:**
- **Evaluation engine** - Walk AST and execute nodes
- **Runtime context** - Track variables, function definitions
- **Built-in operations** - Implement +, -, <, >, etc.
- **Error handling** - Runtime errors (division by zero, undefined variable)

**What NDEL has:** ❌ **EMPTY FILE** - [interpreter.py](reference/python/interpreter.py) is 1 line

**What NDEL needs:**
- ⬜ AST walker/evaluator
- ⬜ Variable storage (context/environment)
- ⬜ Built-in operators (+, -, <, &&, ||)
- ⬜ Function call mechanism
- ⬜ LLM operation execution
- ⬜ Error handling and reporting

### Option B: COMPILER (Not what we need, but for completeness)

**What it does:** Translates AST to another language (usually machine code or bytecode)

**Analogy:** Like translating a book from English to French - translate everything first, read later

**Examples:**
- C compiler → machine code
- TypeScript → JavaScript
- NDEL → Python (could be useful later!)

**We don't need this for v1**, but could add later to compile NDEL → Python for performance.

---

## 5. RUNTIME SYSTEM

**What it does:** Provides the "built-in" capabilities the language can use

**Analogy:** Like the OS on your computer - provides services programs can use

**Components:**

### a) Standard Operations
```python
# Basic arithmetic
3 + 5  → runtime provides addition
10 / 2 → runtime provides division

# Comparisons
5 < 10 → runtime provides comparison

# Logical operations
true && false → runtime provides AND/OR/NOT
```

### b) Built-in Functions
```python
# String functions
len("hello")  → 5
upper("hello") → "HELLO"

# Math functions
sqrt(16) → 4
random() → 0.7238...

# LLM functions (NDEL-specific!)
llm.complete(prompt) → calls actual LLM
```

### c) Memory Management
- Allocate memory for variables
- Garbage collection (clean up unused memory)
- Reference counting

### d) I/O Operations
- Print to console
- Read/write files
- Network requests

**What NDEL has:** ❌ **MISSING**

**What NDEL needs:**
- ⬜ Basic operations (arithmetic, comparison, logical)
- ⬜ String operations
- ⬜ List/map operations
- ⬜ **LLM runtime** - The core feature! ⭐
- ⬜ Fuzzy parameter resolver
- ⬜ Confidence tracking
- ⬜ Quality gate checker

---

## 6. STANDARD LIBRARY

**What it does:** Provides common functions and utilities that aren't built into the runtime

**Analogy:** Like a toolbox - the language gives you a hammer (runtime), the standard library gives you screwdrivers, wrenches, etc.

**Examples in other languages:**

**Python:**
```python
import os        # File system operations
import datetime  # Date/time handling
import json      # JSON parsing
import requests  # HTTP requests
```

**JavaScript:**
```javascript
Math.max(1, 2, 3)     // Math utilities
Array.from([1, 2, 3]) // Array utilities
JSON.parse("{}")      // JSON parsing
```

**What NDEL needs:**

### Core Library
```ndel
// Text operations
text.split(string, delimiter)
text.join(list, delimiter)
text.format(template, values)

// List operations
list.map(items, function)
list.filter(items, condition)
list.reduce(items, function)

// LLM utilities
llm.complete(prompt, params)
llm.analyze(text, focus)
llm.summarize(text, length)
llm.translate(text, language)
```

### Domain Libraries
```ndel
// Code generation domain
@import "domains/code_generation"

code.review(source_code)
code.refactor(source_code, style)
code.test(source_code)

// Content generation domain
@import "domains/content"

content.write(topic, style, length)
content.edit(text, instructions)
content.translate(text, language)
```

**What NDEL has:** ❌ **MISSING**

**What NDEL needs:**
- ⬜ Core library with basic utilities
- ⬜ LLM operation library
- ⬜ Domain-specific libraries
- ⬜ Import/module system

---

## 7. LANGUAGE SPECIFICATION

**What it does:** Documents what the language is, how it works, and how to use it

**Analogy:** Like a dictionary + grammar book + style guide for a spoken language

**What it contains:**

### a) Language Reference
- **Syntax** - What's valid code?
- **Grammar** - Formal rules (like BNF notation)
- **Keywords** - Reserved words
- **Operators** - What they do and precedence
- **Types** - What data types exist

### b) Semantics
- **Type rules** - How types interact
- **Evaluation order** - Left-to-right? Short-circuit?
- **Scoping rules** - Where are variables accessible?
- **Error conditions** - What causes errors?

### c) Standard Library Documentation
- Function signatures
- Parameters and return types
- Examples and use cases

### d) Style Guide
- Best practices
- Naming conventions
- Code formatting

**What NDEL has:** ⚠️ **PARTIAL**
- ✅ [langdef.md](doc/langdef.md) - Basic language definition
- ✅ [fuzzy-resolution.md](doc/fuzzy-resolution.md) - Fuzzy value spec
- ❌ Missing: LLM operations, stdlib docs, examples

**What NDEL needs:**
- ⬜ Complete syntax specification with LLM operations
- ⬜ Type system documentation
- ⬜ Standard library reference
- ⬜ Tutorial and examples
- ⬜ Best practices guide

---

## Summary: Current State of NDEL

| Component | Status | What Exists | What's Missing |
|-----------|--------|-------------|----------------|
| **1. Lexer** | ✅ 90% | Tokenizes most syntax | LLM keywords, `with`, `require` |
| **2. Parser** | ✅ 70% | Parses basic expressions | LLM calls, functions, control flow |
| **3. Semantic Analyzer** | ❌ 0% | Nothing | Everything (types, scopes, checking) |
| **4. Interpreter** | ❌ 5% | Empty file, imports defined | Evaluation engine, LLM execution |
| **5. Runtime** | ❌ 0% | Nothing | Built-in ops, LLM runtime, fuzzy resolver |
| **6. Standard Library** | ❌ 0% | Nothing | Core functions, LLM operations, domains |
| **7. Specification** | ⚠️ 40% | Basic language docs | LLM features, complete reference |

---

## What We Need to Build (Prioritized)

### Phase 1: Make it executable (Minimum Viable Language)
1. ✅ Lexer - mostly done
2. ⬜ Parser extensions - add LLM syntax
3. ⬜ Basic interpreter - evaluate simple expressions
4. ⬜ LLM runtime - actually call LLMs
5. ⬜ Fuzzy resolver - map fuzzy params to LLM settings

**Result:** Can run: `result = llm.complete("hello") with creativity: "high"`

### Phase 2: Make it useful (Real Programs)
6. ⬜ Variables and assignment
7. ⬜ Control flow (if/else)
8. ⬜ Functions
9. ⬜ Error handling
10. ⬜ Basic type checking

**Result:** Can write multi-step LLM pipelines

### Phase 3: Make it robust (Production Ready)
11. ⬜ Full semantic analyzer
12. ⬜ Standard library
13. ⬜ Domain system
14. ⬜ Complete specification
15. ⬜ Testing framework

**Result:** Production-quality language

---

## Additional Components (Nice to Have)

### 8. REPL (Read-Eval-Print Loop)
**What it does:** Interactive shell to test code

```bash
$ ndel repl
ndel> result = llm.complete("What is 2+2?")
4
ndel> result.confidence
0.95
```

**Status:** ⚠️ CLI exists but doesn't work yet

### 9. Debugger
**What it does:** Step through code, inspect variables

```bash
$ ndel debug script.ndel
Breaking at line 5
> print(variables)
> step
> continue
```

**Status:** ❌ Not started

### 10. Package Manager
**What it does:** Install and share libraries

```bash
$ ndel install code-generation
$ ndel publish my-domain
```

**Status:** ❌ Not started

### 11. Tooling
- Syntax highlighting
- IDE extensions (VS Code)
- Linters
- Formatters

**Status:** ❌ Not started

---

## Learning Resources

If you want to understand this deeper, these are great resources:

1. **"Crafting Interpreters"** by Robert Nystrom
   - Free online: https://craftinginterpreters.com/
   - Best book on building languages from scratch

2. **"Programming Language Pragmatics"** by Michael Scott
   - Comprehensive textbook

3. **"Writing An Interpreter In Go"** by Thorsten Ball
   - Very practical, hands-on

4. **"Language Implementation Patterns"** by Terence Parr
   - The ANTLR creator's guide

---

## The Beauty of Starting Simple

You don't need all of this to start! The minimal language needs:

1. ✅ Lexer (have it)
2. ✅ Parser (mostly have it)
3. ⬜ Interpreter (need to build)
4. ⬜ Runtime (need to build)

That's it! Everything else is iteration and improvement.

**Python didn't have all its features on day 1.**
**JavaScript was built in 10 days (and it shows! 😄).**
**We can build NDEL incrementally.**

---

## Next Steps

Now that you understand what a language needs, we can:

1. **Map out exactly what to build** - Break each component into tasks
2. **Build iteratively** - Start with smallest working version
3. **Test continuously** - Make sure each piece works
4. **Expand gradually** - Add features one at a time

Ready to start building? Where would you like to begin?
