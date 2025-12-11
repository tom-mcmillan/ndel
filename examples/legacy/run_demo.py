#!/usr/bin/env python3
"""NDEL Demo - Parse and visualize a complex query"""

import sys
sys.path.insert(0, '/Users/thomasmcmillan/projects/ndel/reference/python')

from parser import parse_ndel, print_ast

# Read the NDEL code
with open('/Users/thomasmcmillan/projects/ndel/demo.ndel', 'r') as f:
    ndel_code = f.read()

print("╔" + "═" * 70 + "╗")
print("║" + " " * 20 + "NDEL PARSER DEMO" + " " * 34 + "║")
print("╚" + "═" * 70 + "╝")
print()
print("NDEL Query:")
print("─" * 72)
print(ndel_code)
print("─" * 72)

# Parse it
print("\n🔍 Parsing NDEL expression...\n")
try:
    ast = parse_ndel(ndel_code)

    print("✓ Parse successful!")
    print("\n📊 Abstract Syntax Tree (AST):")
    print("─" * 72)
    for node in ast:
        print_ast(node)
    print("─" * 72)

    # Count fuzzy predicates
    def count_fuzzy(node, counter={'count': 0}):
        from parser import FuzzyPredicate
        if isinstance(node, FuzzyPredicate):
            counter['count'] += 1
        for attr in ['left', 'right', 'subject', 'operand', 'condition', 'true_branch', 'false_branch']:
            if hasattr(node, attr):
                child = getattr(node, attr)
                if child:
                    count_fuzzy(child, counter)
        return counter['count']

    fuzzy_count = sum(count_fuzzy(node, {'count': 0}) for node in ast)

    print(f"\n📈 Statistics:")
    print(f"  • AST nodes: {len(ast)}")
    print(f"  • Fuzzy predicates: {fuzzy_count}")
    print(f"\n💡 This query mixes deterministic logic (salary < 50000)")
    print(f"   with fuzzy terms that get resolved based on domain context!")

except Exception as e:
    print(f"\n✗ Parse error: {e}")
    import traceback
    traceback.print_exc()
