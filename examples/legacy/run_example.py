#!/usr/bin/env python3
"""Run a simple NDEL example"""

import sys
sys.path.insert(0, '/Users/thomasmcmillan/projects/ndel/reference/python')

from parser import parse_ndel, print_ast

# Read the NDEL code
with open('/Users/thomasmcmillan/projects/ndel/test_example.ndel', 'r') as f:
    ndel_code = f.read()

print("NDEL Code:")
print("=" * 60)
print(ndel_code)
print("=" * 60)

# Parse it
print("\nParsing...")
try:
    ast = parse_ndel(ndel_code)
    print("\nGenerated AST:")
    print("-" * 60)
    for node in ast:
        print_ast(node)
    print("-" * 60)
    print("\n✓ Successfully parsed!")
except Exception as e:
    print(f"\n✗ Error: {e}")
    import traceback
    traceback.print_exc()
