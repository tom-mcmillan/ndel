#!/usr/bin/env python3
"""
NDEL Command Line Interface
"""

import argparse
import json
import sys
from typing import Dict, Any

from . import evaluate, parse_ndel, __version__


def main():
    parser = argparse.ArgumentParser(
        description="NDEL - Non-Deterministic Expression Language"
    )
    parser.add_argument(
        "--version", 
        action="version", 
        version=f"NDEL {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Parse command
    parse_parser = subparsers.add_parser("parse", help="Parse an expression to AST")
    parse_parser.add_argument("expression", help="NDEL expression to parse")
    parse_parser.add_argument("--json", action="store_true", help="Output as JSON")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate an expression")
    eval_parser.add_argument("expression", help="NDEL expression to evaluate")
    eval_parser.add_argument("-c", "--context", help="JSON context data")
    eval_parser.add_argument("-d", "--domain", default="general", help="Domain context")
    eval_parser.add_argument("-v", "--verbose", action="store_true", help="Show details")
    
    # REPL command
    repl_parser = subparsers.add_parser("repl", help="Start interactive REPL")
    repl_parser.add_argument("-d", "--domain", default="general", help="Domain context")
    
    args = parser.parse_args()
    
    if args.command == "parse":
        parse_command(args)
    elif args.command == "eval":
        eval_command(args)
    elif args.command == "repl":
        repl_command(args)
    else:
        parser.print_help()


def parse_command(args):
    """Parse an expression and show AST."""
    try:
        ast = parse_ndel(args.expression)
        if args.json:
            # Convert to JSON representation
            print(json.dumps(ast_to_dict(ast), indent=2))
        else:
            for node in ast:
                print_ast_pretty(node)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def eval_command(args):
    """Evaluate an expression."""
    context = {}
    if args.context:
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError as e:
            print(f"Invalid JSON context: {e}", file=sys.stderr)
            sys.exit(1)
    
    try:
        result = evaluate(args.expression, context, args.domain)
        
        if args.verbose:
            print(f"Expression: {args.expression}")
            print(f"Domain: {args.domain}")
            print(f"Context: {json.dumps(context, indent=2)}")
            print("-" * 40)
        
        print(f"Result: {result.value}")
        print(f"Confidence: {result.confidence:.2f}")
        
        if result.alternatives and args.verbose:
            print("\nAlternatives:")
            for alt in result.alternatives:
                print(f"  - {alt.value} (confidence: {alt.confidence:.2f})")
                
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def repl_command(args):
    """Start an interactive REPL."""
    print(f"NDEL REPL v{__version__}")
    print(f"Domain: {args.domain}")
    print("Type 'help' for help, 'exit' to quit\n")
    
    context = {}
    
    while True:
        try:
            line = input("ndel> ").strip()
            
            if not line:
                continue
            
            if line == "exit":
                break
            elif line == "help":
                print_repl_help()
            elif line.startswith("@domain"):
                # Change domain
                parts = line.split()
                if len(parts) > 1:
                    args.domain = parts[1]
                    print(f"Domain changed to: {args.domain}")
            elif line.startswith("@context"):
                # Set context
                json_str = line[8:].strip()
                try:
                    context = json.loads(json_str)
                    print(f"Context updated: {context}")
                except json.JSONDecodeError as e:
                    print(f"Invalid JSON: {e}")
            else:
                # Evaluate expression
                try:
                    result = evaluate(line, context, args.domain)
                    print(f"= {result.value}")
                    print(f"  (confidence: {result.confidence:.2f})")
                except Exception as e:
                    print(f"Error: {e}")
                    
        except KeyboardInterrupt:
            print("\nUse 'exit' to quit")
        except EOFError:
            break
    
    print("\nGoodbye!")


def print_repl_help():
    """Print REPL help."""
    print("""
NDEL REPL Commands:
  exit                - Exit the REPL
  help                - Show this help
  @domain <name>      - Change domain context
  @context <json>     - Set context variables
  
Examples:
  age < "young"
  player is "promising"
  @domain soccer
  @context {"age": 25, "goals": 10}
    """)


def ast_to_dict(ast):
    """Convert AST to dictionary for JSON serialization."""
    # Implementation depends on AST structure
    # This is a placeholder
    return {"type": "ast", "nodes": str(ast)}


def print_ast_pretty(node, indent=0):
    """Pretty print AST node."""
    # Implementation from parser.py
    pass


if __name__ == "__main__":
    main()
