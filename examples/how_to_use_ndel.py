"""
Examples of how to use NDEL in your Python code
"""

# ============================================================================
# METHOD 1: Install as a Package
# ============================================================================
"""
# From the ndel directory, install it:
pip install -e .

# Or install directly from GitHub:
pip install git+https://github.com/yourusername/ndel.git

# Then use it in your code:
"""

import ndel

# Simple evaluation
result = ndel.evaluate('age < "young"', {"age": 22}, domain="soccer")
print(f"Result: {result.value}, Confidence: {result.confidence}")


# ============================================================================
# METHOD 2: Direct Import (Development)
# ============================================================================
"""
# Add the path to your Python path
import sys
sys.path.append('/path/to/ndel/reference/python')
"""

from ndel import parse_ndel, NDELInterpreter

# Parse an expression
ast = parse_ndel('player is "promising"')

# Create interpreter and evaluate
interpreter = NDELInterpreter(domain="soccer")
result = interpreter.evaluate('salary < 100000 && potential is "high"', {
    "salary": 75000,
    "potential": 82
})


# ============================================================================
# METHOD 3: Embed in Your Application
# ============================================================================

class SoccerAnalytics:
    """Example of embedding NDEL in a soccer analytics app."""
    
    def __init__(self):
        self.ndel = ndel.NDELInterpreter(domain="soccer")
        
    def find_players(self, expression: str, player_data: list):
        """Find players matching an NDEL expression."""
        matches = []
        
        for player in player_data:
            result = self.ndel.evaluate(expression, player)
            if result.value and result.confidence > 0.7:
                matches.append({
                    "player": player,
                    "confidence": result.confidence
                })
        
        return sorted(matches, key=lambda x: x["confidence"], reverse=True)

# Usage
analytics = SoccerAnalytics()
players = [
    {"name": "John", "age": 21, "position": "ST", "goals": 15},
    {"name": "Mike", "age": 28, "position": "CM", "goals": 5},
    {"name": "Sam", "age": 19, "position": "ST", "goals": 8},
]

young_strikers = analytics.find_players(
    'position == "ST" && age < "young"',
    players
)


# ============================================================================
# METHOD 4: REST API Service
# ============================================================================

from flask import Flask, request, jsonify
import ndel

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate_expression():
    """REST endpoint for NDEL evaluation."""
    data = request.json
    
    try:
        result = ndel.evaluate(
            data['expression'],
            data.get('context', {}),
            data.get('domain', 'general')
        )
        
        return jsonify({
            'success': True,
            'value': result.value,
            'confidence': result.confidence,
            'alternatives': [
                {'value': alt.value, 'confidence': alt.confidence}
                for alt in result.alternatives
            ]
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# Run with: flask run


# ============================================================================
# METHOD 5: Jupyter Notebook Integration
# ============================================================================
"""
# In a Jupyter notebook:

# Install
!pip install /path/to/ndel

# Create magic command
from IPython.core.magic import register_line_magic
import ndel

@register_line_magic
def ndel_eval(line):
    '''Evaluate NDEL expression in notebook'''
    result = ndel.evaluate(line)
    print(f"Result: {result.value}")
    print(f"Confidence: {result.confidence:.2%}")
    return result

# Use in notebook:
%ndel_eval age < "young"
"""


# ============================================================================
# METHOD 6: Command Line Tool
# ============================================================================
"""
# After installation, use from command line:

# Parse an expression
ndel parse 'player is "young"'

# Evaluate with context
ndel eval 'age < "young"' --context '{"age": 22}' --domain soccer

# Start interactive REPL
ndel repl --domain soccer

# Use in shell scripts
result=$(ndel eval 'temperature > "hot"' --context '{"temperature": 35}')
"""


# ============================================================================
# METHOD 7: Database Integration
# ============================================================================

import sqlite3
import ndel

class NDELDatabase:
    """Example of using NDEL with a database."""
    
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.ndel = ndel.NDELInterpreter(domain="soccer")
        
    def query_with_fuzzy(self, table: str, expression: str):
        """Query database using NDEL expressions."""
        
        # Get all rows
        cursor = self.conn.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        
        # Filter using NDEL
        results = []
        for row in rows:
            context = dict(zip(columns, row))
            result = self.ndel.evaluate(expression, context)
            
            if result.value:
                results.append({
                    'data': context,
                    'confidence': result.confidence
                })
        
        return results

# Usage
db = NDELDatabase("players.db")
young_talents = db.query_with_fuzzy(
    "players",
    'age < "young" && potential is "high"'
)


# ============================================================================
# METHOD 8: Configuration Files
# ============================================================================
"""
# Create NDEL configuration files (config.ndel):

@domain("soccer")

rules:
  transfer_policy: age < "young" && potential is "high" && price is "reasonable"
  starting_eleven: form is "good" && fitness is "match ready"
  youth_promotion: age < 21 && development is "ahead of schedule"
"""

import yaml
import ndel

def load_ndel_config(path):
    """Load NDEL rules from configuration file."""
    with open(path) as f:
        content = f.read()
    
    # Parse rules
    rules = {}
    current_domain = "general"
    
    for line in content.split('\n'):
        if line.startswith('@domain'):
            current_domain = line.split('"')[1]
        elif ':' in line:
            key, expr = line.split(':', 1)
            rules[key.strip()] = {
                'expression': expr.strip(),
                'domain': current_domain
            }
    
    return rules

# Use configuration
config = load_ndel_config('config.ndel')
transfer_rule = config['transfer_policy']
result = ndel.evaluate(
    transfer_rule['expression'],
    player_data,
    transfer_rule['domain']
)


# ============================================================================
# METHOD 9: Testing with NDEL
# ============================================================================

import pytest
import ndel

class TestPlayerFilters:
    """Test player filtering rules."""
    
    def test_young_player_detection(self):
        """Test that young players are correctly identified."""
        
        test_cases = [
            ({"age": 19}, True, 0.85),   # Should match with high confidence
            ({"age": 22}, True, 0.85),   # Should match
            ({"age": 24}, False, 0.85),  # Should not match
            ({"age": 30}, False, 0.95),  # Definitely not young
        ]
        
        for context, expected_match, min_confidence in test_cases:
            result = ndel.evaluate('age < "young"', context, "soccer")
            assert result.value == expected_match
            assert result.confidence >= min_confidence


# ============================================================================
# METHOD 10: Real-time Processing
# ============================================================================

import asyncio
import ndel

class NDELStreamProcessor:
    """Process streaming data with NDEL."""
    
    def __init__(self, expression: str, domain: str = "general"):
        self.expression = expression
        self.interpreter = ndel.NDELInterpreter(domain)
        
    async def process_stream(self, data_stream):
        """Process streaming data."""
        async for data in data_stream:
            result = self.interpreter.evaluate(self.expression, data)
            
            if result.value and result.confidence > 0.75:
                yield {
                    'data': data,
                    'match': True,
                    'confidence': result.confidence
                }

# Usage
async def main():
    processor = NDELStreamProcessor(
        'temperature is "anomalous" && pressure shows "increasing"',
        domain="sensors"
    )
    
    async for alert in processor.process_stream(sensor_data_stream):
        print(f"Alert: {alert}")
