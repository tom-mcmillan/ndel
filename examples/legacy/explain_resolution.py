#!/usr/bin/env python3
"""Show how NDEL fuzzy terms could be resolved in the soccer domain"""

print("╔" + "═" * 78 + "╗")
print("║" + " " * 20 + "NDEL FUZZY RESOLUTION EXAMPLE" + " " * 28 + "║")
print("╚" + "═" * 78 + "╝")
print()

ndel_query = '''
@domain("soccer")

(position is "striker" || position is "attacking midfielder")
  && age < "young prospect"
  && potential is "high"
  && current_form shows "improving"
  && salary < 50000
  && injury_record is "acceptable"
'''

print("📝 ORIGINAL NDEL QUERY:")
print("─" * 80)
print(ndel_query)
print("─" * 80)

print("\n🔄 FUZZY TERM RESOLUTION (Soccer Domain):\n")

resolutions = [
    ('position is "striker"', 'position IN ["ST", "CF"]', 0.95),
    ('position is "attacking midfielder"', 'position IN ["CAM", "AM"]', 0.93),
    ('age < "young prospect"', 'age < 22', 0.88),
    ('potential is "high"', 'potential_rating > 78', 0.90),
    ('current_form shows "improving"', '(form_last_5_games > form_previous_5_games)', 0.85),
    ('injury_record is "acceptable"', 'days_injured_last_season < 45', 0.80),
]

for original, resolved, confidence in resolutions:
    print(f"  {original}")
    print(f"    ➜  {resolved}")
    print(f"    📊 Confidence: {confidence:.0%}\n")

print("─" * 80)
print("\n✨ FINAL RESOLVED SQL-LIKE QUERY:\n")

final_query = '''
SELECT * FROM players
WHERE (
  (position IN ["ST", "CF"] OR position IN ["CAM", "AM"])
  AND age < 22
  AND potential_rating > 78
  AND (form_last_5_games > form_previous_5_games)
  AND salary < 50000
  AND days_injured_last_season < 45
)
'''

print(final_query)
print("─" * 80)
print("\n💡 Key Insight: Same NDEL query with @domain(\"finance\") would resolve")
print("   'young prospect' differently (e.g., startup < 3 years old)\n")
