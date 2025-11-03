# NDEL - Non-Deterministic Expression Language

NDEL is a revolutionary expression language that combines deterministic structure with non-deterministic value resolution, enabling natural language-like queries that are interpreted into precise computational expressions.

## Key Features

- **Mixed Deterministic/Non-Deterministic Expressions**: Write `age < "young"` and let the system interpret "young" based on context
- **Confidence Scoring**: Every interpretation has an associated confidence score
- **Domain Adaptable**: Same expression means different things in different domains
- **Learning Capable**: Improves interpretations based on usage and feedback
- **CEL-Inspired**: Built on solid foundations from Google's Common Expression Language

## Quick Example
```ndel
// Soccer domain
where player is "promising young striker" 
  and performance shows "improving trend"
  and salary is "reasonable"

// Resolves to:
where (age < 23 AND position IN ['ST', 'CF'] AND potential_rating > 0.75)
  and (form_last_5 > form_previous_5)  
  and (salary < team_average * 0.8)
```

## Repository Structure

- `doc/` - Language specifications and documentation
- `grammar/` - Formal grammar definition (ANTLR4)
- `proto/` - Protocol buffer definitions
- `conformance/` - Test suite
- `examples/` - Example NDEL programs
- `reference/` - Reference implementation

## Status

⚠️ **UNDER ACTIVE DEVELOPMENT** - This is a language specification in progress.

## License

Apache 2.0 (same as CEL)
