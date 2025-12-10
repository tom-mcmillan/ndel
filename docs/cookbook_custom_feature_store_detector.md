# Writing a Custom Detector for a Feature-Store-Like Client

This guide shows how to extend NDEL with a custom detector for a synthetic feature-store client. The example uses the pipeline in `examples/python_feature_store_pipeline.py`.

## The example pipeline
- Defines a stub `FeatureStoreClient` with `load_features(entity_ids, feature_names)`.
- Requests features like `user_activity_7d` and `user_tenure_days`.
- Trains a simple `LogisticRegression` model on the loaded features.

## Custom detector API
- Custom detectors receive `(AnalysisContext, ast.AST)` for each node during analysis.
- `AnalysisContext` exposes `datasets`, `models`, `transformations`, `features`, and the active `NdelConfig` for mutation.

## Detector example

```python
import ast
from ndel.semantic_model import Dataset, Feature


def feature_store_detector(ctx, node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr == "load_features":
            ctx.datasets.setdefault("feature_store_inputs", Dataset(name="feature_store_inputs"))

            if len(node.args) >= 2 and isinstance(node.args[1], ast.List):
                for elt in node.args[1].elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        fname = elt.value
                        ctx.features.setdefault(
                            fname,
                            Feature(
                                name=fname,
                                description="loaded from feature store",
                                origin="feature_store_inputs",
                                data_type=None,
                            ),
                        )
```

## Wiring the detector

```python
from ndel import describe_python_source
from examples.python_feature_store_pipeline import train_purchase_model
from detectors import feature_store_detector  # your custom detector

ndel_text = describe_python_source(
    source=open("examples/python_feature_store_pipeline.py").read(),
    config=None,
    custom_detectors=[feature_store_detector],
)
print(ndel_text)
```

## Before/after NDEL output
- Before: only generic feature engineering from the feature store call.
- After (see `examples/python_feature_store_pipeline.ndel.txt`): explicit features `user_activity_7d`, `user_tenure_days` with origin `feature_store_inputs` and a model consuming them.

Keep detectors generic and lightweight; they should rely on AST patterns, not execution. This approach lets you enrich NDEL’s output for project-specific patterns like feature store reads, custom loaders, or orchestration frameworks.
