from __future__ import annotations

import ast
from typing import Dict

from ndel.config import DomainConfig, NdelConfig, PrivacyConfig
from ndel.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


class PythonAnalyzer(ast.NodeVisitor):
    """Static analyzer to build a Pipeline from Python source.

    This is intentionally conservative and focuses on common DS/ML patterns.
    Future work will expand to filters, aggregations, feature engineering,
    and metric calculation with privacy-aware handling.
    """

    def __init__(self, source: str, config: NdelConfig | None = None) -> None:
        self.source = source
        self.tree = ast.parse(source)
        self.config = config

        self.datasets: Dict[str, Dataset] = {}
        self.models: Dict[str, Model] = {}

    def analyze(self) -> Pipeline:
        self.visit(self.tree)

        domain = self.config.domain if self.config and self.config.domain else None

        datasets = list(self.datasets.values())
        models = list(self.models.values())

        if domain:
            for ds in datasets:
                alias = domain.dataset_aliases.get(ds.name)
                if alias:
                    ds.name = alias
            for model in models:
                alias = domain.model_aliases.get(model.name)
                if alias:
                    model.name = alias

        pipeline_name = domain.pipeline_name if domain and domain.pipeline_name else "python_pipeline"

        return Pipeline(
            name=pipeline_name,
            datasets=datasets,
            transformations=[],
            features=[],
            models=models,
            metrics=[],
            description=None,
        )

    def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
        if not isinstance(node.value, ast.Call):
            return

        func = node.value.func
        target = node.targets[0] if node.targets else None

        dataset_funcs = {"read_csv", "read_parquet", "read_table", "read_sql"}
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id
        elif isinstance(func, ast.Attribute):
            func_name = func.attr

        if func_name in dataset_funcs and isinstance(target, ast.Name):
            dataset_name = target.id
            self.datasets[dataset_name] = Dataset(
                name=dataset_name,
                description=f"dataset loaded via {func_name}",
                source_type=None,
                notes=[],
            )
            return

        if isinstance(func, ast.Name) and isinstance(target, ast.Name):
            model_name = target.id
            self.models[model_name] = Model(
                name=model_name,
                task="unknown",
                algorithm_family=func.id,
                inputs=[],
                target=None,
                description=None,
                hyperparameters=None,
            )

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "fit":
            if isinstance(func.value, ast.Name):
                model_var = func.value.id
                model = self.models.get(model_var)
                if model and model.description is None:
                    model.description = "trained via .fit() call"

        self.generic_visit(node)


def analyze_python_source(source: str, config: NdelConfig | None = None) -> Pipeline:
    """Analyze Python source into an NDEL Pipeline (early prototype)."""

    analyzer = PythonAnalyzer(source, config=config)
    return analyzer.analyze()


__all__ = ["PythonAnalyzer", "analyze_python_source"]
