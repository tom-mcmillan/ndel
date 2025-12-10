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
        self.features: Dict[str, Feature] = {}
        self.transformations: list[Transformation] = []

    def analyze(self) -> Pipeline:
        self.visit(self.tree)

        domain = self.config.domain if self.config and self.config.domain else None

        datasets = list(self.datasets.values())
        models = list(self.models.values())
        features = list(self.features.values())

        if domain:
            for ds in datasets:
                alias = domain.dataset_aliases.get(ds.name)
                if alias:
                    ds.name = alias
            for model in models:
                alias = domain.model_aliases.get(model.name)
                if alias:
                    model.name = alias
            for feature in features:
                alias = domain.feature_aliases.get(feature.name)
                if alias:
                    feature.name = alias

        pipeline_name = domain.pipeline_name if domain and domain.pipeline_name else "python_pipeline"

        return Pipeline(
            name=pipeline_name,
            datasets=datasets,
            transformations=self.transformations,
            features=features,
            models=models,
            metrics=[],
            description=None,
        )

    def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
        if not isinstance(node.value, ast.Call):
            self._maybe_record_transformation(node)
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
            return

        self._maybe_record_transformation(node)

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "fit":
            if isinstance(func.value, ast.Name):
                model_var = func.value.id
                model = self.models.get(model_var)
                if model and model.description is None:
                    model.description = "trained via .fit() call"
            if node.args:
                self._capture_features_from_fit(node.args[0])

        self.generic_visit(node)

    def _maybe_record_transformation(self, node: ast.Assign) -> None:
        """Detect simple pandas transformations from assignment patterns."""

        if not node.targets:
            return

        target = node.targets[0]

        # Filtering: df = df[df["x"] > 0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Subscript):
            if isinstance(node.value.value, ast.Name):
                base_name = node.value.value.id
                if base_name == target.id:
                    self.transformations.append(
                        Transformation(
                            name=f"filter_{target.id}",
                            description=f"filter rows in {target.id}",
                            kind="filter",
                            inputs=[base_name],
                            outputs=[target.id],
                        )
                    )
                    return

        # New column assignment: df["col"] = ...
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            dataset_name = target.value.id
            col_name = self._extract_subscript_key(target)
            output = col_name if col_name else dataset_name
            self.transformations.append(
                Transformation(
                    name=f"add_col_{output}",
                    description=f"add or update column {output}",
                    kind="feature_engineering",
                    inputs=[dataset_name],
                    outputs=[output],
                )
            )
            if col_name:
                self._add_feature(col_name, origin=dataset_name)
            return

        # Aggregation: df = df.groupby(...).agg(...)
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
            call = node.value
            if isinstance(call.func, ast.Attribute) and call.func.attr in {"agg", "aggregate"}:
                base = call.func.value
                dataset_name = self._base_name_from_groupby(base)
                if dataset_name:
                    self.transformations.append(
                        Transformation(
                            name=f"aggregate_{target.id}",
                            description=f"aggregate {dataset_name}",
                            kind="aggregation",
                            inputs=[dataset_name],
                            outputs=[target.id],
                        )
                    )
                    return

            # Join/Merge: df = df.merge(...)
            if isinstance(call.func, ast.Attribute) and call.func.attr == "merge":
                base = call.func.value
                if isinstance(base, ast.Name):
                    other = None
                    if call.args:
                        first_arg = call.args[0]
                        other = self._name_from_node(first_arg)
                    self.transformations.append(
                        Transformation(
                            name=f"merge_{target.id}",
                            description=f"merge {base.id} with {other or 'other'}",
                            kind="join",
                            inputs=[n for n in [base.id, other] if n],
                            outputs=[target.id],
                        )
                    )
                    return

    def _extract_subscript_key(self, node: ast.Subscript) -> str | None:
        """Get the string key from a subscript like df["col"]."""

        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return key.value
        if isinstance(key, ast.Index):  # type: ignore[attr-defined]
            inner = key.value  # type: ignore[assignment]
            if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                return inner.value
        return None

    def _base_name_from_groupby(self, node: ast.AST) -> str | None:
        """Extract dataset name from a groupby chain."""

        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "groupby":
            if isinstance(node.func.value, ast.Name):
                return node.func.value.id
        if isinstance(node, ast.Attribute) and node.attr == "groupby":
            if isinstance(node.value, ast.Name):
                return node.value.id
        return None

    def _name_from_node(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        return None

    def _add_feature(self, name: str, origin: str | None = None, description: str | None = None) -> None:
        if name in self.features:
            return
        self.features[name] = Feature(
            name=name,
            description=description or "derived feature from column assignment",
            origin=origin,
            data_type=None,
        )

    def _capture_features_from_fit(self, arg: ast.AST) -> None:
        """Capture feature names from model.fit arguments like df[["a", "b"]]."""

        if isinstance(arg, ast.Subscript):
            cols = self._extract_columns_from_subscript(arg)
            origin = arg.value.id if isinstance(arg.value, ast.Name) else None
            for col in cols:
                self._add_feature(col, origin=origin, description="feature used in model training")

    def _extract_columns_from_subscript(self, node: ast.Subscript) -> list[str]:
        cols: list[str] = []
        slice_node = node.slice
        if isinstance(slice_node, ast.List):
            for elt in slice_node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    cols.append(elt.value)
        elif isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
            cols.append(slice_node.value)
        elif isinstance(slice_node, ast.Index):  # type: ignore[attr-defined]
            inner = slice_node.value  # type: ignore[assignment]
            if isinstance(inner, ast.List):
                for elt in inner.elts:
                    if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                        cols.append(elt.value)
            elif isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                cols.append(inner.value)
        return cols


def analyze_python_source(source: str, config: NdelConfig | None = None) -> Pipeline:
    """Analyze Python source into an NDEL Pipeline (early prototype)."""

    analyzer = PythonAnalyzer(source, config=config)
    return analyzer.analyze()


__all__ = ["PythonAnalyzer", "analyze_python_source"]
