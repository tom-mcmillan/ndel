from __future__ import annotations

import ast
from typing import Dict, List, Tuple

from src.config.core import DomainConfig, NdelConfig, PrivacyConfig
from src.pipeline.semantic_model import Dataset, Feature, Metric, Model, Pipeline, Transformation


class AnalysisContext:
    """Shared mutable context passed to custom detectors."""

    def __init__(
        self,
        datasets: Dict[str, Dataset],
        models: Dict[str, Model],
        transformations: list[Transformation],
        features: Dict[str, Feature],
        config: NdelConfig | None,
    ) -> None:
        self.datasets = datasets
        self.models = models
        self.transformations = transformations
        self.features = features
        self.config = config


class PythonAnalyzer(ast.NodeVisitor):
    """Static analyzer to build a Pipeline from Python source.

    This is intentionally conservative and focuses on common DS/ML patterns.
    Future work will expand to filters, aggregations, feature engineering,
    and metric calculation with privacy-aware handling.
    """

    def __init__(
        self,
        source: str,
        config: NdelConfig | None = None,
        custom_detectors: list[callable] | None = None,
    ) -> None:
        self.source = source
        self.tree = ast.parse(source)
        self.config = config
        self.custom_detectors = custom_detectors or []

        self.datasets: Dict[str, Dataset] = {}
        self.models: Dict[str, Model] = {}
        self.features: Dict[str, Feature] = {}
        self.metrics: list[Metric] = []
        self.transformations: list[Transformation] = []
        self.current_origin_for_dataset: Dict[str, str] = {}

        self.context = AnalysisContext(
            datasets=self.datasets,
            models=self.models,
            transformations=self.transformations,
            features=self.features,
            config=self.config,
        )

    def visit(self, node: ast.AST):  # type: ignore[override]
        for detector in self.custom_detectors:
            detector(self.context, node)
        return super().visit(node)

    def analyze(self) -> Pipeline:
        self.visit(self.tree)

        domain = self.config.domain if self.config and self.config.domain else None

        datasets = list(self.datasets.values())
        models = list(self.models.values())
        features = list(self.features.values())
        metrics = list(self.metrics)

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
            metrics=metrics,
            description=None,
        )

    def visit_Assign(self, node: ast.Assign) -> None:  # type: ignore[override]
        if not node.targets:
            return

        target = node.targets[0]

        if isinstance(node.value, ast.Call):
            func = node.value.func
            dataset_funcs = {"read_csv", "read_parquet", "read_table", "read_sql"}
            func_name = None
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr

            if func_name in dataset_funcs and isinstance(target, ast.Name):
                dataset_name = target.id
                source = self._extract_source_from_call(node.value)
                self.datasets[dataset_name] = Dataset(
                    name=dataset_name,
                    source=source,
                    description=f"dataset loaded via {func_name}",
                    source_type=None,
                    notes=[],
                )
                self.current_origin_for_dataset[dataset_name] = dataset_name
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
                # Fall through to handle special sklearn pipeline structures if applicable

            if func_name in {"Pipeline", "ColumnTransformer"}:
                self._capture_sklearn_structure(node.value)
                return

        self._record_assignment_transformation(target, node.value)

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # type: ignore[override]
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "fit":
            if isinstance(func.value, ast.Name):
                model_var = func.value.id
                model = self.models.get(model_var)
                if model and model.description is None:
                    model.description = "trained via .fit() call"
            fit_inputs, fit_origin = None, None
            if node.args:
                fit_inputs, fit_origin = self._capture_features_from_fit(node.args[0])
            if isinstance(func.value, ast.Name):
                model_var = func.value.id
                model = self.models.get(model_var)
                if model:
                    if fit_inputs:
                        model.inputs = fit_inputs
                    elif fit_origin:
                        model.inputs = [fit_origin]

        # Detect inline sklearn Pipeline/ColumnTransformer creations
        if isinstance(func, (ast.Name, ast.Attribute)):
            callee = func.id if isinstance(func, ast.Name) else func.attr
            if callee in {"Pipeline", "ColumnTransformer"}:
                self._capture_sklearn_structure(node)

        self._capture_metrics(node)

        self.generic_visit(node)

    def _record_assignment_transformation(self, target: ast.AST, value: ast.AST) -> None:
        # Column assignment: df["col"] = ...
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            dataset_name = target.value.id
            col_name = self._extract_subscript_key(target)
            output = col_name if col_name else dataset_name
            self._add_transformation(
                kind="feature_engineering",
                dataset=dataset_name,
                description=f"add or update column {output}",
                inputs=[self.current_origin_for_dataset.get(dataset_name, dataset_name)],
                outputs=[output],
            )
            if col_name:
                self._add_feature(col_name, origin=self.current_origin_for_dataset.get(dataset_name, dataset_name))
            self.current_origin_for_dataset[dataset_name] = self.transformations[-1].name
            return

        if not isinstance(target, ast.Name):
            return

        target_name = target.id

        # Filters using boolean indexing or .loc
        base_name = self._base_name_from_subscript(value)
        if base_name and base_name == target_name:
            self._add_transformation(
                kind="filter",
                dataset=base_name,
                description=f"filter rows in {base_name}",
                inputs=[self.current_origin_for_dataset.get(base_name, base_name)],
                outputs=[target_name],
            )
            self.current_origin_for_dataset[target_name] = self.transformations[-1].name
            return

        # Calls and chained operations
        if isinstance(value, ast.Call):
            self._handle_call_assignment(target_name, value)

    def _handle_call_assignment(self, target_name: str, call: ast.Call) -> None:
        base_name, ops = self._extract_call_chain(call)

        # Handle pd.concat as a special top-level call
        if base_name == "pd" and ops and ops[-1][0] == "concat":
            inputs = self._extract_names_from_iterable(ops[-1][1].args[0] if ops[-1][1].args else None)
            self._add_transformation(
                kind="join",
                dataset=target_name,
                description="concatenate dataframes",
                inputs=inputs,
                outputs=[target_name],
            )
            return

        current_input = base_name or target_name
        current_origin = self.current_origin_for_dataset.get(current_input, current_input)

        for attr, call_node in ops:
            if attr == "query":
                condition = None
                if call_node.args and isinstance(call_node.args[0], ast.Constant) and isinstance(call_node.args[0].value, str):
                    condition = call_node.args[0].value
                desc = f"filter rows{f' where {condition}' if condition else ''}".strip()
                self._add_transformation(
                    kind="filter",
                    dataset=target_name,
                    description=desc,
                    inputs=[current_origin] if current_origin else [],
                    outputs=[target_name],
                )
                current_origin = self.current_origin_for_dataset.get(target_name, target_name)
            elif attr == "assign":
                cols = self._extract_assign_columns(call_node)
                desc = f"add columns {', '.join(cols)}" if cols else "add columns"
                self._add_transformation(
                    kind="feature_engineering",
                    dataset=target_name,
                    description=desc,
                    inputs=[current_origin] if current_origin else [],
                    outputs=cols or [target_name],
                )
                for col in cols:
                    self._add_feature(col, origin=current_origin)
                current_origin = self.current_origin_for_dataset.get(target_name, target_name)
            elif attr in {"agg", "aggregate", "sum", "mean", "count", "apply"}:
                desc = f"{attr} aggregation"
                self._add_transformation(
                    kind="aggregation",
                    dataset=target_name,
                    description=desc,
                    inputs=[current_origin] if current_origin else [],
                    outputs=[target_name],
                )
                current_origin = self.current_origin_for_dataset.get(target_name, target_name)
            elif attr == "merge":
                other = None
                if call_node.args:
                    other = self._name_from_node(call_node.args[0])
                inputs = [n for n in [current_origin, other] if n]
                desc = f"merge {inputs[0]} with {other or 'other'}" if inputs else "merge"
                self._add_transformation(
                    kind="join",
                    dataset=target_name,
                    description=desc,
                    inputs=inputs,
                    outputs=[target_name],
                )
                current_origin = self.current_origin_for_dataset.get(target_name, target_name)
            elif attr == "concat":
                inputs = []
                if call_node.args:
                    inputs = self._extract_names_from_iterable(call_node.args[0])
                self._add_transformation(
                    kind="join",
                    dataset=target_name,
                    description="concatenate dataframes",
                    inputs=inputs,
                    outputs=[target_name],
                )
                current_origin = self.current_origin_for_dataset.get(target_name, target_name)

        if not ops and isinstance(call.func, ast.Name):
            self._add_transformation(
                kind="other",
                dataset=target_name,
                description=f"call {call.func.id}",
                inputs=[current_origin] if current_origin else [],
                outputs=[target_name],
            )

    def _extract_call_chain(self, node: ast.Call) -> tuple[str | None, List[Tuple[str, ast.Call]]]:
        ops: List[Tuple[str, ast.Call]] = []
        current: ast.AST = node
        while isinstance(current, ast.Call) and isinstance(current.func, ast.Attribute):
            ops.append((current.func.attr, current))
            current = current.func.value
        base_name = current.id if isinstance(current, ast.Name) else None
        ops.reverse()
        return base_name, ops

    def _add_transformation(
        self,
        *,
        kind: str,
        dataset: str | None,
        description: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        name_override: str | None = None,
    ) -> None:
        name = name_override or f"{(dataset or 'trans')}_{kind}_{len(self.transformations) + 1}"
        trans_inputs = inputs or ([] if dataset is None else [dataset])
        trans_outputs = outputs or ([dataset] if dataset else [])
        transformation = Transformation(
            name=name,
            description=description,
            kind=kind,
            inputs=trans_inputs,
            outputs=trans_outputs,
        )
        self.transformations.append(transformation)

        if dataset:
            self.current_origin_for_dataset[dataset] = transformation.name

    def _capture_metrics(self, node: ast.Call) -> None:
        func = node.func
        metric_name = None
        if isinstance(func, ast.Name):
            metric_name = func.id
        elif isinstance(func, ast.Attribute):
            metric_name = func.attr

        if not metric_name:
            return

        metric_info = self._metric_info(metric_name)
        if not metric_info:
            return

        name, higher_is_better = metric_info
        dataset_guess = None
        for arg in node.args[:2]:
            if isinstance(arg, ast.Name):
                if "val" in arg.id:
                    dataset_guess = "validation"
                elif "test" in arg.id:
                    dataset_guess = "test"

        description = f"{name} evaluation"
        self.metrics.append(
            Metric(
                name=name,
                description=description,
                dataset=dataset_guess,
                higher_is_better=higher_is_better,
            )
        )

    def _metric_info(self, name: str) -> tuple[str, bool] | None:
        metrics_higher = {
            "roc_auc_score": True,
            "accuracy_score": True,
            "f1_score": True,
            "precision_score": True,
            "recall_score": True,
            "r2_score": True,
            "log_loss": False,
            "mean_squared_error": False,
            "mean_absolute_error": False,
        }
        for key, hib in metrics_higher.items():
            if name == key:
                return key, hib
        return None

    def _capture_sklearn_structure(self, call: ast.Call) -> None:
        if isinstance(call.func, ast.Name) and call.func.id == "Pipeline":
            steps = self._extract_steps_kw(call)
            for idx, (step_name, estimator_name) in enumerate(steps):
                if idx < len(steps) - 1:
                    self._add_transformation(
                        kind="feature_engineering",
                        dataset=None,
                        description=f"pipeline step {step_name} using {estimator_name}",
                        inputs=[],
                        outputs=[step_name],
                        name_override=f"{step_name}_feature_engineering" if step_name else None,
                    )
                else:
                    model_name = step_name or estimator_name or f"model_{len(self.models)+1}"
                    self.models[model_name] = Model(
                        name=model_name,
                        task="unknown",
                        algorithm_family=estimator_name or "pipeline_estimator",
                        inputs=[],
                        target=None,
                        description="model from sklearn Pipeline",
                        hyperparameters=None,
                    )
        if isinstance(call.func, ast.Name) and call.func.id == "ColumnTransformer":
            transformers = self._extract_transformers_kw(call)
            for name, estimator_name, cols in transformers:
                desc = f"column transformer {name} on {', '.join(cols) if cols else 'columns'}"
                self._add_transformation(
                    kind="feature_engineering",
                    dataset=None,
                    description=desc,
                    inputs=cols,
                    outputs=[name],
                    name_override=f"{name}_feature_engineering" if name else None,
                )

    def _extract_steps_kw(self, call: ast.Call) -> list[tuple[str, str | None]]:
        steps: list[tuple[str, str | None]] = []
        iterables = []
        if call.args:
            iterables.append(call.args[0])
        for kw in call.keywords:
            if kw.arg == "steps" and isinstance(kw.value, (ast.List, ast.Tuple)):
                iterables.append(kw.value)
        for iterable in iterables:
            if not isinstance(iterable, (ast.List, ast.Tuple)):
                continue
            for elt in iterable.elts:
                if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) >= 2:
                    name_node = elt.elts[0]
                    estimator_node = elt.elts[1]
                    name = name_node.value if isinstance(name_node, ast.Constant) else None
                    estimator = self._name_from_node(estimator_node)
                    steps.append((name or "", estimator))
        return steps

    def _extract_transformers_kw(self, call: ast.Call) -> list[tuple[str, str | None, list[str]]]:
        transformers: list[tuple[str, str | None, list[str]]] = []
        # transformers can be passed positionally or via keyword
        args_iterables = []
        if call.args:
            args_iterables.append(call.args[0])
        for kw in call.keywords:
            if kw.arg is None and isinstance(kw.value, (ast.List, ast.Tuple)):
                args_iterables.append(kw.value)
            elif kw.arg == "transformers" and isinstance(kw.value, (ast.List, ast.Tuple)):
                args_iterables.append(kw.value)
        for iterable in args_iterables:
            for elt in iterable.elts:
                if isinstance(elt, (ast.List, ast.Tuple)) and len(elt.elts) >= 3:
                    name_node, estimator_node, cols_node = elt.elts[:3]
                    name = name_node.value if isinstance(name_node, ast.Constant) else ""
                    estimator = self._name_from_node(estimator_node)
                    cols = self._extract_names_from_iterable(cols_node)
                    transformers.append((name, estimator, cols))
        return transformers

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

    def _base_name_from_subscript(self, node: ast.AST) -> str | None:
        if not isinstance(node, ast.Subscript):
            return None
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute) and node.value.attr == "loc" and isinstance(node.value.value, ast.Name):
            return node.value.value.id
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
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            return node.func.id
        return None

    def _extract_assign_columns(self, call: ast.Call) -> list[str]:
        cols: list[str] = []
        for kw in call.keywords:
            if kw.arg:
                cols.append(kw.arg)
        return cols

    def _extract_names_from_iterable(self, node: ast.AST | None) -> list[str]:
        if node is None:
            return []
        names: list[str] = []
        if isinstance(node, ast.List):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.append(elt.value)
                else:
                    name = self._name_from_node(elt)
                    if name:
                        names.append(name)
        elif isinstance(node, ast.Tuple):
            for elt in node.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.append(elt.value)
                else:
                    name = self._name_from_node(elt)
                    if name:
                        names.append(name)
        else:
            name = self._name_from_node(node)
            if name:
                names.append(name)
        return names

    def _add_feature(self, name: str, origin: str | None = None, description: str | None = None) -> None:
        if name in self.features:
            return
        if origin is None and self.current_origin_for_dataset:
            origin = next(iter(self.current_origin_for_dataset.values()))
        self.features[name] = Feature(
            name=name,
            description=description or "derived feature from column assignment",
            origin=origin,
            data_type=None,
        )

    def _capture_features_from_fit(self, arg: ast.AST) -> tuple[list[str], str | None]:
        """Capture feature names from model.fit arguments like df[["a", "b"]]."""

        feature_names: list[str] = []
        origin: str | None = None
        if isinstance(arg, ast.Subscript):
            cols = self._extract_columns_from_subscript(arg)
            origin = arg.value.id if isinstance(arg.value, ast.Name) else None
            origin = self.current_origin_for_dataset.get(origin, origin)
            for col in cols:
                self._add_feature(col, origin=origin, description="feature used in model training")
            feature_names = cols
        return feature_names, origin

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

    def _extract_source_from_call(self, call: ast.Call) -> str | None:
        # Look for first string literal arg as source (e.g., file path, table name)
        for arg in call.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                return arg.value
        return None


def analyze_python_source(
    source: str,
    config: NdelConfig | None = None,
    custom_detectors: list[callable] | None = None,
) -> Pipeline:
    """Analyze Python source into an NDEL Pipeline (early prototype)."""

    analyzer = PythonAnalyzer(source, config=config, custom_detectors=custom_detectors)
    return analyzer.analyze()


__all__ = ["PythonAnalyzer", "analyze_python_source"]
