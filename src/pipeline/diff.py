from __future__ import annotations

from dataclasses import dataclass, field

from src.pipeline.semantic_model import Pipeline


@dataclass
class PipelineDiff:
    datasets_added: list[str] = field(default_factory=list)
    datasets_removed: list[str] = field(default_factory=list)
    transformations_added: list[str] = field(default_factory=list)
    transformations_removed: list[str] = field(default_factory=list)
    features_added: list[str] = field(default_factory=list)
    features_removed: list[str] = field(default_factory=list)
    models_added: list[str] = field(default_factory=list)
    models_removed: list[str] = field(default_factory=list)
    metrics_added: list[str] = field(default_factory=list)
    metrics_removed: list[str] = field(default_factory=list)


def diff_pipelines(old: Pipeline, new: Pipeline) -> PipelineDiff:
    def names(items):
        return {getattr(i, "name", str(i)) for i in items}

    old_ds, new_ds = names(old.datasets), names(new.datasets)
    old_tf, new_tf = names(old.transformations), names(new.transformations)
    old_feat, new_feat = names(old.features), names(new.features)
    old_models, new_models = names(old.models), names(new.models)
    old_metrics, new_metrics = names(old.metrics), names(new.metrics)

    return PipelineDiff(
        datasets_added=sorted(new_ds - old_ds),
        datasets_removed=sorted(old_ds - new_ds),
        transformations_added=sorted(new_tf - old_tf),
        transformations_removed=sorted(old_tf - new_tf),
        features_added=sorted(new_feat - old_feat),
        features_removed=sorted(old_feat - new_feat),
        models_added=sorted(new_models - old_models),
        models_removed=sorted(old_models - new_models),
        metrics_added=sorted(new_metrics - old_metrics),
        metrics_removed=sorted(old_metrics - new_metrics),
    )


__all__ = ["PipelineDiff", "diff_pipelines"]
