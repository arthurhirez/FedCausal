"""Composite FL pipeline: preprocessing -> federated training -> drift."""
from kedro.pipeline import Pipeline

from fedwater.pipelines import drift_detection, fl_preprocessing, fl_training


def create_pipeline(**kwargs) -> Pipeline:
    return (fl_preprocessing.create_pipeline()
            + fl_training.create_pipeline()
            + drift_detection.create_pipeline())
