from kedro.pipeline import Pipeline, node

from .nodes import add_measurement_noise, extract_sensor_series, package_client_datasets


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                extract_sensor_series,
                inputs=["pressures", "flows", "params:sensors"],
                outputs="sensor_series_true",
                name="extract_sensor_series",
            ),
            node(
                add_measurement_noise,
                inputs=["sensor_series_true", "params:noise", "params:seed"],
                outputs="sensor_series",
                name="add_measurement_noise",
            ),
            node(
                package_client_datasets,
                inputs=["sensor_series", "params:time", "params:start_date"],
                outputs="client_datasets",
                name="package_client_datasets",
            ),
        ]
    )
