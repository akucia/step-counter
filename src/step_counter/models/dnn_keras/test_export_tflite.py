import keras
from click.testing import CliRunner

from step_counter.models.dnn_keras.export_tflite import main as export_tflite_main


def test_main(dummy_train_and_test_data, tmp_path):
    model = keras.models.Sequential(
        [
            keras.layers.Dense(32, input_shape=(9,), activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.save(tmp_path / "model")

    export_path = tmp_path / "export"
    export_path.mkdir(parents=True, exist_ok=True)
    _, test_path = dummy_train_and_test_data

    runner = CliRunner()
    result = runner.invoke(
        export_tflite_main,
        [
            str(tmp_path / "model"),
            str(export_path),
            str(test_path),
        ],
    )
    assert result.exit_code == 0, result.output
    output_files = list(export_path.glob("*.tflite"))
    output_files = [str(file.relative_to(tmp_path)) for file in output_files]
    assert len(output_files) == 3
    assert output_files == [
        "export/model_quantized.tflite",
        "export/model_optimized.tflite",
        "export/model.tflite",
    ]
