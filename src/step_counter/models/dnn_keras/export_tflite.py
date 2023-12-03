from pathlib import Path

import click
import keras
import tensorflow as tf

from step_counter.datasets import load_data_as_dataframe


@click.command()
@click.argument("model_path", type=click.Path(path_type=Path))
@click.argument("export_path", type=click.Path(path_type=Path))
@click.argument("data_path", type=click.Path(exists=True, path_type=Path))
def main(model_path: Path, export_path: Path, data_path: Path):
    """Export Keras model to TensorFlow Lite format

    Args:
        model_path: Path to trained model
        export_path: Path to save converted model


    """
    export_path.mkdir(parents=True, exist_ok=True)

    model_path = str(model_path)

    base_model = keras.models.load_model(model_path)
    base_model.summary()

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()

    save_path = export_path / "model.tflite"
    with save_path.open("wb") as f:
        f.write(tflite_model)

    print(f"Model saved to {save_path}")

    # Convert the model.
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    save_path = export_path / "model_optimized.tflite"
    with save_path.open("wb") as f:
        f.write(tflite_model)

    print(f"Optimized model saved to {save_path}")

    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    data = load_data_as_dataframe(data_path)

    X = data[["x", "y", "z"]]
    # add shift columns
    X["x-1"] = X["x"].shift(1).fillna(0).values
    X["y-1"] = X["y"].shift(1).fillna(0).values
    X["z-1"] = X["z"].shift(1).fillna(0).values

    X["x-2"] = X["x"].shift(2).fillna(0).values
    X["y-2"] = X["y"].shift(2).fillna(0).values
    X["z-2"] = X["z"].shift(2).fillna(0).values

    y = data["button_state"].values.reshape(-1, 1)

    dataset = tf.data.Dataset.from_tensor_slices((dict(X), y))
    dataset = dataset.shuffle(buffer_size=len(X))

    def representative_dataset():
        for data in dataset:
            yield data

    converter.representative_dataset = representative_dataset
    converter.optimizations = [tf.lite.Optimize.EXPERIMENTAL_SPARSITY]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.float32  # or tf.uint8
    converter.inference_output_type = tf.float32  # or tf.uint8
    tflite_quant_model = converter.convert()

    save_path = export_path / "model_quantized.tflite"
    with save_path.open("wb") as f:
        f.write(tflite_quant_model)

    print(f"Quantized model saved to {save_path}")


if __name__ == "__main__":
    main()
