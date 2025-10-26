#!/usr/bin/env python3

import argparse
import tensorflow as tf
from tensorflow import keras


def convert_h5_to_tflite(h5_path, tflite_path):

    print(f"Loading model from {h5_path}...")
    model = keras.models.load_model(h5_path)
    print("Model loaded successfully")


    print("\nConverting to TensorFlow Lite format...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    tflite_model = converter.convert()

    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {tflite_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")

def main():
    parser = argparse.ArgumentParser(
        description="Convert Keras H5 model to TensorFlow Lite"
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to input .h5 model'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Path to save .tflite model'
    )
    parser.add_argument(
        '--quantize',
        action='store_true',
        help='Apply dynamic range quantization (reduces model size)'
    )

    args = parser.parse_args()

    convert_h5_to_tflite(args.model, args.output)


if __name__ == '__main__':
    main()