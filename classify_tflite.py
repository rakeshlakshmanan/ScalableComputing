#!/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import argparse

try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow.lite as tflite


def decode_predictions(preds, symbols):
    results = []
    for pred in zip(*preds):
        text = ''.join(symbols[np.argmax(p)] for p in pred)
        text = text.replace(' ', '')
        results.append(text)
    return results


def preprocess_image(image_path, img_width, img_height):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Failed to load image: %s" % image_path)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # add channel dim
    return img


def main():
    parser = argparse.ArgumentParser(
        description="Classify CAPTCHAs using TFLite model (32-bit compatible)"
    )
    parser.add_argument(
        '--model',
        required=True,
        help='Path to TFLite model (.tflite)'
    )
    parser.add_argument(
        '--config',
        required=True,
        help='Path to model_config.json'
    )
    parser.add_argument(
        '--captcha-dir',
        required=True,
        help='Directory with CAPTCHA images'
    )
    parser.add_argument(
        '--output-file',
        required=True,
        help='Path to save results CSV'
    )
    parser.add_argument(
        '--name-in-output',
        required=True,
        help='Name to be added in output file'
    )

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    symbols = config['symbols'] + ' '
    max_length = config['max_length']
    img_width = config['img_width']
    img_height = config['img_height']

    print("Loaded config: {} symbols, max length = {}".format(len(symbols) - 1, max_length))
    print("Image size: {}x{}".format(img_width, img_height))
    print("Loading TFLite model from {}...".format(args.model))
    interpreter = tflite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("Model loaded successfully")
    print("Input shape: {}".format(input_details[0]['shape']))
    print("Number of outputs: {}".format(len(output_details)))

    image_files = [f for f in os.listdir(args.captcha_dir) if f.endswith('.png')]
    image_files.sort()

    if not image_files:
        print("No .png images found in directory!")
        return

    print("Found {} images\n".format(len(image_files)))

    results = []
    for idx, fname in enumerate(image_files):
        img_path = os.path.join(args.captcha_dir, fname)

        try:
            img = preprocess_image(img_path, img_width, img_height)
            img_batch = np.expand_dims(img, axis=0)
            interpreter.set_tensor(input_details[0]['index'], img_batch)
            interpreter.invoke()
            sorted_outputs = sorted(output_details, key=lambda x: x['name'])
            preds = []
            for output_detail in sorted_outputs:
                output_data = interpreter.get_tensor(output_detail['index'])
                preds.append(output_data)
            decoded = decode_predictions(preds, symbols)[0]
            results.append((fname, decoded))
            print("[{}/{}] {}: {}".format(idx + 1, len(image_files), fname, decoded))

        except Exception as e:
            print("Error processing {}: {}".format(fname, e))
            results.append((fname, "ERROR"))

    with open(args.output_file, 'w') as f:
        f.write(args.name_in_output+"\n")
        for fname, pred in results:
            f.write("{},{}\n".format(fname, pred))

    print("\nResults saved to {}".format(args.output_file))
    print("Classification complete!")

if __name__ == '__main__':
    main()