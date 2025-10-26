# !/usr/bin/env python3

import os
import json
import cv2
import numpy as np
import argparse
from tensorflow import keras

def decode_predictions(preds, symbols):
    results = []
    for pred in zip(*preds):
        text = ''.join(symbols[np.argmax(p)] for p in pred)
        text = text.replace(' ', '')
        results.append(text)
    return results


def preprocess_image(image_path, img_width, img_height):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (img_width, img_height))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Classify CAPTCHAs using trained Keras model")
    parser.add_argument('--model', required=True, help='Path to trained model (.h5)')
    parser.add_argument('--config', required=True, help='Path to model_config.json')
    parser.add_argument('--captcha-dir', required=True, help='Directory with CAPTCHA images')
    parser.add_argument('--output-file', required=True, help='Path to save results CSV')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    symbols = config['symbols'] + ' '
    max_length = config['max_length']
    img_width = config['img_width']
    img_height = config['img_height']

    print(f"Loaded config: {len(symbols) - 1} symbols, max length = {max_length}")
    print(f"Image size: {img_width}x{img_height}")

    print(f"Loading model from {args.model} ...")
    model = keras.models.load_model(args.model)
    print("Model loaded successfully")
    image_files = [f for f in os.listdir(args.captcha_dir) if f.endswith('.png')]
    image_files.sort()

    if not image_files:
        print(" No .png images found in directory!")
        return

    print(f"Found {len(image_files)} images")

    results = []
    for fname in image_files:
        img_path = os.path.join(args.captcha_dir, fname)
        img = preprocess_image(img_path, img_width, img_height)
        img_batch = np.expand_dims(img, axis=0)
        preds = model.predict(img_batch, verbose=0)
        decoded = decode_predictions(preds, symbols)[0]
        results.append((fname, decoded))
        print(f"{fname}: {decoded}")

    with open(args.output_file, 'w') as f:
        f.write("ralakshm\n")
        for fname, pred in results:
            f.write(f"{fname},{pred}\n")

    print(f"\nResults saved to {args.output_file}")
    print("Classification complete!")


if __name__ == '__main__':
    main()