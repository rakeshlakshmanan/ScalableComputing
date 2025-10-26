# !/usr/bin/env python3

import os
import numpy as np
from tensorflow import keras
import cv2
import argparse
import json
from sklearn.model_selection import train_test_split


class CaptchaDataGenerator(keras.utils.Sequence):

    def __init__(self, image_paths, labels, symbols, img_width, img_height,
                 max_length, batch_size=32, shuffle=True):
        self.image_paths = image_paths
        self.labels = labels
        self.symbols = symbols
        self.img_width = img_width
        self.img_height = img_height
        self.max_length = max_length
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.char_to_num = {char: idx for idx, char in enumerate(symbols)}
        self.indexes = np.arange(len(self.image_paths))
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[k] for k in indexes]
        batch_labels = [self.labels[k] for k in indexes]

        X, y = self.__data_generation(batch_paths, batch_labels)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_paths, batch_labels):
        X = np.zeros((self.batch_size, self.img_height, self.img_width, 1), dtype=np.float32)
        y = {f'char_{i}': np.zeros((self.batch_size,), dtype=np.int32) for i in range(self.max_length)}

        for i, (img_path, label) in enumerate(zip(batch_paths, batch_labels)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (self.img_width, self.img_height))
            img = img.astype(np.float32) / 255.0
            X[i] = np.expand_dims(img, axis=-1)
            for j, char in enumerate(label):
                if j < self.max_length:
                    y[f'char_{j}'][i] = self.char_to_num[char]

        return X, y


def load_data(data_dir, symbols):
    image_paths = []
    labels = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.png'):
                label = file.split('_')[0]
                if all(char in symbols for char in label):
                    image_paths.append(os.path.join(root, file))
                    labels.append(label)

    print(f"Found {len(image_paths)} valid images")
    return image_paths, labels


def build_model(img_width, img_height, max_length, num_classes):

    inputs = keras.layers.Input(shape=(img_height, img_width, 1))
    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = []
    for i in range(max_length):
        output = keras.layers.Dense(num_classes, activation='softmax', name=f'char_{i}')(x)
        outputs.append(output)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def train_model(args):
    with open(args.symbols, 'r') as f:
        symbols = f.readline().strip()

    print(f"Using symbols: {symbols}")
    print(f"Number of classes: {len(symbols)}")

    print("\nLoading data...")
    image_paths, labels = load_data(args.data_dir, symbols)

    max_length = max(len(label) for label in labels)
    print(f"Maximum CAPTCHA length: {max_length}")

    padded_labels = [label + ' ' * (max_length - len(label)) for label in labels]

    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, padded_labels, test_size=args.validation_split, random_state=42
    )

    print(f"Training samples: {len(train_paths)}")
    print(f"Validation samples: {len(val_paths)}")

    train_gen = CaptchaDataGenerator(
        train_paths, train_labels, symbols + ' ',
        args.width, args.height, max_length,
        batch_size=args.batch_size, shuffle=True
    )

    val_gen = CaptchaDataGenerator(
        val_paths, val_labels, symbols + ' ',
        args.width, args.height, max_length,
        batch_size=args.batch_size, shuffle=False
    )

    print("\nBuilding model...")
    model = build_model(args.width, args.height, max_length, len(symbols) + 1)

    metrics_dict = {f'char_{i}': 'accuracy' for i in range(max_length)}

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=metrics_dict
    )

    model.summary()

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            os.path.join(args.output_dir, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )
    ]

    print("\nStarting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=callbacks
    )

    model.save(os.path.join(args.output_dir, 'model_captcha.h5'))

    config = {
        'symbols': symbols,
        'max_length': max_length,
        'img_width': args.width,
        'img_height': args.height
    }

    with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nTraining complete!")
    print(f"Models saved to: {args.output_dir}")
    print(f"Best validation loss: {min(history.history['val_loss']):.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CNN model for CAPTCHA recognition")
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing CAPTCHA images')
    parser.add_argument('--symbols', type=str, required=True,
                        help='File containing symbols used in CAPTCHAs')
    parser.add_argument('--output_dir', type=str, default='./models',
                        help='Directory to save trained models')
    parser.add_argument('--width', type=int, default=192,
                        help='Width of CAPTCHA images')
    parser.add_argument('--height', type=int, default=96,
                        help='Height of CAPTCHA images')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--validation_split', type=float, default=0.2,
                        help='Fraction of data for validation')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_model(args)


if __name__ == '__main__':
    main()