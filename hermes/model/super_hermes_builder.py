import os
from typing import Any, Generator

import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
import numpy as np

DATASET_PATH = os.path.join("src", "dataset")


class ModelBuilder:
    def __init__(self) -> None:
        self.model: Sequential | None = None

    def build_model(self) -> Sequential:
        self._setup_model_shape()
        self._train_model()

        model = self.model
        self.model = None
        return model

    def _setup_model_shape(self) -> None:
        self.model = Sequential()
        self.model.add(
            Conv2D(
                32, (3, 3), input_shape=(28, 28, 1), activation="relu", padding="same"
            )
        )
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(50, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )

    def _calculate_number_of_categories(self) -> int:
        return len(
            list(
                filter(
                    lambda filename: filename.endswith(".npy"), os.listdir(DATASET_PATH)
                )
            )
        )

    def _find_categories(self) -> Generator[Any, Any, None]:
        categories_filenames = filter(
            lambda filename: filename.endswith(".npy"), sorted(os.listdir(DATASET_PATH))
        )
        for category_filename in categories_filenames:
            category_filepath = os.path.join(DATASET_PATH, category_filename)
            yield category_filename, np.load(category_filepath)

    def _generate_training_sample(self):
        num_categories = self._calculate_number_of_categories()
        num_examples_per_category = 6000

        labels = np.zeros(
            shape=(
                num_categories * num_examples_per_category,
                num_categories,
            )
        )
        examples = np.zeros(
            shape=(num_categories * num_examples_per_category, 28, 28, 1)
        )

        for i, (current_category_filename, current_category) in enumerate(
            self._find_categories()
        ):
            print(
                f"Generating training samples for the category in the file {current_category_filename}"
            )

            current_category = current_category.astype("float32") / 255

            current_examples = current_category.reshape(
                current_category.shape[0], 28, 28, 1
            )
            examples[
                i * num_examples_per_category : (i + 1) * num_examples_per_category
            ] = current_examples[
                np.random.choice(
                    current_examples.shape[0],
                    size=num_examples_per_category,
                    replace=False,
                )
            ]

            current_labels = tf.keras.utils.to_categorical(
                i * np.ones(num_examples_per_category), num_classes=num_categories
            )
            labels[
                i * num_examples_per_category : (i + 1) * num_examples_per_category
            ] = current_labels

        return examples, labels

    def _train_model(self) -> None:
        for i in range(0, 20):
            examples, labels = self._generate_training_sample()
            self.model.fit(x=examples, y=labels, epochs=5, batch_size=64)
