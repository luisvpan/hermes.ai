import os

import keras
import pandas

class ModelBuilder():
    def __init__(self) -> None:
        self.model: keras.models.Sequential | None = None

    def build_model(self) -> keras.models.Sequential:
        self.model = keras.models.Sequential()
        self._setup_model_shape()
        self._train_model()
        self._evaluate_model()

        model = self.model
        self.model = None
        return model
    
    def _setup_model_shape(self) -> None:
        self.model.add(keras.Input(shape=(28, 28, 1)))
        
        self.model.add(keras.layers.Conv2D(32, 3, activation='relu'))
        self.model.add(keras.layers.MaxPooling2D(2, 2))
        self.model.add(keras.layers.Conv2D(64, 3, activation='relu'))
        
        self.model.add(keras.layers.Flatten())
        
        self.model.add(keras.layers.Dense(128, activation='relu'))
        self.model.add(keras.layers.Dropout(0.20))
        self.model.add(keras.layers.Dense(26, activation='softmax'))

    def _train_model(self) -> None:
        training_data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/emnist-letters-train.csv'))
        normalized_training_images = training_data.iloc[:, 1:]
        normalized_training_images = normalized_training_images.values.reshape(-1, 28, 28)
        normalized_training_images = normalized_training_images.astype('float32') / 255
        normalized_training_labels = keras.utils.to_categorical(
            training_data.iloc[:, 0] - 1,
            num_classes=26,
        )

        self.model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'],
        )

        self.model.fit(
            x=normalized_training_images,
            y=normalized_training_labels,
            epochs=5,
            batch_size=20,
            validation_split=0.2,
        )

    def _evaluate_model(self) -> None:
        testing_data = pandas.read_csv(os.path.join(os.path.dirname(__file__), '../datasets/emnist-letters-test.csv'))
        normalized_testing_images = testing_data.iloc[:, 1:]
        normalized_testing_images = normalized_testing_images.values.reshape(-1, 28, 28)
        normalized_testing_images = normalized_testing_images.astype('float32') / 255
        normalized_testing_labels = keras.utils.to_categorical(
            testing_data.iloc[:, 0] - 1,
            num_classes=26,
        )
        
        self.model.evaluate(
            x=normalized_testing_images,
            y=normalized_testing_labels,
            batch_size=20,
        )