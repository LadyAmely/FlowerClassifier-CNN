import os
import sys
from dataset.dataset_loader import prepare_datasets
models_dir = os.path.join(os.path.dirname(__file__), '../models')
sys.path.append(models_dir)
from models.cnn_model import CNN_Model


def train_model():
    data_dir = "C://Users//Amelia//PycharmProjects//flowers_model_CNN//dataset//flower_photos"
    img_size = (150, 150)
    batch_size = 32

    train_ds, test_ds = prepare_datasets(data_dir, img_size=img_size, batch_size=batch_size)

    model = CNN_Model(input_shape=(img_size[0], img_size[1], 3), num_classes=5)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=20,
        verbose=1
    )

    model.export('../saved_models/cnn_model')
    model.save('../saved_models/cnn_model.keras')
    model.save('../saved_models/cnn_model.h5')


if __name__ == "__main__":
    train_model()