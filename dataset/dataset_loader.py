import tensorflow as tf

data_dir = "C://Users//Amelia//PycharmProjects//flowers_model_CNN//dataset//flower_photos"
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

def prepare_datasets(data_dir, img_size=(150, 150), batch_size=32, train_split=0.8):

    dataset = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size
    )

    train_size = int(len(dataset) * train_split)
    train_ds = dataset.take(train_size)
    test_ds = dataset.skip(train_size)

    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    return train_ds, test_ds