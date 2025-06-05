import tensorflow as tf
def build_model(num_classes=5):
    base_model = tf.keras.applications.InceptionV3(
        input_shape=(224, 224, 3), include_top=False, weights='weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.Rescaling(1./255)(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base_model

def compile_model(model, learning_rate=0.0001):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(),
                 tf.keras.metrics.AUC(multi_label=True),
                 tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(),
                 tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives()]
    )
    return model


def fine_tune_model(base_model, model, fine_tune_at=270):
    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    return model