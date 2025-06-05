from src.data_loader import load_images
from src.feature_engineering import preprocess_images, one_hot_encode
from src.model_building import build_model, compile_model, fine_tune_model
from src.utils import split_data, get_labels_dict
import tensorflow as tf

# Load and preprocess data
data_dir_name = "Penalty_new_modified"
labels_dict = get_labels_dict()
X, y = load_images(data_dir_name, labels_dict)
X = preprocess_images(X)
y = one_hot_encode(y, num_classes=len(labels_dict))

# Split dataset
X_train, X_test, y_train, y_test = split_data(X, y)

# Build and compile model
model, base_model = build_model(num_classes=len(labels_dict))
model = compile_model(model)

# Train initial model
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=20, mode="max", restore_best_weights=True
)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=200, batch_size=32,
    callbacks=[early_stop]
)

# Fine-tuning
model = fine_tune_model(base_model, model, fine_tune_at=270)
model = compile_model(model)
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    initial_epoch=200, epochs=400,
    batch_size=32, callbacks=[early_stop]
)

# Save the trained model
import os
#os.makedirs("models", exist_ok=True)
model.save("penalty_classifier.keras")


#model.save("models/penalty_classifier.h5")

