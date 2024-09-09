import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def train_model(model, X_train, y_train, X_test, y_test):
    # Data Augmentation
    data_aug = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=20, zoom_range=0.2,
                                  width_shift_range=0.2, height_shift_range=0.2, shear_range=0.1, fill_mode="nearest")

    # Callbacks
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model.weights.h5', monitor='val_loss', save_best_only=True, save_weights_only=True)

    # Train the model
    history = model.fit(data_aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=50,
                        callbacks=[early_stopping, reduce_lr, model_checkpoint])

    # Fine-tuning
    model.trainable = True
    model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), metrics=['accuracy'])

    history_fine = model.fit(data_aug.flow(X_train, y_train, batch_size=32), validation_data=(X_test, y_test), epochs=20,
                             callbacks=[early_stopping, reduce_lr, model_checkpoint])

    return history, history_fine

def evaluate_model(model, X_test, y_test):
    # Prediction
    y_pred = model.predict(X_test)
    pred = np.argmax(y_pred, axis=1)
    ground = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(ground, pred))

def plot_results(history):
    # Plot accuracy
    epochs = range(len(history.history['accuracy']))
    plt.plot(epochs, history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(epochs, history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend(loc='lower right')
    plt.show()

    # Plot loss
    plt.plot(epochs, history.history['loss'], 'r', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend(loc='upper right')
    plt.show()
