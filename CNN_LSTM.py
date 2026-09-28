import os
import random
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pyeeg as pe
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, LSTM, Dropout, BatchNormalization, GaussianNoise
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
from keras import regularizers

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEAP_DATA_DIR = os.path.join(BASE_DIR, "data", "raw_deap")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_DIR, "best_model.keras")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def FFT_Processing(sub, channel, band, window_size, step_size, sample_rate):
    meta_features = []
    meta_labels = []

    file_path = os.path.join(DEAP_DATA_DIR, f"s{sub}.dat")
    with open(file_path, "rb") as file:
        subject = pickle.load(file, encoding="latin1")

        for i in range(0, 40):
            data = subject["data"][i]
            labels = subject["labels"][i]
            start = 0

            while start + window_size <= data.shape[1]:
                meta_data = []
                for j in channel:
                    X = data[j][start : start + window_size]
                    Y = pe.bin_power(X, band, sample_rate)
                    meta_data += list(Y[0])

                meta_features.append(np.array(meta_data))
                meta_labels.append(np.array(labels))
                start = start + step_size

    meta_features = np.array(meta_features).astype(np.float32)
    meta_labels = np.array(meta_labels).astype(np.float32)

    np.save(os.path.join(DATA_DIR, f"s{sub}_features.npy"), meta_features, allow_pickle=True)
    np.save(os.path.join(DATA_DIR, f"s{sub}_labels.npy"), meta_labels, allow_pickle=True)
    print(f"Processed subject {sub}")


def save_results(history, score, cmatrix, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    results = {
        'history': history.history,
        'score': score,
        'confusion_matrix': cmatrix
    }
    joblib.dump(results, os.path.join(results_dir, 'results.pkl'))
    print(f"Results saved at {os.path.join(results_dir, 'results.pkl')}")


def plot_results(results, save_dir=None):
    history = results['history']
    cmatrix = results['confusion_matrix']

    # Accuracy Plot
    plt.figure(figsize=(7, 5))
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'accuracy.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Loss Plot
    plt.figure(figsize=(7, 5))
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'])
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'loss.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(5, 4))
    sns.heatmap(cmatrix, annot=True, cmap=plt.cm.Blues, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    # Parameters
    subjectList = ["{:02d}".format(i) for i in range(1, 33)]
    channel = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]
    band = [4, 8, 12, 16, 25, 45]
    window_size = 512
    step_size = 16
    sample_rate = 128

    # Check cache / feature extraction
    all_files_exist = all(
        os.path.exists(os.path.join(DATA_DIR, f"s{sub}_features.npy"))
        and os.path.exists(os.path.join(DATA_DIR, f"s{sub}_labels.npy"))
        for sub in subjectList
    )

    if not all_files_exist:
        print('Processed files missing. Running feature extraction...')
        for subjects in subjectList:
            FFT_Processing(subjects, channel, band, window_size, step_size, sample_rate)

    # Reproducibility
    seed = 2024
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    
    label_idx = 0  # 0: Arousal, 1: Valence, 2: Dominance, 3: Liking
    all_features = []
    all_labels = []

    for subjects in subjectList:
        f_path = os.path.join(DATA_DIR, f's{subjects}_features.npy')
        l_path = os.path.join(DATA_DIR, f's{subjects}_labels.npy')
        features = np.load(f_path, allow_pickle=True).astype(np.float32)
        labels = np.load(l_path, allow_pickle=True).astype(np.float32)
        all_features.append(features)
        all_labels.append(labels)

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    print('All data shape:', all_features.shape, all_labels.shape)

    # Train/Test Split (80/20)
    label_bin = (all_labels[:, label_idx] > 5).astype(int)
    X_train, X_test, Y_train, Y_test = train_test_split(
        all_features, all_labels,
        test_size=0.20, random_state=seed, shuffle=True, stratify=label_bin
    )
    print('X_train:', X_train.shape, 'Y_train:', Y_train.shape)
    print('X_test:', X_test.shape, 'Y_test:', Y_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    Z_train = (Y_train[:, label_idx] > 5).astype(int)
    L_test = (Y_test[:, label_idx] > 5).astype(int)
    y_train = to_categorical(Z_train)
    y_test = to_categorical(L_test)

    # Reshape for CNN-LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    # Noise addition (explicit float32 to prevent memory bloat)
    noise = np.random.normal(0, 0.01, X_train.shape).astype(np.float32)
    X_train += noise

    # Model Hyperparameters
    BATCH_SIZE = 1024
    EPOCHS = 300
    LEARNING_RATE = 0.0004
    PATIENCE = 12

    model = Sequential([
        Conv1D(256, kernel_size=7, activation='relu', padding='same', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(128, kernel_size=5, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        GaussianNoise(0.02),

        LSTM(128, activation='tanh',
             kernel_regularizer=regularizers.l2(1e-4),
             recurrent_regularizer=regularizers.l2(1e-4),
             bias_regularizer=regularizers.l2(1e-5),
             return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),

        Dense(384, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        Dropout(0.15),
        Dense(y_train.shape[1], activation='softmax')
    ])
    model.summary()

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=2, min_delta=0.0001)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.1, min_lr=1e-5, verbose=2)
    chkpt = ModelCheckpoint(MODEL_PATH, verbose=1, save_best_only=True, monitor='val_loss', mode='min')
    callbacks = [early_stop, reduce_lr, chkpt]

    # Training
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_test, y_test),
        callbacks=callbacks,
        verbose=2
    )

    # Evaluation
    score = model.evaluate(X_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    y_pred = model.predict(X_test)
    y_test_classes = np.argmax(y_test, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    cmatrix = confusion_matrix(y_test_classes, y_pred_classes)

    # Save & Plot
    save_results(history, score, cmatrix, RESULTS_DIR)
    results_data = {
        'history': history.history,
        'score': score,
        'confusion_matrix': cmatrix
    }
    plot_results(results_data, save_dir=RESULTS_DIR)
    print("Execution completed successfully.")


if __name__ == "__main__":
    main()
