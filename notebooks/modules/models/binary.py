import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def fully_connected(n_features, n_steps, params):
    '''Create a binary classifier NN with a single fully connected hidden layer'''
    # Input layer
    inputs = tf.keras.Input(shape=(None, n_features), name='inputs')
    # Masking layer
    x = tf.keras.layers.Masking(mask_value=params['mask_value'])(inputs)
    # Reshape layer
    x = tf.keras.layers.Reshape((n_features*n_steps,))(x)
    # Fully connected layer
    x = tf.keras.layers.Dense(n_features*n_steps, input_shape=(n_steps, n_features), activation='relu')(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name='Baseline')

    # Compile model
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=params['optimizer'],
                metrics=[tf.keras.metrics.AUC(), 'binary_accuracy'])

    return model

def create_model(n_features, params, recurrent_layer='LSTM'):
    '''Create a simple recurrent neural network binary classifier'''
    # Input layer
    inputs = tf.keras.Input(shape=(None, n_features), name='inputs')

    # Mask layer to ignore masked padded values
    x = tf.keras.layers.Masking(mask_value=params['mask_value'])(inputs)

    # Simple Recurrent layer
    if recurrent_layer == 'RNN':
        x = tf.keras.layers.SimpleRNN(params['n_units'], return_sequences=False, 
                                      dropout=params['dropout'])(x)

    # Many-to-one LSTM layer
    if recurrent_layer == 'LSTM':
        x = tf.keras.layers.LSTM(params['n_units'], return_sequences=False, 
                                 dropout=params['dropout'])(x)

    if recurrent_layer == 'GRU':
        x = tf.keras.layers.GRU(params['n_units'], return_sequences=False,
                                dropout=params['dropout'])(x)

    # Output prediction layer
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    # Completed model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs, name=f'Binary{recurrent_layer}')
    # Compile model
    model.compile(loss=tf.keras.losses.binary_crossentropy, optimizer=params['optimizer'],
                metrics=[tf.keras.metrics.AUC(), 'binary_accuracy'])
    return model

def fit_model(model, params, name, X_train, y_train, X_val, y_val):
    '''Fit model on train and validation data'''
    # Save best model during training process
    save_best = tf.keras.callbacks.ModelCheckpoint(f'{params["weights_dir"]}/{name}-binary', 
                                              save_best_only=True, save_weights_only=True)

    # Early stop: halt training when a validation loss has stopped improving
    early_stop = tf.keras.callbacks.EarlyStopping(patience=params['patience'])

    # Fit model on the data
    history = model.fit(X_train, y_train, epochs=params['epochs'], validation_data=(X_val, y_val),
                   callbacks=[save_best, early_stop], verbose=params['verbose'])

def evaluate_model(model, params, modelname, X_test, y_test):
    '''Evaluate model on test set (accuracy and AUC)'''
    model.load_weights(f'{params["weights_dir"]}/{modelname}-binary')
    predictions = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, predictions>0.5)
    auc = roc_auc_score(y_test, predictions)
    print(f'Model: {modelname}')
    print(f'Balanced accuracy: {acc:0.4f}')
    print(f'AUC: {auc:0.4f}')
    return acc, auc
