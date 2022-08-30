# Any code helping for session happiness predictions
import numpy as np
import pandas as pd

import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression


def simple_NN(n_features, out_act=tf.keras.layers.ReLU(max_value=1.0), loss=tf.keras.losses.MeanSquaredError(),
              metrics=[tf.keras.losses.MeanAbsoluteError()]):
    '''Simple fully connected NN architecture'''
    # Input layer
    inputs = tf.keras.Input(shape=(n_features))
    # Two hidden layers
    x = tf.keras.layers.Dense(n_features, input_shape=(None, n_features), activation='relu')(inputs)
    x = tf.keras.layers.Dense(n_features, input_shape=(None, n_features), activation='relu')(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation=out_act)(x)

    # Model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(loss=loss, optimizer='adam', metrics=metrics)

    # Return model
    return model

def deep_NN(n_features):
    '''Simple Deep NN architecture'''
    # Input layer
    inputs = tf.keras.Input(shape=(n_features))
    # Eight hidden layers
    x = tf.keras.layers.Dense(n_features, input_shape=(None, n_features), activation='relu')(inputs)
    for i in range(7):
        x = tf.keras.layers.Dense(n_features, input_shape=(None, n_features), activation='relu')(x)
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation=tf.keras.layers.ReLU(max_value=1.0))(x)

    # Model
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    # Compile model
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer='adam', 
                  metrics=[tf.keras.losses.MeanAbsoluteError()])

    # Return model
    return model

def fit_model(model, X_train, y_train, X_val, y_val, verbose=1):
    '''Fit NN model on training and validation data'''
    history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), verbose=verbose)
    
def eval_nn_model(model, X_test, y_test, verbose=True):
    '''Evaluate NN model on testing data'''
    pred = model.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    if verbose:
        print('== Model performance on test set ==')
        print(f'\tMSE: {mse:.4f}')
        print(f'\tMAE: {mae:.4f}')
    return mse, mae

def eval_reg_mod(model, X_train, y_train, X_test, y_test, verbose=True):
    '''Evaluate regression model'''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse, mae = mean_squared_error(y_test, y_pred), mean_absolute_error(y_test, y_pred)
    if verbose:
        print(f'Model: {model}')
        print(f'\tMSE: {mse:.4f}')
        print(f'\tMAE: {mae:.4f}')
    return mse, mae

def low_participants_removed(data):
    '''Visualize regression performance when low number of participant sessions removed'''
    mae_results = []
    mse_results = []
    mins = np.arange(1, 21)
    for m in mins:
        # Consider data subset
        X_sub = data[data.n_participants >= m]
        y_sub = X_sub.pop('response')

        # Apply train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.1, random_state=42)

        # Instantiate model and fit to training data
        gbr = GradientBoostingRegressor()
        mse, mae = eval_reg_mod(gbr, X_train, y_train, X_test, y_test, False)
        mae_results.append(mae)
        mse_results.append(mse)

    plt.plot(mins, mae_results, label='MAE')
    plt.plot(mins, mse_results, label='MSE')
    plt.title('Regression performance v.s. minimum required answering participants')
    plt.xlabel('Minimum number of required answering participants')
    plt.ylabel('Metric score')
    plt.legend()
    plt.xticks(ticks=[1, 5, 10, 15, 20], labels=[1, 5, 10, 15, 20])
    plt.show()
    
    
def compare_reg_deep(data, include_nn=True, include_baseline=False):
    '''Visualize performance between regression and NN models'''
    gbr_results = []
    ridge_results = []
    nn_results = []
    deep_results = []
    baseline_results = []
    mins = np.arange(1, 21)
    for m in mins:
        # Consider data subset
        X_sub = data[data.n_participants >= m]
        y_sub = X_sub.pop('response')

        # Apply train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.1, random_state=123)

        if include_baseline:
            y_pred = [np.mean(y_train)] * len(y_test)
            baseline_results.append(mean_absolute_error(y_test, y_pred))

        # Gradient Boosting Regression model
        gbr = GradientBoostingRegressor()
        _, mae = eval_reg_mod(gbr, X_train, y_train, X_test, y_test, verbose=False)
        gbr_results.append(mae)

        # Ridge Reg (CV) model
        alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation
        ridge = RidgeCV(alphas=alphas)
        _, mae = eval_reg_mod(ridge, X_train, y_train, X_test, y_test, verbose=False)
        ridge_results.append(mae)


        if include_nn:
            # Train-val split 80-20 for NN models
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123)

            # Simple NN model
            nn = simple_NN(X_train.shape[1])
            # Train with training and validation data
            fit_model(nn, X_train, y_train, X_val, y_val, verbose=0)
            # Evaluate on test data
            _, mae = eval_nn_model(nn, X_test, y_test, verbose=False)
            nn_results.append(mae)

            # Deep NN model
            deep = deep_NN(X_train.shape[1])
            # Train with training and validation data
            fit_model(deep, X_train, y_train, X_val, y_val, verbose=0)
            # Evaluate on test data
            _, mae = eval_nn_model(deep, X_test, y_test, verbose=False)
            deep_results.append(mae)

    plt.title('Regression and NN models compared')
    if include_baseline:
        plt.plot(mins, baseline_results, label='Baseline')
    plt.plot(mins, gbr_results, label='Gradient Boosting Reg')
    plt.plot(mins, ridge_results, label='RidgeCV Reg')
    if include_nn:
        plt.plot(mins, nn_results, label='Simple NN')
        plt.plot(mins, deep_results, label='Deep NN')
        plt.title('Regression models compared w/ min # of required participants')
    plt.xlabel('Minimum number of required answering participants')
    plt.ylabel('Mean absolute error')
    plt.legend()
    plt.xticks(ticks=[1, 5, 10, 15, 20], labels=[1, 5, 10, 15, 20])
    plt.ylim(bottom=0)
    plt.show()
    
def rf_reg_importance(data, min_):
    '''Inspect random forest regressor feature importances'''
    # Keep relevant data
    X_sub = data[data.n_participants >= min_]
    y_sub = X_sub.pop('response')

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.1, random_state=123)

    forest = RandomForestRegressor()
    # Specif feature names
    feature_names = X_sub.columns.values

    # Fit model on training data
    forest.fit(X_train, y_train)

    # The impurity-based feature importances
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)

    # Attach feature names to importances
    forest_importances = pd.Series(importances, index=feature_names).sort_values()

    # Plot
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title(f"Feature importances using MDI (min participants = {min_})")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    
def ridge_reg_importance(data, min_):
    '''Show feature importance on ridge regression'''
    X_sub = data[data.n_participants >= min_]
    y_sub = X_sub.pop('response')

    X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.1, random_state=123)

    alphas = np.logspace(-10, 10, 21)  # alpha values to be chosen from by cross-validation
    model = make_pipeline(
      StandardScaler(),
      RidgeCV(alphas=alphas)
    )

    mse, mae = eval_reg_mod(model, X_train, y_train, X_test, y_test, False)
    print(f'MAE: {mae}')

    feature_names = X_sub.columns.values

    coefs = pd.DataFrame(
      model[-1].coef_,
      columns=["Coefficients importance"],
      index=feature_names,
    )

    coefs.plot.barh(figsize=(9, 7))
    plt.title("Ridge model, with regularization, normalized variables")
    plt.xlabel("Raw coefficient values")
    plt.axvline(x=0, color=".5")
    plt.subplots_adjust(left=0.3)
    plt.show()
    
def binary_lower_participants(data):
    '''Visualize binary prediction performance when low number of participant sessions removed'''
    nn_results = []
    gb_results = []
    forest_results = []
    mins = np.arange(1, 21)
    for m in mins:
        # Consider data subset
        X_sub = data[data.n_participants >= m]
        y_sub = X_sub.pop('response')

        # Apply train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_sub, y_sub, test_size=0.1, random_state=123)

        # Gradient Boosting Classifier
        gbc = GradientBoostingClassifier().fit(X_train, y_train)
        y_pred = gbc.predict(X_test)
        gb_results.append(balanced_accuracy_score(y_test, y_pred))

        # Random Forest Classifier
        forest = RandomForestClassifier().fit(X_train, y_train)
        y_pred = forest.predict(X_test)
        forest_results.append(balanced_accuracy_score(y_test, y_pred))

        # Simple Neural Network
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=123)
        simple = simple_NN(X_train.shape[1], out_act='sigmoid', loss=tf.keras.losses.BinaryCrossentropy(),
                       metrics=[tf.keras.metrics.BinaryAccuracy()])
        fit_model(simple, X_train, y_train, X_val, y_val, verbose=0)
        y_pred = simple.predict(X_test) >= 0.5
        nn_results.append(balanced_accuracy_score(y_test, y_pred))   


    plt.plot(mins, nn_results, label='Simple NN')
    plt.plot(mins, gb_results, label='GradientBoostingClassifier')
    plt.plot(mins, forest_results, label='RandomForestClassifier')
    plt.title('Binary prediction performance v.s. minimum required answering participants')
    plt.xlabel('Minimum number of required answering participants')
    plt.ylabel('Balanced accuracy')
    plt.legend()
    plt.xticks(ticks=[1, 5, 10, 15, 20], labels=[1, 5, 10, 15, 20])
    plt.show()