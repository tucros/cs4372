import os
import warnings
from itertools import chain, combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

columns = [
    "mpg",
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
    "origin",
    "car name",
]

numerical_columns = [
    "mpg",
    "horsepower",
    "weight",
    "acceleration",
    "displacement",
]


def load_data():
    url = "https://github.com/williamphyoe/datasets/blob/main/auto-mpg.csv?raw=true"
    df = pd.read_csv(url, header=None, names=columns, sep="\s+")
    return df


def generate_exploration_plots(df):
    if not os.path.exists("graphs"):
        os.mkdir("graphs")

    # Pairplot numerical columns
    plt.figure(figsize=(12, 12))
    sns.pairplot(df[numerical_columns], diag_kind="kde")
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    # save to file in graphs folder
    plt.savefig("graphs/pairplot.png")

    plt.figure(figsize=(12, 8))
    sns.stripplot(x="cylinders", y="mpg", data=df, jitter=0.3)
    plt.title("MPG vs Cylinders")
    plt.savefig("graphs/mpg_vs_cylinders.png")

    # barplot of origin
    plt.figure(figsize=(12, 8))
    sns.stripplot(x="origin", y="mpg", data=df, jitter=0.3)
    plt.title("MPG vs Origin")
    plt.savefig("graphs/mpg_vs_origin.png")

    return df


def clean_data(df):
    print("Nulls:")
    print(df.isnull().sum())
    print("\nNaN values:")
    print(df.isna().sum())
    print("\n?s:")
    print(df.isin(["?"]).sum())

    df = df.mask(df == "?").dropna()

    df["horsepower"] = df["horsepower"].astype(float)
    df["origin"] = pd.Categorical(df["origin"])
    df["cylinders"] = pd.Categorical(df["cylinders"])

    # Drop unused columns and add bias
    df = df.drop(["car name", "model year"], axis=1)
    df["bias"] = 1

    return df


def vifs(X):
    print("\nVIFs:")
    vif = [variance_inflation_factor(X, i) for i in range(len(X.columns))]
    for i, col in enumerate(X.columns):
        print(f"{col}: {vif[i]:.2f}")


def ols_run_all(X, y):
    columns = list(X.columns)
    columns.remove("bias")
    combination_iter = chain.from_iterable(
        combinations(columns, i) for i in range(len(columns) + 1)
    )

    all_runs = []
    for combination in combination_iter:
        features = list(combination)
        features.append("bias")
        if len(combination) > 0:
            model = sm.OLS(y, X[features]).fit()
            all_runs.append((features, model.rsquared_adj))

    all_runs.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 3 OLS Results:")
    for i in range(3):
        print(f"  Features: {all_runs[i][0]}")
        print(f"  Adj R2: {all_runs[i][1]:.4f}\n")


def feature_eng(df, use_ols=True):
    print("\nCorrelation Matrix: ")
    print(df.corr())

    X = df.drop("mpg", axis=1)
    y = df["mpg"]

    if use_ols:
        ols_run_all(X, y)
    vifs(X)

    df = df.drop(["displacement"], axis=1)

    X = df.drop("mpg", axis=1)
    vifs(X)
    return df


def sgd_model(X_train, y_train, X_test, y_test):
    X_train_scaled = X_train.copy()
    scaler = StandardScaler()
    numeric_columns = ["horsepower", "weight", "acceleration"]
    X_train_scaled[numeric_columns] = scaler.fit_transform(
        X_train_scaled[numeric_columns]
    )

    X_test_scaled = X_test.copy()
    X_test_scaled[numeric_columns] = scaler.transform(X_test_scaled[numeric_columns])

    # One-hot encode 'origin' and 'cylinders'
    X_train_scaled = pd.get_dummies(
        X_train_scaled, columns=["origin", "cylinders"], drop_first=True
    )
    X_test_scaled = pd.get_dummies(
        X_test_scaled, columns=["origin", "cylinders"], drop_first=True
    )

    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "eta0": [0.001, 0.005, 0.01, 0.05, 0.1],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [1000, 2000, 5000],
        "tol": [1e-3, 1e-4, 1e-5],
    }
    print(f"\nNumber of combinations: {np.prod([len(v) for v in param_grid.values()])}")
    print("Running GridSearchCV...")
    sgd = SGDRegressor()
    grid_search = GridSearchCV(
        sgd,
        param_grid,
        return_train_score=True,
        scoring=["neg_mean_squared_error", "r2"],
        refit="r2",
        n_jobs=-1,
    )  # , verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    results = pd.DataFrame(grid_search.cv_results_)
    results.sort_values("mean_test_r2", ascending=False, inplace=True)

    print("SGDRegressor Results:")
    print(f"Best parameters: {grid_search.best_params_}")

    best_sgd = grid_search.best_estimator_
    y_train_pred = best_sgd.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    print(f"Training RMSE: {train_rmse:.4f}")
    train_r2 = r2_score(y_train, y_train_pred)
    print(f"Training R2: {train_r2:.4f}")

    y_test_pred = grid_search.predict(X_test_scaled)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    print(f"Test RMSE: {test_rmse:.4f}")
    test_r2 = r2_score(y_test, y_test_pred)
    print(f"Test R2: {test_r2:.4f}")

    # print coefficient values and names of features
    print("\nCoefficients:")
    for i, coef in enumerate(best_sgd.coef_):
        print(f"{X_train_scaled.columns[i]}: {coef:.4f}")

    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_ev = explained_variance_score(y_train, y_train_pred)
    print(f"Train MAE: {train_mae:.4f}")
    print(f"Train EV: {train_ev:.4f}")

    sgd = SGDRegressor()
    sgd.fit(X_train_scaled, y_train)
    y_train_pred = sgd.predict(X_train_scaled)
    y_test_pred = sgd.predict(X_test_scaled)
    # pint r2 and rmse
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    print("\nSGD Results:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    results.to_excel("sgd_results.xlsx")

    # scatter plot of residuals
    plt.figure(figsize=(12, 8))
    plt.scatter(y_test_pred, y_test - y_test_pred)
    plt.xlabel("Predicted")
    plt.ylabel("Residual")
    plt.title("Residuals")
    plt.savefig("graphs/residuals.png")


def ols_model(X, y):
    model = sm.OLS(y, X).fit()
    print("\nOLS Summary:")
    print(model.summary())

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nOLS Results:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")


def split_data(df):
    # Split data into training and test sets
    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    return train_test_split(X, y, test_size=0.2)


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = (
        load_data()
        .pipe(clean_data)
        .pipe(generate_exploration_plots)
        .pipe(feature_eng)
        .pipe(split_data)
    )

    sgd_model(X_train, y_train, X_test, y_test)
    ols_model(X_train, y_train)
