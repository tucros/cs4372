import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

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
    plt.figure(figsize=(20, 20))

    # Pairplot numerical columns
    sns.pairplot(df[numerical_columns], diag_kind="kde")
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.show()

    # Boxplots for each numerical variable
    plt.figure(figsize=(15, 10))
    for idx, column in enumerate(numerical_columns, 1):
        print(f"Boxplot of {column}")
        plt.subplot(3, 3, idx)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.show()

    # Heatmap of correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # barplot of cyclinders
    plt.figure(figsize=(12, 8))
    sns.countplot(x="cylinders", data=df)
    plt.title("Count of Cylinders")
    plt.show()

    # barplot of origin
    plt.figure(figsize=(12, 8))
    sns.countplot(x="origin", data=df)
    plt.title("Count of Origin")
    plt.show()

    # Scatterplot of MPG vs Horsepower, with color based on cylinders
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x="weight", y="mpg", hue="origin", data=df)
    plt.title("MPG vs Weight, colored by Origin")
    plt.show()

    # Distribution of MPG
    plt.figure(figsize=(12, 6))
    sns.histplot(df["mpg"], kde=True)
    plt.title("Distribution of MPG")
    plt.show()


def clean_data(df):
    print(f"Nulls: {df.isnull().sum()}")
    print(f"\nNaN values: {df.isna().sum()}")
    print("\n?s:", df.isin(["?"]).sum())

    df = df.mask(df == "?").dropna()

    df["horsepower"] = df["horsepower"].astype(float)
    df["origin"] = pd.Categorical(df["origin"])
    df["cylinders"] = pd.Categorical(df["cylinders"])

    # Drop unused columns and add bias
    df = df.drop(["car name", "model year"], axis=1)
    df["bias"] = 1

    return df


def feature_analysis(df):
    print("\nData Types")
    print(df.dtypes)

    print("\nCorrelation")
    print(df.corr())

    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    model = sm.OLS(y, X).fit()
    print("\nModel Summary")
    print(model.summary())


def feature_eng(df):
    feature_analysis(df)

    # Drop columns with high correlation and/or p-value
    df = df.drop(["displacement"], axis=1)

    feature_analysis(df)
    return df


def sgd_model(X_train, X_test, y_train, y_test):
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
        # "eta0": [0.001, 0.005, 0.01, 0.05, 0.1],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        # "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [1000, 2000, 5000],
        "tol": [1e-3, 1e-4, 1e-5],
    }
    print(f"Number of combinations: {np.prod([len(v) for v in param_grid.values()])}")

    print("\nRunning GridSearchCV...")
    sgd = SGDRegressor()
    grid_search = GridSearchCV(sgd, param_grid, return_train_score=True)  # , verbose=2)
    grid_search.fit(X_train_scaled, y_train)

    best_sgd = grid_search.best_estimator_
    y_train_pred = best_sgd.predict(X_train_scaled)
    y_test_pred = best_sgd.predict(X_test_scaled)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    print("SGDRegressor Results:")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")

    results = pd.DataFrame(grid_search.cv_results_)
    results.sort_values("mean_test_score", ascending=False, inplace=True)
    results.to_excel("sgd_results.xlsx")


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

    print("OLS Results:")
    print(f"Training RMSE: {train_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Training R2: {train_r2:.4f}")
    print(f"Test R2: {test_r2:.4f}")


if __name__ == "__main__":
    df = load_data()

    df = clean_data(df)
    generate_exploration_plots(df)
    df = feature_eng(df)

    # Split data into training and test sets
    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and evaluate models
    sgd_model(X_train, X_test, y_train, y_test)

    # X = pd.concat([X_train, X_test])
    # y = pd.concat([y_train, y_test])
    # ols_model(X, y)
    ols_model(X_train, y_train)
