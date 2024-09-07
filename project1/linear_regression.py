import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


# Importing the dataset
def load_data():
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
    url = "https://github.com/williamphyoe/datasets/blob/main/auto-mpg.csv?raw=true"
    df = pd.read_csv(url, header=None, names=columns, sep="\s+")

    # Drop unused columns and add bias
    df = df.drop(["car name", "model year"], axis=1)
    df["bias"] = 1

    return df


def generate_exploration_plots(df):
    plt.figure(figsize=(20, 20))

    # Pairplot for numerical columns
    numerical_columns = df.select_dtypes(include=[np.number]).columns
    sns.pairplot(df[numerical_columns], diag_kind="kde")
    plt.suptitle("Pairplot of Numerical Variables", y=1.02)
    plt.show()

    # Boxplots for each numerical variable
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(numerical_columns, 1):
        plt.subplot(3, 3, i)
        sns.boxplot(x=df[column])
        plt.title(f"Boxplot of {column}")
    plt.tight_layout()
    plt.show()

    # Heatmap of correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

    # Countplot for 'origin' (categorical variable)
    plt.figure(figsize=(10, 6))
    sns.countplot(x="origin", data=df)
    plt.title("Count of Cars by Origin")
    plt.show()

    # Scatterplot of MPG vs Weight, with color based on Origin
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

    # Drop columns with high correlation and p-value
    df = df.drop(["displacement"], axis=1)

    # print(df)
    feature_analysis(df)
    return df


def sgd_model(X_train, X_test, y_train, y_test):
    # Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define hyperparameters to tune
    param_grid = {
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "optimal", "invscaling", "adaptive"],
        "loss": ["squared_error", "huber", "epsilon_insensitive"],
        "penalty": ["l2", "l1", "elasticnet"],
        "max_iter": [1000, 2000, 5000],
    }

    # Define scoring metrics
    scoring = {"r2": "r2", "neg_root_mean_squared_error": "neg_root_mean_squared_error"}

    # Perform grid search
    sgd = SGDRegressor(random_state=42)
    grid_search = GridSearchCV(
        sgd,
        param_grid,
        cv=5,
        scoring=scoring,
        refit="neg_root_mean_squared_error",
        return_train_score=True,
    )
    grid_search.fit(X_train_scaled, y_train)

    # Get best model
    best_sgd = grid_search.best_estimator_

    # Make predictions
    y_train_pred = best_sgd.predict(X_train_scaled)
    y_test_pred = best_sgd.predict(X_test_scaled)

    # Calculate metrics
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

    # Create a DataFrame with all parameter combinations and their scores
    results = pd.DataFrame(grid_search.cv_results_)

    # Convert negative RMSE to positive RMSE
    results["mean_test_rmse"] = -results["mean_test_neg_root_mean_squared_error"]
    results["mean_train_rmse"] = -results["mean_train_neg_root_mean_squared_error"]


def ols_model(X_train, y_train):
    # Add constant term to the features
    X_train_sm = sm.add_constant(X_train)

    # Fit OLS model
    model = sm.OLS(y_train, X_train_sm).fit()

    # Print summary
    print("\nOLS Model Summary:")
    print(model.summary())

    # Interpret results
    print("\nOLS Model Interpretation:")
    print("1. R-squared:", model.rsquared)
    print(
        "   - This indicates that {:.2f}% of the variance in MPG is explained by our model.".format(
            model.rsquared * 100
        )
    )

    print("\n2. Adjusted R-squared:", model.rsquared_adj)
    print(
        "   - This adjusted value accounts for the number of predictors in the model."
    )

    print("\n3. F-statistic:", model.fvalue)
    print("   - The F-statistic tests the overall significance of the model.")
    print("   - P-value for F-statistic:", model.f_pvalue)
    print(
        "   - A very low p-value suggests that the model is statistically significant."
    )

    print("\n4. Coefficients:")
    for name, coef, std_err, p_value in zip(
        model.model.exog_names, model.params, model.bse, model.pvalues
    ):
        print(f"   - {name}:")
        print(f"     Coefficient: {coef:.4f}")
        print(f"     Std. Error: {std_err:.4f}")
        print(f"     P-value: {p_value:.4f}")
        if p_value < 0.05:
            print("     This variable is statistically significant at the 5% level.")
        else:
            print(
                "     This variable is not statistically significant at the 5% level."
            )
        print()


if __name__ == "__main__":
    df = load_data()
    # generate_exploration_plots(df)

    # Preprocess data
    df = df.pipe(clean_data).pipe(feature_eng)

    # Split data into training and test sets
    X = df.drop("mpg", axis=1)
    y = df["mpg"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Train and evaluate models
    sgd_model(X_train, X_test, y_train, y_test)
    ols_model(X_train, y_train)
