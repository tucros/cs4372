import logging
import os

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.datasets
import statsmodels.api as sm
import statsmodels.formula.api as smf
import ucimlrepo
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)


def load_data():
    logging.info("Loading data")

    return wine_df


def preprocess(data):
    logging.info("\nPreprocessing.....")
    logging.info(f"{data.shape} Rows and Columns")
    logging.info(data.describe())

    null_rows = data[data.isnull().any(axis=1)]
    logging.info(f"\nRows with null values: {len(null_rows)}")
    if len(null_rows) > 0:
        data.dropna(inplace=True)

    duplicate_rows = data[data.duplicated()]
    logging.info(f"Duplicate rows: {len(duplicate_rows)}")
    if len(duplicate_rows) > 0:
        data.drop_duplicates(inplace=True)

    return data


def explore(data):
    plt.figure(figsize=(10, 6))
    sns.pairplot(data.drop("Region", axis=1))
    plt.savefig("graphs/q1_pairplot.png")

    plt.figure(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig("graphs/q1_correlation_heatmap.png")

    for feature in data.columns[:-1]:
        plt.figure(figsize=(10, 6))
        sns.stripplot(x="Region", y=feature, data=data, jitter=True)
        plt.title(f"{feature} by Region")
        plt.savefig(f"graphs/q1_{feature}_by_region.png")

    return data


def feature_engineer(data):
    full_model = smf.ols(
        "Quality ~ Clarity + Aroma + Body + Flavor + Oakiness + C(Region)",
        data=data,
    ).fit()
    logging.info("\nFull Model Summary:")
    logging.info(full_model.summary().tables[1])

    model = smf.ols("Quality ~ Oakiness + Flavor  + C(Region)", data=data).fit()
    logging.info("\nReduced Model Summary:")
    logging.info(model.summary().tables[1])

    anova = sm.stats.anova_lm(full_model, model)
    logging.info("\nANOVA:")
    logging.info(anova)

    data.drop(["Clarity", "Aroma", "Body"], axis=1, inplace=True)

    # NOTE: No need to scale since we are using tree-based models
    # NOTE: No need to encode categorical variables since we are using tree-based models
    return data


def split(data):
    X = data.drop("Quality", axis=1)
    # NOTE: Convert target to binary representing high quality wines
    y = data["Quality"].apply(lambda x: 1 if x >= 7 else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def visualize_decision_tree(clf, data):
    # TODO: Implement
    pass


def visualize_random_forest(clf, data):
    # TODO: Implement
    pass


def visualize_adaboost(clf, data):
    # TODO: Implement
    pass


def visualize_xgboost(clf, data):
    # TODO: Implement
    pass


# TODO: The param combinations below are just what Copilot generated, may want to adjust
CLASSIFIERS = [
    (
        DecisionTreeClassifier(),
        {
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        visualize_decision_tree,
    ),
    (
        RandomForestClassifier(),
        {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
        },
        visualize_random_forest,
    ),
    (
        AdaBoostClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1],
        },
        visualize_adaboost,
    ),
    (
        XGBClassifier(),
        {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.1, 0.5, 1],
        },
        visualize_xgboost,
    ),
]

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = (
        load_data().pipe(preprocess).pipe(explore).pipe(feature_engineer).pipe(split)
    )

    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    for model, params, viz_fn in CLASSIFIERS:
        logging.info(f"\nTraining {model.__class__.__name__}")
        clf = GridSearchCV(model, params, cv=5, n_jobs=-1)
        clf.fit(X_train, y_train)

        # TODO: All the evaluation code
        # Classifcation report
        # precision, recall and f1 score
        # confusion matrix and save plot to graphs folder
        # ROC curve and save plot to graphs folder
        # Precision-Recall curve and save plot to graphs folder

        viz_fn(clf, X_train)
