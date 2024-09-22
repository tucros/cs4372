import logging
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, precision_recall_curve, precision_recall_fscore_support, auc
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import PrecisionRecallDisplay
import graphviz

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)

# columns = [
#     "fixed acidity", "volatile acidity", "citric acid", "residual sugar",
#     "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
#     "pH", "sulphates", "alcohol", "quality"
# ]


def load_data():
    logging.info("Loading data")
    wine_df = pd.read_csv(
        "https://github.com/tucros/school_datasets/blob/main/winequality-white.csv?raw=true",
        sep=";",
    )
    wine_df["quality"] = wine_df["quality"].astype("category")
    return wine_df


def preprocess(data):
    logging.info("\nPreprocessing.....")
    logging.info(f"{data.shape} Rows and Columns")

    logging.info("\nFrame info")
    logging.info(data.info())

    logging.info("\nFrame description")
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
    # plt.figure(figsize=(20, 20))
    # sns.pairplot(data)
    # plt.savefig("graphs/pairplot.png")

    plt.figure(figsize=(15, 15))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.savefig("graphs/correlation_heatmap.png")

    plt.figure(figsize=(10, 10))
    sns.histplot(data["quality"], bins=10)
    plt.title("Quality Histogram")
    plt.savefig("graphs/quality_histogram.png")

    plt.figure(figsize=(20, 20))
    for i, column in enumerate(data.columns):
        if column != "quality":
            plt.subplot(4, 3, i + 1)
            sns.stripplot(x="quality", y=column, data=data, jitter=True)
            plt.title(f"{column} vs Quality")
    plt.tight_layout()
    plt.savefig("graphs/features_vs_quality_stripplot.png")

    return data


def feature_engineer(data):
    features = data.drop("quality", axis=1)
    target = data["quality"]

    # Determine feature correlation with quality
    alpha = 0.05
    remove_cols = []
    for column in features.columns:
        _, pval = pearsonr(features[column], target)
        logging.debug(f"{column} vs quality pval: {pval}")
        if pval > alpha:
            remove_cols.append(column)

    logging.info(f"Removing {len(remove_cols)} uncorrelated columns")
    if len(remove_cols) > 0:
        logging.info(f"Removing {remove_cols}")
        features.drop(remove_cols, axis=1, inplace=True)

    # NOTE: Convert target to binary representing high quality wines
    data["highquality"] = data["quality"].apply(lambda x: 1 if x >= 7 else 0)

    highquality_percentage = data["highquality"].value_counts(normalize=True) * 100
    logging.info(f"Percentage of each class in 'highquality':\n{highquality_percentage}")

    # NOTE: No need to scale since we are using tree-based models
    return data


def split(data):
    X = data.drop(["quality", "highquality"], axis=1)
    y = data["highquality"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, X_test, y_test


def visualize_decision_tree(clf, data):
    # TODO: Implement
    dot_data = tree.export_graphviz(clf.best_estimator_, out_file=None,
                     feature_names=data.columns,
                     class_names=['0', '1'],
                     filled=True, rounded=True,
                     special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("graphs/decision_tree")
    pass


def visualize_random_forest(clf, data):
    # TODO: Implement
    dot_data = tree.export_graphviz(clf.best_estimator_.estimators_[0], out_file=None,
                        feature_names=data.columns,
                        class_names=['0', '1'],
                        filled=True, rounded=True,
                        special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("graphs/random_forest")
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
        AdaBoostClassifier(algorithm='SAMME'),
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
    if not os.path.exists("graphs"):
        os.makedirs("graphs")

    X_train, y_train, X_test, y_test = (
        load_data().pipe(preprocess).pipe(explore).pipe(feature_engineer).pipe(split)
    )

    for model, params, viz_fn in CLASSIFIERS:
        logging.info(f"\nTraining {model.__class__.__name__}")
        clf = GridSearchCV(model, params, cv=5, n_jobs=-1, scoring="accuracy")
        clf.fit(X_train, y_train)

        # TODO: All the evaluation code
        logging.info(f"Best parameters: {clf.best_params_}")
        logging.info(f"Best Training Accuracy for {model.__class__.__name__}: {clf.best_score_}")

        # Classifcation report
        # precision, recall and f1 score
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logging.info(f"Testing Accuracy: {accuracy}")

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
        logging.info(f"Precision: {precision}")
        logging.info(f"Recall: {recall}")
        logging.info(f"F1 Score: {f1_score}")

        logging.info(f"Classification Report: \n{classification_report(y_test, y_pred)}")

        # confusion matrix and save plot to graphs folder
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {model.__class__.__name__}')
        plt.savefig(f"graphs/confusion_matrix_{model.__class__.__name__}.png")
        plt.close()

        # ROC curve and save plot to graphs folder
        fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
        plt.figure(figsize=(10, 7))
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker='.', label=f'AUC = {roc_auc:.2f}')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r', label=f'AUC = 0.5')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve for {model.__class__.__name__}')
        plt.legend(loc='lower right')
        plt.savefig(f"graphs/roc_curve_{model.__class__.__name__}.png")
        plt.close()

        # Precision-Recall curve and save plot to graphs folder
        class_counts = y_test.value_counts(normalize=True) * 100
        logging.info(f"Percentage of each class in y_test:\n{class_counts}")

        precision_recall_display = PrecisionRecallDisplay.from_estimator(clf, X_test, y_test)
        precision_recall_display.plot()
        plt.axhline(y=class_counts[1] / 100, color='r', linestyle='--', label='Baseline Classifier')
        plt.legend(loc='lower left')
        plt.title(f'Precision-Recall curve for {model.__class__.__name__}')
        plt.savefig(f"graphs/precision_recall_curve_{model.__class__.__name__}.png")
        plt.close()

        viz_fn(clf, X_train)
