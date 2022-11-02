"""
Report module.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.simplefilter("ignore")


def prepare_data(
    dataframe: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    encoder=LabelEncoder(),
    only_num=True,
) -> pd.DataFrame:
    """
    Prepare data to research;
    :param dataframe: datagrame;
    :param numeric_cols: numerical features;
    :param categorical_cols: categorical features;
    :param encoder: encoder method;
    :param only_num: if only numerical features;
    :return: dataframe.
    """
    for column in dataframe.columns:
        if column in numeric_cols:
            dataframe[column] = dataframe[column].fillna(
                np.median(dataframe[column].dropna())
            )
        elif column in categorical_cols:
            dataframe[column] = dataframe[column].fillna("unknown")
            dataframe[column] = encoder.fit_transform(dataframe[column])
    if only_num:
        return dataframe[numeric_cols]
    return dataframe


with open("result.txt", "w+", encoding="UTF-8") as file:
    db_train = pd.read_csv("../data/train.csv")
    db_test = pd.read_csv("../data/holdout.csv")

    file.write("TRAIN SHAPE:\n")
    file.write(str(db_train.shape) + 2 * "\n")

    plt.title("target distribution")
    sns.countplot(x=db_train["raintomorrow"])
    plt.savefig("target_distribution.pdf")

    file.write("NANs:\n")
    file.write(str(db_train.isnull().sum()) + 2 * "\n")

    plt.figure(figsize=(16, 14))
    plt.suptitle("possible ouliers")
    plt.subplot(321)
    sns.boxplot(x=db_train["pressure3pm"].dropna())
    plt.subplot(322)
    sns.boxplot(x=db_train["humidity3pm"].dropna())
    plt.subplot(323)
    sns.boxplot(x=db_train["windgustspeed"].dropna())
    plt.subplot(324)
    sns.boxplot(x=db_train["rainfall"].dropna())
    plt.subplot(325)
    sns.boxplot(x=db_train["windspeed9am"].dropna())
    plt.subplot(326)
    sns.boxplot(x=db_train["humidity9am"].dropna())
    plt.savefig("possible_outliers.pdf")

    num_cols = list(db_train.select_dtypes(include=np.number).columns)
    cat_cols = list(db_train.select_dtypes(include="object").columns)
    file.write("NUMERICAL FEATURES:\n")
    file.write(str(num_cols) + 2 * "\n")
    file.write("CATEGORICAL FEATURES:\n")
    file.write(str(cat_cols) + 2 * "\n")

    x_train = prepare_data(db_train, num_cols, cat_cols, only_num=False)
    x_test = prepare_data(db_test, num_cols, cat_cols, only_num=False)

    y_train = x_train["raintomorrow"]
    x_train = x_train.drop(["raintomorrow"], axis=1)
    y_test = x_test["raintomorrow"]
    x_test = x_test.drop(["raintomorrow"], axis=1)
    num_cols.remove("raintomorrow")

    correlations = pd.concat((x_train[num_cols], y_train), axis=1).corr()
    plt.figure(figsize=(15, 10))
    plt.title("correlation matrix")
    sns.heatmap(correlations, annot=True, cmap="crest")
    plt.savefig("correlation_matrix.pdf")

    selector = RandomForestClassifier(n_jobs=-1)
    selector.fit(x_train, y_train)
    feature_importances = selector.feature_importances_

    plt.figure(figsize=(14, 8))
    plt.title("feature importances")

    features = x_train.columns

    sns.barplot(y=features, x=feature_importances)
    plt.savefig("feature_importances.pdf")
