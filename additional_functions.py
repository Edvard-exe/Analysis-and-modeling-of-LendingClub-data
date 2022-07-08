import pandas as pd
import matplotlib.pyplot as plt
from pywaffle import Waffle
import plotly.express as px
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from calendar import month_abbr
from math import pi
import math
import numpy as np
import optuna
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import (
    recall_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    f1_score,
    average_precision_score,
    precision_score,
)
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import torch.nn as nn
import warnings


def drop_null(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes to Data Frames and drops null values from column which have
    less than 20 000 Nans
    """

    for column in df:
        if sum(df[column].isnull()) <= 20000 and sum(df[column].isnull()) != 0:
            df = df.dropna(subset=[df[column].name])

    return df


def emp_length_to_int(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Takes to Data Frames and column name.
    Converts str to int.
    Returns new dataframe
    """

    df.replace({column: {"< 1 year": 0, "10+ years": 10}})

    df[column] = df[column].str.extract("(\d+)")

    return df


def cat_to_int(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Takes to Data Frames and column name.
    Converts cat to int.
    Returns new dataframe
    """

    for column in columns:
        df[column] = df[column].str.extract(r"(\d*\.\d+|\d+)").astype(float)

    return df


def drop_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Takes to Data Frames and list of columns.
    Drops those columns
    Returns new dataframe
    """

    df = df.drop(columns=columns)

    return df


def group_by(df: pd.DataFrame, lists: list, column: str) -> pd.DataFrame:
    """
    Takes to Data Frames, list of columns and column name.
    Groups by the list and calculates total amount.
    Returns new dataframe
    """

    new_df = df.groupby(lists).agg(total=(column, "count")).reset_index()

    return new_df


def replace_int(
    df: pd.DataFrame, column: str, value_1: int, value_2: int, value_3: int
):

    df.loc[(df[column] < value_1) | (df[column] > value_2), column] = value_3

    return df


def replace_emp_length(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.replace(
        {
            column: {
                "4 years": 4,
                "< 1 year": 0,
                "1 year": 1,
                "3 years": 3,
                "2 years": 2,
                "10+ years": 10,
                "9 years": 9,
                "5 years": 5,
                "7 years": 7,
                "6 years": 6,
                "8 years": 8,
            }
        }
    )

    return df


def cut(df: pd.DataFrame, column_1: str, column_2: str) -> pd.DataFrame:
    """
    Takes to Data Frame, old column name and new column name.
    Cuts data into groups.
    Returns new dataframe.
    """

    df[column_2] = pd.cut(
        x=df[column_1],
        bins=[299, 575, 669, 739, 799, 860],
        labels=[
            "300 - 575",
            "580 - 669",
            "670 - 739",
            "740 - 799",
            "800 - 850",
        ],
    )

    return df


def word_cloud(df: pd.DataFrame, column: str) -> None:
    """
    Takes to Data Frame and column name.
    Drops stop words.
    Returns wordcloud png.
    """

    loan_words = [
        "my",
        "and",
        "of",
        "a",
        "for",
        "will",
        "Pay",
        "In",
        "I",
        "back",
        "up",
        "new",
        "out",
        "consolidation",
        "buying",
        "major",
    ]

    stopwords = set(STOPWORDS)
    stopwords.update(loan_words)

    text = " ".join(title.lower() for title in df[column])

    wordcloud = WordCloud(
        width=1600,
        height=800,
        random_state=1,
        background_color="#5651c5",
        colormap="Greens_r",
        collocations=False,
        stopwords=stopwords,
    ).generate(text)

    plt.figure(figsize=(16, 7))
    plt.imshow(wordcloud)


def merging(f_df: pd.DataFrame, s_df: pd.DataFrame, l_on: list) -> pd.DataFrame:
    """
    Takes to Data Frames and merges on a given list of row names
    """

    to_merge = pd.merge(f_df, s_df, how="inner", on=l_on)

    return to_merge


def group_for_hypothesis(
    df: pd.DataFrame, column_1: str, column_2: str, column_3: str
) -> pd.DataFrame:

    df = pd.DataFrame(
        df.groupby([column_1]).agg(
            mean=(column_2, "mean"),
            std=(column_2, "std"),
            sample_size=(column_3, "count"),
        )
    ).reset_index()

    return df


def median_two_columns(
    df: pd.DataFrame, column_1: str, column_2: str, column_3: str
) -> pd.DataFrame:

    df[column_3] = (df[column_1] + df[column_2]) / 2

    return df


def ys_column(df: pd.DataFrame, old_column: str, new_column: str) -> pd.DataFrame:

    df[new_column] = df[old_column]
    df[new_column] = df[new_column].fillna(0)
    df.loc[df[new_column] != 0, new_column] = 1

    return df


def replace_zero(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for column in columns:
        df[column] = df[column].fillna(0)

    return df


def replace_uknown(df: pd.DataFrame, columns: list) -> pd.DataFrame:

    for column in columns:
        df[column] = df[column].fillna("uknown")

    return df


def employment_acc(df: pd.DataFrame, columns: str) -> pd.DataFrame:

    df[columns] = df[columns].str.lower()
    df[columns] = df[columns].astype(str)

    engineering = [
        "engineer",
        "technician",
        "mechanic",
        "technologist",
        "architect",
        "mechanical",
    ]

    artist = [
        "artist",
        "art",
        "writer",
        "editor",
        "painter",
        "producer",
        "media",
        "design",
        "designer",
        "culture",
        "entertainment",
        "stylist",
        "teller",
    ]

    business = [
        "sales",
        "director",
        "manager",
        "director",
        "associate",
        "business",
        "account",
        "consultant",
        "president",
        "management",
        "owner",
        "administrator",
        "administrative",
        "executive",
        "financial",
        "banker",
        "vp",
        "receptionist",
        "ceo",
        "dealer",
        "manger",
        "administration",
        "assistant",
        "finance",
        "cfo",
        "hr",
        "trainer",
        "bank",
        "planner",
        "recruiter",
        "training",
        "agent",
        "instructor",
        "secretary",
        "buyer",
        "human",
        "realtor",
        "broker",
    ]

    education = [
        "school",
        "university",
        "education",
        "physician",
        "coach",
        "schools",
        "dean",
        "professor",
        "teacher",
        "principal",
        "bookkeeper",
        "educator",
        "librarian",
        "caregiver",
        "examiner",
    ]

    government = [
        "officer",
        "clerk",
        "police",
        "inspector",
        "counselor",
        "sergeant",
        "investigator",
        "firefighter",
        "sheriff",
        "fire",
        "federal",
        "detective",
        "deputy",
        "captain",
        "army",
    ]

    health = [
        "nurse",
        "therapist",
        "medical",
        "health",
        "clinical",
        "hospital",
        "dental",
        "patient",
        "hygienist",
        "pharmacist",
        "pharmacy",
        "paramedic",
        "nursing",
        "healthcare",
        "pathologist",
        "care",
    ]

    tech = [
        "analyst",
        "tech",
        "senior",
        "lead",
        "support",
        "it",
        "developer",
        "server",
        "admin",
        "software",
        "programmer",
        "data",
        "processor",
        "system",
        "computer",
        "scientist",
        "technology",
        "network",
        "web",
        "tech.",
        "chemist",
    ]

    worker = [
        "specialist",
        "truck",
        "maintenance",
        "foreman",
        "electrician",
        "services",
        "production",
        "machine",
        "controller",
        "carrier",
        "warehouse",
        "machinist",
        "bus",
        "cashier",
        "attendant",
        "dispatcher",
        "carpenter",
        "welder",
        "flight",
        "delivery",
        "shipping",
        "forklift",
        "journeyman",
        "transportation",
        "assembler",
        "pilot",
        "assembly",
        "conductor",
        "crew",
        "courier",
        "electrical",
        "worker",
        "laborer",
        "handler",
        "logistics",
        "construction",
        "labor",
        "plumber",
        "electric",
        "air",
        "loader",
    ]

    law = [
        "representative",
        "attorney",
        "paralegal",
        "superintendent",
        "advisor",
        "insurance",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
    ]

    all_words = (
        law
        + worker
        + tech
        + health
        + government
        + education
        + business
        + artist
        + engineering
    )

    df.loc[~df[columns].str.contains("|".join(all_words)), columns] = "Else"
    df.loc[df[columns].str.contains("|".join(engineering)), columns] = "Engineering"
    df.loc[
        df[columns].str.contains("|".join(artist)), columns
    ] = "Arts, culture and entertainment"
    df.loc[
        df[columns].str.contains("|".join(business)), columns
    ] = "Business, management and administration"
    df.loc[df[columns].str.contains("|".join(education)), columns] = "Education"
    df.loc[df[columns].str.contains("|".join(government)), columns] = "Government"
    df.loc[df[columns].str.contains("|".join(health)), columns] = "Health and medicine"
    df.loc[df[columns].str.contains("|".join(tech)), columns] = "Science and technology"
    df.loc[
        df[columns].str.contains("|".join(worker)), columns
    ] = "Maintenance, workers and transport"
    df.loc[df[columns].str.contains("|".join(law)), columns] = "Law"

    return df


def loan_reason(df: pd.DataFrame, columns: str) -> pd.DataFrame:

    df[columns] = df[columns].str.lower()
    df[columns] = df[columns].astype(str)

    small_business = [
        "refinancing",
        "financing",
        "business",
        "small_business",
        "refinance",
        "starting",
        "project",
        "startup",
        "start-up",
        "company",
        "shop",
        "bussiness",
        "bussines",
        "production",
        "entrepreneur",
    ]

    debt_consolidation = [
        "debt!",
        "consolodation",
        "dept",
        "debts",
        "consolidating",
        "payoff",
        "consolidate",
        "debt_consolidation",
        "consolidation",
        "debt",
        "consolidated",
        "consilidation",
        "consolodate",
        "loan",
    ]

    major_purchase = [
        "buying",
        "purchase",
        "major_purchase",
        "expenses",
        "personal",
        "motorcycle",
        "life",
        "club",
        "buy",
        "dream",
        "christmas",
        "restaurant",
        "bike",
        "purchasing",
        "boat",
        "computer",
        "yamaha",
        "engine",
    ]

    credit_card = ["creditcard", "debit", "credit_card", "card", "credit"]

    home_improvement = [
        "home_improvement",
        "repairs",
        "improvements",
        "kitchen",
        "roof",
        "improve",
        "room",
        "renovation",
        "bathroom",
        "swimming",
        "basement",
        "fixing",
        "furniture",
        "building",
        "pool",
        "repair",
        "fix",
    ]

    house = [
        "land",
        "housing",
        "rental",
        "apartment",
        "estate",
        "property",
        "home",
        "rent",
        "studio",
    ]
    vacation = ["trip", "travel", "trailer", "vacation"]

    medical = [
        "medical",
        "dental",
        "surgery",
        "nursing",
        "care",
        "rehab",
        "health",
        "hospital",
        "cancer",
    ]

    car = [
        "car",
        "truck",
        "track",
        "vehicle",
        "honda",
        "auto",
        "automobile",
        "bmw",
        "toyota",
        "ford",
    ]

    educational = [
        "teacher",
        "study",
        "learning",
        "university",
        "college",
        "education",
        "educational",
        "student",
        "school",
        "tuition",
        "degree",
        "course",
        "graduation",
        "academy",
    ]

    moving = ["moving", "move", "relocation", "transportation"]

    renewable_energy = [
        "energy",
        "solar",
        "green",
        "renewable_energy",
        "advisor",
        "insurance",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
    ]

    wedding = [
        "married",
        "wife",
        "ring",
        "engagement",
        "divorce",
        "wedding",
        "county",
        "legal",
        "adjuster",
        "billing",
        "law",
        "marriage",
        "honeymoon",
    ]

    other = (
        wedding
        + renewable_energy
        + moving
        + educational
        + car
        + vacation
        + vacation
        + house
        + home_improvement
        + credit_card
        + major_purchase
        + debt_consolidation
        + small_business
    )

    df.loc[~df[columns].str.contains("|".join(other)), columns] = "other"
    df.loc[df[columns].str.contains("|".join(wedding)), columns] = "wedding"
    df.loc[
        df[columns].str.contains("|".join(renewable_energy)), columns
    ] = "renewable_energy"
    df.loc[df[columns].str.contains("|".join(educational)), columns] = "educational"
    df.loc[df[columns].str.contains("|".join(moving)), columns] = "moving"
    df.loc[df[columns].str.contains("|".join(medical)), columns] = "medical"
    df.loc[df[columns].str.contains("|".join(car)), columns] = "car"
    df.loc[
        df[columns].str.contains("|".join(small_business)), columns
    ] = "small_business"
    df.loc[
        df[columns].str.contains("|".join(debt_consolidation)), columns
    ] = "debt_consolidation"
    df.loc[
        df[columns].str.contains("|".join(major_purchase)), columns
    ] = "major_purchase"
    df.loc[df[columns].str.contains("|".join(credit_card)), columns] = "credit_card"
    df.loc[
        df[columns].str.contains("|".join(home_improvement)), columns
    ] = "home_improvement"
    df.loc[df[columns].str.contains("|".join(house)), columns] = "house"
    df.loc[df[columns].str.contains("|".join(vacation)), columns] = "vacation"

    return df


def drop_spec(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    for column in columns:
        df = df[df[column].notna()]

    return df


def date_extract(
    df: pd.DataFrame, column_old: str, column_new1: str, column_new2: str
) -> pd.DataFrame:

    df[column_new1] = df[column_old].astype(str).str[4:]
    df[column_new1] = df[column_new1].astype(int)
    df[column_new2] = df[column_old].astype(str).str[:3]

    lower_ma = [m.lower() for m in month_abbr]
    df[column_new2] = (
        df[column_new2].str.lower().map(lambda m: lower_ma.index(m)).astype("int")
    )

    return df


def rejected_date_separate(
    df, old_column: str, new_column1: str, new_column2: str
) -> pd.DataFrame:

    df[new_column1] = df[old_column].astype(str).str[5:7]
    df[new_column1] = df[new_column1].astype(int)

    df[new_column2] = df[old_column].astype(str).str[:4]
    df[new_column2] = df[new_column2].astype(int)

    return df


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    df.rename(
        columns={
            "Amount Requested": "loan_amnt",
            "Loan Title": "purpose",
            "Risk_Score": "fico_range",
            "Debt-To-Income Ratio": "dti",
            "State": "addr_state",
            "Employment Length": "emp_length",
        },
        inplace=True,
    )

    return df


def status(df: pd.DataFrame, value: int) -> pd.DataFrame:

    df["status"] = value

    return df


def transformation_mnths(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Takes list of month column values.
    Converts them to sin and coss values.
    Returns two new lists.
    """
    max_value = df[column].max()
    df["sin_mnths"] = [math.sin((2 * pi * x) / max_value) for x in list(df[column])]
    df["cos_mnths"] = [math.cos((2 * pi * x) / max_value) for x in list(df[column])]

    return df


def column_index(df, query_cols):
    cols = df.columns.values
    sidx = np.argsort(cols)
    return sidx[np.searchsorted(cols, query_cols, sorter=sidx)]


def base_line(
    X: pd.DataFrame, y: pd.DataFrame, numeric_scaler: np.array, preprocessor: np.array
) -> pd.DataFrame:
    """
    Takes x and y dataframes.
    Perform cross validation with different models.
    Returns table with metrics and results
    """
    balanced_accuracy = []
    roc_auc = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    kfold = StratifiedKFold(n_splits=5)
    classifiers = [
        "XGB classifier",
        "Random Forest",
        "Logistic regression",
        "LGBM classifier",
    ]
    models = [
        XGBClassifier(),
        RandomForestClassifier(n_estimators=100),
        LogisticRegression(),
        LGBMClassifier(),
    ]
    for model in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("numeric_scaler", numeric_scaler),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
                "roc_auc",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())
        roc_auc.append(result["test_roc_auc"].mean())
    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Roc Auc": roc_auc,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="PiYG_r")
    return base_models


def base_line_multi(
    X: pd.DataFrame, y: pd.DataFrame, preprocessor: np.array
) -> pd.DataFrame:
    """
    Takes x and y dataframes.
    Perform cross validation with different models.
    Returns table with metrics and results
    """
    balanced_accuracy = []
    accuracy = []
    recall = []
    precision = []
    f1_score = []
    fit_time = []
    kfold = StratifiedKFold(n_splits=5)
    classifiers = ["XGB classifier", "Logistic regression", "LGBM classifier"]
    models = [
        XGBClassifier(),
        LogisticRegression(),
        LGBMClassifier(),
    ]
    for model in models:
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )
        result = cross_validate(
            pipeline,
            X,
            y,
            cv=kfold,
            scoring=(
                "balanced_accuracy",
                "accuracy",
                "f1_macro",
                "recall_macro",
                "precision_macro",
            ),
        )
        fit_time.append(result["fit_time"].mean())
        balanced_accuracy.append(result["test_balanced_accuracy"].mean())
        accuracy.append(result["test_accuracy"].mean())
        recall.append(result["test_recall_macro"].mean())
        precision.append(result["test_precision_macro"].mean())
        f1_score.append(result["test_f1_macro"].mean())

    base_models = pd.DataFrame(
        {
            "Balanced accuracy": balanced_accuracy,
            "Accuracy": accuracy,
            "Recall": recall,
            "Precision": precision,
            "f1": f1_score,
            "Fit_time": fit_time,
        },
        index=classifiers,
    )
    base_models = base_models.style.background_gradient(cmap="PiYG_r")
    return base_models


def LGBM_objective(trial, X, y, preprocessor) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    model = LGBMClassifier(
        objective="binary",
        **param_grid,
    )

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    cv_scores = np.empty(5)
    for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[test_idx]

        X_train = preprocessor.fit_transform(X_train, y_train)
        X_valid = preprocessor.transform(X_valid)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        early_stopping_rounds=100,
        callbacks=[optuna.integration.LightGBMPruningCallback(trial, "binary_logloss")],
    )

    preds = model.predict(X_valid)
    cv_scores[idx] = log_loss(y_valid, preds)

    return np.mean(cv_scores)


def LGBM_objective_multi(trial, X, y, preprocessor):
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    model = LGBMClassifier(
        objective="multiclass", class_weight="balanced", **param_grid, verbose=-1
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    X_train = preprocessor.fit_transform(X_train, y_train)
    X_valid = preprocessor.transform(X_valid)

    model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = model.predict(X_valid)

    return f1_score(y_valid, yhat, average="macro")


def lgbm_multi_sub(
    trial,
    X,
    y,
):
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 200, 10000, step=100),
        "lambda_l1": trial.suggest_int("lambda_l1", 0, 100, step=5),
        "lambda_l2": trial.suggest_int("lambda_l2", 0, 100, step=5),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 15),
    }

    model = LGBMClassifier(
        objective="multiclass", class_weight="balanced", verbosity=-1, **param_grid
    )

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    predictor = model.fit(
        X_train,
        y_train,
        early_stopping_rounds=100,
        eval_set=[(X_valid, y_valid)],
        verbose=False,
    )

    yhat = predictor.predict(X_valid)

    trial.set_user_attr(key="best_booster", value=predictor)

    return f1_score(y_valid, yhat, average="macro")


def train_lgbm_subgrade(grade, df, preprocessor):
    per_grade = df[df["grade"] == grade]
    y_s = per_grade["sub_grade"]
    X_s = per_grade.drop(["sub_grade", "grade"], axis=1)
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X_s, y_s, test_size=0.20, stratify=y_s, random_state=42
    )

    X_train_tr = preprocessor.fit_transform(X_train_s)
    X_test_tr = preprocessor.transform(X_test_s)

    study_lgbm_subgrade = optuna.create_study(direction="maximize")
    study_lgbm_subgrade.optimize(
        lambda trial: objective_lgbm_subgrade(trial, X_train_tr, y_train_s),
        n_trials=20,
        show_progress_bar=True,
        callbacks=[callback],
    )

    best_value = round(study_lgbm_subgrade.best_value, 3)
    lgbm_hp = study_lgbm_subgrade.best_params
    best_cls = study_lgbm_subgrade.user_attrs["best_booster"]

    subgrade_pred = best_cls.predict(X_test_tr)

    metric_dict = {
        "Grade": grade,
        "F1-score": f1_score(y_test_s, subgrade_pred, average="macro"),
        "Precision": precision_score(y_test_s, subgrade_pred, average="macro"),
        "Recall": recall_score(y_test_s, subgrade_pred, average="macro"),
    }

    return best_value, lgbm_hp, metric_dict, best_cls


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_booster", value=trial.user_attrs["best_booster"])


def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """

    # Remove the internal helper function
    # check_is_fitted(column_transformer)

    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == "drop" or (hasattr(column, "__len__") and not len(column)):
            return []
        if trans == "passthrough":
            if hasattr(column_transformer, "_df_columns"):
                if (not isinstance(column, slice)) and all(
                    isinstance(col, str) for col in column
                ):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ["x%d" % i for i in indices[column]]
        if not hasattr(trans, "get_feature_names"):
            # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn(
                "Transformer %s (type %s) does not "
                "provide get_feature_names. "
                "Will return input column names if available"
                % (str(name), type(trans).__name__)
            )
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [f for f in column]

        return [f for f in trans.get_feature_names()]

    ### Start of processing
    feature_names = []

    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [
            (name, trans, None, None)
            for step, name, trans in column_transformer._iter()
        ]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))

    for name, trans, column, _ in l_transformers:
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names) == 0:
                _names = [f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))

    return feature_names


def get_calibration_curve_values(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    prob_pos = model.predict_proba(X_test)[:, 1]
    model_score = brier_score_loss(y_test, y_pred, pos_label=y.max())

    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_test, prob_pos, n_bins=10
    )

    return (fraction_of_positives, mean_predicted_value, model_score)


def xgboost_objective(trial, X, y, preprocessor) -> dict:
    """
    Takes x and y and dataframes
    Performs model training.
    Returns best parameters for particular model.
    """

    param_grid = {
        "n_estimators": trial.suggest_int("n_estimators", 0, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 20, 3000, step=20),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-3, 100),
        "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-3, 100),
        "gama": trial.suggest_float("gama", 0, 20),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 1, 100),
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = np.empty(5)

    for idx, (train_idx, eval_idx) in enumerate(cv.split(X, y)):
        X_train, X_eval = X.iloc[train_idx], X.iloc[eval_idx]
        y_train, y_eval = y.iloc[train_idx], y.iloc[eval_idx]

        X_train = preprocessor.fit_transform(X_train)
        X_eval = preprocessor.transform(X_eval)

        model = XGBClassifier(**param_grid, tree_method="gpu_hist")
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_eval, y_eval)],
            eval_metric="logloss",
            early_stopping_rounds=100,
            verbose=False,
        )
        preds = model.predict(X_eval)
        cv_scores[idx] = f1_score(y_eval, preds, average="macro")

    return np.mean(cv_scores)


def cat_to_torch(df: pd.DataFrame) -> list:

    onehot_features = ["purpose", "emp_length"]

    for category in onehot_features:
        df[category] = df[category].astype("category")

    purp = df["purpose"].cat.codes
    emp = df["emp_length"].cat.codes

    categorical_data = np.stack([purp, emp], 1)

    categorical_data = torch.tensor(categorical_data, dtype=torch.int64)

    return categorical_data


def test_data(categorical_data, numerical_data, outputs) -> list:
    total_records = 2789679
    test_records = int(total_records * 0.2)

    categorical_train_data = categorical_data[: total_records - test_records]
    categorical_test_data = categorical_data[
        total_records - test_records : total_records
    ]
    numerical_train_data = numerical_data[: total_records - test_records]
    numerical_test_data = numerical_data[total_records - test_records : total_records]
    train_outputs = outputs[: total_records - test_records]
    test_outputs = outputs[total_records - test_records : total_records]

    return (
        categorical_train_data,
        categorical_test_data,
        numerical_train_data,
        numerical_test_data,
        train_outputs,
        test_outputs,
    )


class Plots:
    def __init__(self, first_df, second_df):

        self.first_df = first_df
        self.second_df = second_df

    def population_plot(self):

        diffrence = self.second_df.shape[0] // self.first_df.shape[0]
        data = {"Unapproved loans": 20 * diffrence, "Approved loans": 20}
        fig = plt.figure(
            FigureClass=Waffle,
            rows=5,
            values=data,
            colors=["#5651c5", "#2e8b57"],
            legend={"loc": "upper left", "bbox_to_anchor": (1, 1), "fontsize": 18},
            icons="child",
            font_size=15,
            figsize=(16, 7),
            icon_legend=True,
        )

        plt.show()

    def map_plot(self):

        fig = px.scatter_geo(
            locations=self.first_df["addr_state"],
            locationmode="USA-states",
            scope="usa",
            size=self.first_df["total"],
        )

        fig.update_traces(
            marker=dict(
                size=self.first_df["total"] * 2,
                opacity=0.8,
                symbol="circle",
                color="#2e8b57",
                cmin=0,
                line_color="rgb(40,40,40)",
                line_width=0.5,
                colorbar_title="",
                showscale=False,
            )
        )

        fig.update_layout(
            title_text="States with the largest number <br> of approved and unapproved loans",
        )

        fig_b = px.scatter_geo(
            locations=self.second_df["State"],
            locationmode="USA-states",
            scope="usa",
            size=self.second_df["total"],
            labels={"total": "all rejected"},
        )

        fig_b.update_traces(
            marker=dict(
                size=self.second_df["total"] * 3,
                opacity=0.8,
                symbol="circle",
                color="#5651c5",
                cmin=0,
                line_color="rgb(40,40,40)",
                line_width=0.5,
                cmax=self.second_df["total"].max(),
                colorbar_title="",
                showscale=False,
            )
        )

        fig.add_trace(fig_b.data[0])

        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=0.95, xanchor="right", x=0.9
            )
        )
        fig["data"][0]["showlegend"] = True
        fig["data"][1]["showlegend"] = True
        fig["data"][0]["name"] = "Rejected loans"
        fig["data"][1]["name"] = "Accepted loans"
        fig.show()

    def dist_plot(self, column, x_label):
        plt.figure(figsize=(14, 6))
        sns.set_style("ticks")

        ax = sns.kdeplot(
            self.first_df[column],
            color="#2e8b57",
            shade=True,
        )
        ax.set(xlabel=x_label)
        sns.despine()
        plt.xlim(0)
        plt.show(ax)

    def kde_plot(
        self,
        column_1: str,
        column_2: str,
        x_name: str,
        first_label: str,
        second_label: str,
    ) -> None:
        """
        Takes labels of two selected columns.
        Selects rows with only a certain value and from certain columns.
        Returns graphs of the distribution of values with a specific x-axis name.
        """
        sns.set_style("ticks")
        fig, ax1 = plt.subplots(1, figsize=(16, 6))

        sns.kdeplot(
            self.first_df[column_1],
            color="#2e8b57",
            shade=True,
            label=first_label,
            ax=ax1,
        )

        sns.kdeplot(
            self.second_df[column_2],
            color="#5651c5",
            shade=True,
            label=second_label,
            ax=ax1,
        )

        ax1.set_xlabel(x_name, fontsize=14)
        ax1.set_ylabel("Density", fontsize=14)
        sns.despine()
        plt.legend(title_fontsize=13)
        plt.xlim(0)
        plt.show()

    def strip_plot_sub(
        self,
        column_1: str,
        column_2: str,
        column_3: str,
        column_4: str,
        title_1: str,
        title_2: str,
    ) -> None:
        fig, ax = plt.subplots(1, 2, figsize=(16, 7))
        sns.set_style("ticks")

        sns.stripplot(
            data=self.first_df,
            x=column_1,
            y=column_2,
            jitter=0.25,
            size=8,
            linewidth=0.2,
            edgecolor="black",
            ax=ax[0],
            palette="PRGn",
        )

        sns.stripplot(
            data=self.second_df,
            x=column_3,
            y=column_4,
            jitter=0.25,
            size=8,
            linewidth=0.2,
            edgecolor="black",
            ax=ax[1],
            palette="PRGn",
        )
        ax[0].set(xlabel=column_3, ylabel=column_4)
        ax[1].set(xlabel=column_3, ylabel=" ")
        ax[0].title.set_text(title_1)
        ax[1].title.set_text(title_2)
        sns.despine()
        plt.show()

    def line_plot(self, column_1: str, column_2: str, column_3: str) -> None:
        """
        Takes labels of two selected columns.
        Selects rows with only a certain value and from certain columns.
        Returns graphs of the distribution of values with a specific x-axis name.
        """
        sns.set_style("ticks")
        fig, ax = plt.subplots(2, 1, figsize=(16, 7))

        sns.lineplot(
            data=self.first_df,
            x=column_1,
            y=column_2,
            color="#2e8b57",
            ax=ax[0],
        )

        sns.lineplot(
            data=self.second_df, x=column_3, y=column_2, color="#5651c5", ax=ax[1]
        )

        ax[0].set(xlabel=" ", ylabel=column_2)
        ax[1].set(xlabel=column_3, ylabel=column_2)
        fig.legend(
            ["Accepted loans", "Rejected loans"],
            loc="upper left",
            bbox_to_anchor=(0.1, 0.85),
        )
        sns.despine()
        plt.show()

    def strip_plot(
        self, column_1: str, column_2: str, xlable: str, ylabel: str
    ) -> None:
        sns.set_style("ticks")
        plt.figure(figsize=(16, 7))

        ax = sns.stripplot(
            data=self.first_df,
            x=column_1,
            y=column_2,
            jitter=0.25,
            size=8,
            linewidth=0.2,
            edgecolor="black",
            palette="PRGn",
        )

        ax.set(xlabel=xlable, ylabel=ylabel)
        sns.despine()
        plt.show()

    def box_plot(self, column_1: str, column_2: str) -> None:
        sns.set_style("ticks")
        plt.figure(figsize=(16, 7))
        sns.boxplot(
            data=self.first_df,
            x=column_1,
            y=column_2,
            palette="PRGn",
            showmeans=True,
            meanprops={
                "marker": "o",
                "markerfacecolor": "white",
                "markeredgecolor": "black",
                "markersize": "10",
            },
        )

        sns.despine()
        plt.show()

    def bar_with_hue(
        self, column_1: str, column_2: str, column_3: str, title: str
    ) -> None:
        plt.figure(figsize=(16, 7))
        sns.set_style("ticks")

        sns.barplot(
            data=self.first_df, x=column_1, y=column_2, hue=column_3, palette="PRGn"
        )
        plt.yscale("log")
        plt.title(title, fontsize=18)

        plt.ylabel("")
        plt.xlabel("")
        sns.despine()
        plt.show()

    def dist_plot_hue(self, column_1: str, column_2: str, x_label: str) -> None:

        palette = [
            "#7800FF",
            "#D5B1FE",
            "#9F60DE",
            "#000000",
            "#0A6C00",
            "#72B36C",
            "#ADFFA5",
        ]
        sns.set_style("ticks")
        sns.set_palette(palette)
        plt.rcParams["figure.figsize"] = (14, 6)

        ax = sns.displot(
            data=self.first_df,
            x=column_1,
            hue=column_2,
            kind="kde",
            height=6,
            aspect=1.4,
            palette="PRGn",
        ).set(xlim=0)

        sns.despine()
        ax.set(xlabel=x_label)
        plt.show()

    def bar_plot(self, column_1: str, column_2: str, title: str) -> None:
        plt.figure(figsize=(20, 8))
        sns.set_style("ticks")

        splot = sns.barplot(
            data=self.first_df, x=column_1, y=column_2, palette="PRGn", ci=None
        )
        for p in splot.patches:
            splot.annotate(
                format(p.get_height(), ".1f"),
                (p.get_x() + p.get_width() / 2.0, p.get_height()),
                ha="center",
                va="center",
                xytext=(0, 9),
                textcoords="offset points",
                fontsize=14,
            )

        widthbars = [0.6, 0.6]
        for bar, newwidth in zip(splot.patches, widthbars):
            x = bar.get_x()
            width = bar.get_width()
            centre = x + width / 2.0
            bar.set_x(centre - newwidth / 2.0)
            bar.set_width(newwidth)

        plt.title(title, fontsize=18)
        plt.yticks([])
        plt.ylabel("")
        plt.xlabel("")
        sns.despine(bottom=True, left=True)
        plt.show()


def conf_matrix(model, X, y, title1, title2) -> None:
    y_pred = model.predict(X)

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    plt.rc("font", size=9)
    sns.heatmap(
        confusion_matrix(y, y_pred, normalize="true"),
        annot=True,
        fmt=".2%",
        ax=ax[0],
        cmap="PRGn",
        cbar=False,
    )

    sns.set_style("ticks")

    lr_probs = model.predict_proba(X)

    lr_probs = lr_probs[:, 1]

    ap = average_precision_score(y, lr_probs)

    lr_precision, lr_recall, _ = precision_recall_curve(y, lr_probs, pos_label=1)
    lr_f1, lr_auc = f1_score(y, y_pred), auc(lr_recall, lr_precision)

    ax[1].plot(
        lr_recall, lr_precision, color="#7800FF", lw=2, label="PR curve (AP  %.2f)" % ap
    )
    ax[1].fill_between(
        lr_recall,
        lr_precision,
        y2=np.min(lr_precision),
        color="#7800FF",
        alpha=0.3,
        hatch="/",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    ax[0].tick_params(labelsize=9)
    plt.legend(loc="upper right")
    sns.despine()

    ax[0].set_title(title1)
    ax[1].set_title(title2)
    plt.show()


def neural_conf_matrix(
    test_outputs, y_val, aggregated_losses, epochs, title1, title2
) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    plt.rc("font", size=9)
    sns.heatmap(
        confusion_matrix(test_outputs, y_val, normalize="true"),
        annot=True,
        fmt=".2%",
        ax=ax[0],
        cmap="PRGn",
        cbar=False,
    )
    sns.set_style("ticks")

    ax[1].plot(range(epochs), aggregated_losses, color="#7800FF")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    ax[0].tick_params(labelsize=9)
    ax[0].set_title(title1)
    ax[1].set_title(title2)
    sns.despine()

    plt.show()


class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        for feat in self.feature_names:
            X[feat + "_quadratic"] = self.quadratic_transformation(X[feat])
            X[feat + "_cubic"] = self.cubic_transformation(X[feat])
            X[feat + "_log"] = self.log_transformation(X[feat])
            X[feat + "_root"] = self.root_transformation(X[feat])
        self.new_df = X
        return X

    def quadratic_transformation(self, x_col):
        return (x_col) ** 2

    def cubic_transformation(self, x_col):
        return (x_col) ** 3

    def log_transformation(self, x_col):
        return np.log(x_col + 0.0001)

    def root_transformation(self, x_col):
        return 2 * np.sqrt(x_col)

    def get_feature_names(self):
        return self.new_df.columns.tolist()


class TorchStandardScaler:
    def fit(self, x):
        self.mean = x.mean(0, keepdim=True)
        self.std = x.std(0, unbiased=False, keepdim=True)

    def transform(self, x):
        x -= self.mean
        x /= self.std + 1e-7
        return x


class Model(nn.Module):
    def __init__(self, embedding_size, num_numerical_cols, output_size, layers, p=0.4):
        super().__init__()
        self.all_embeddings = nn.ModuleList(
            [nn.Embedding(ni, nf) for ni, nf in embedding_size]
        )
        self.embedding_dropout = nn.Dropout(p)
        self.batch_norm_num = nn.BatchNorm1d(num_numerical_cols)

        all_layers = []
        num_categorical_cols = sum((nf for ni, nf in embedding_size))
        input_size = num_categorical_cols + num_numerical_cols

        for i in layers:
            all_layers.append(nn.Linear(input_size, i))
            all_layers.append(nn.ReLU(inplace=True))
            all_layers.append(nn.BatchNorm1d(i))
            all_layers.append(nn.Dropout(p))
            input_size = i

        all_layers.append(nn.Linear(layers[-1], output_size))

        self.layers = nn.Sequential(*all_layers)

    def forward(self, x_categorical, x_numerical):
        embeddings = []
        for i, e in enumerate(self.all_embeddings):
            embeddings.append(e(x_categorical[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.embedding_dropout(x)

        x_numerical = self.batch_norm_num(x_numerical)
        x = torch.cat([x, x_numerical], 1)
        x = self.layers(x)
        return x
