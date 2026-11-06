"""Train the final model (gradient boosting) and save it to model.joblib.

Usage:  python train.py
Steps: download data -> clean (same steps as in the notebook) -> time-based split
       -> choose threshold on a validation period -> train final model -> save.
"""
import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline

URL = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/main/data/2020/2020-02-11/hotels.csv"
MODEL_FILE = "model.joblib"

NUMERICAL = ["lead_time", "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
             "stays_in_week_nights", "adults", "children", "babies", "is_repeated_guest", "previous_cancellations",
             "previous_bookings_not_canceled", "days_in_waiting_list", "adr", "total_of_special_requests", "has_agent"]
CATEGORICAL = ["hotel", "arrival_date_month", "meal", "country", "market_segment",
               "reserved_room_type", "deposit_type", "customer_type"]
FEATURES = NUMERICAL + CATEGORICAL

# best parameters from the random search in the notebook
BEST_PARAMS = {"learning_rate": 0.05, "max_depth": 3, "max_iter": 400, "min_samples_leaf": 20}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]


def load_and_clean():
    df = pd.read_csv(URL)
    df = df.drop(columns="company")
    df["has_agent"] = df["agent"].notna().astype(int)
    df = df.drop(columns="agent")
    df["country"] = df["country"].fillna("Unknown")
    df["children"] = df["children"].fillna(0)
    df = df.drop(columns=["reservation_status", "reservation_status_date"])
    df = df.drop_duplicates()
    df = df[(df["adults"] + df["children"] + df["babies"]) > 0]
    df = df[(df["adr"] >= 0) & (df["adr"] < 1000)]
    df = df.drop(columns=["distribution_channel", "required_car_parking_spaces",
                          "assigned_room_type", "booking_changes"])
    month_num = df["arrival_date_month"].map({m: i + 1 for i, m in enumerate(MONTHS)})
    df["arrival_date"] = pd.to_datetime(dict(year=df["arrival_date_year"], month=month_num,
                                             day=df["arrival_date_day_of_month"]))
    return df


def make_model():
    return Pipeline([("dv", DictVectorizer(sparse=False)),
                     ("model", HistGradientBoostingClassifier(random_state=1, **BEST_PARAMS))])


def to_records(d):
    return d[FEATURES].to_dict(orient="records")


def main():
    df = load_and_clean()
    train = df[df["arrival_date"] < "2017-06-01"].sort_values("arrival_date").reset_index(drop=True)
    test = df[df["arrival_date"] >= "2017-06-01"].reset_index(drop=True)

    counts = train["country"].value_counts()
    frequent_countries = sorted(counts[counts >= 500].index)
    for d in (train, test):
        d["country"] = d["country"].where(d["country"].isin(frequent_countries), "other")

    # threshold: model trained before March 2017, checked on March-May 2017
    early = train[train["arrival_date"] < "2017-03-01"]
    late = train[train["arrival_date"] >= "2017-03-01"]
    model_v = make_model().fit(to_records(early), early["is_canceled"].values)
    proba_late = model_v.predict_proba(to_records(late))[:, 1]
    _, recall, thresholds = precision_recall_curve(late["is_canceled"].values, proba_late)
    threshold = float(thresholds[recall[:-1] >= 0.70].max())

    # final model on the whole training period
    model = make_model().fit(to_records(train), train["is_canceled"].values)
    auc = roc_auc_score(test["is_canceled"].values, model.predict_proba(to_records(test))[:, 1])
    print(f"threshold: {threshold:.3f} | test ROC AUC: {auc:.3f}")

    joblib.dump({"model": model, "threshold": threshold, "frequent_countries": frequent_countries,
                 "features": FEATURES}, MODEL_FILE)
    print("saved", MODEL_FILE)


if __name__ == "__main__":
    main()
