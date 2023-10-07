import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from os.path import join
from linearbuddy.config import PATH_TO_REPO


def raw_to_abbrev(feat_name: str):
    return feat_name.split("_")[0]


def abbrevs_to_idxs_raw(feats_abbrev, feats_raw: pd.Series):
    return feats_raw.apply(lambda x: raw_to_abbrev(x) in feats_abbrev).values


def get_iai_data(outcome="iai"):
    df_full = pd.read_pickle(
        join(PATH_TO_REPO, f"notebooks/data/pecarn/{outcome}.pkl")).infer_objects()
    y = df_full["outcome"].values
    df = df_full.drop(columns=["outcome"])
    X = df.values
    feats_raw = pd.Series(df.columns)

    # remove redundant features
    idxs = feats_raw.str.endswith("_no") | feats_raw.str.endswith("_unknown")

    # remove compound features
    idxs |= feats_raw.str.contains("_or_")

    # remove ambiguous features
    idxs |= feats_raw.str.lower().str.startswith("other")

    # remove specific features
    idxs |= feats_raw.isin(["Age<2_yes"])
    for k in ["LtCostalTender", "RtCostalTender", "Race", "Sex"]:
        idxs |= feats_raw.str.startswith(k)
    for k in ['AbdTenderDegree_Moderate', 'AbdTenderDegree_Mild',
              'GCSScore', 'MOI_Fall down stairs', 'MOI_Fall from an elevation',
              'MOI_Motorcycle/ATV/Scooter collision', 'MOI_Object struck abdomen',
              'MOI_Pedestrian/bicyclist struck by moving vehicle', 'MOI_Bike collision/fall']:
        idxs |= feats_raw == k

    # these features are fine, just removing to make it smaller
    for k in ['AbdDistention_yes', 'InitHeartRate', 'InitSysBPRange']:
        idxs |= feats_raw == k

    # apply
    X = X[:, ~idxs]
    feats_raw = feats_raw[~idxs]
    feats_abbrev = feats_raw.apply(raw_to_abbrev)

    return X, y, feats_raw, feats_abbrev


ABBREV_TO_CLEAN_IAI = {
    "AbdDistention": "Abdominal distention",
    "AbdTenderDegree": "Abdominal tenderness",
    "AbdTrauma": "Abdominal wall trauma",
    "AbdomenPain": "Abdominal pain",
    "Age": "Age",
    "CostalTender": "Costal margin tenderness",
    "DecrBreathSound": "Decreased breath sounds",
    "DistractingPain": "Distracting pain",
    "GCSScore": "Full GCS score",
    "Hypotension": "Hypotension",
    "InitHeartRate": "Heart rate",
    "InitSysBPRange": "Systolic blood pressure",
    "LtCostalTender": "Left costal tenderness",
    "MOI": "Involvement in motor vehicle collision",
    # "Race": "Race",
    "RtCostalTender": "Right costal tenderness",
    "SeatBeltSign": "Seatbelt sign",
    # "Sex": "Sex",
    "ThoracicTender": "Thoracic tenderness",
    "ThoracicTrauma": "Thoracic trauma",
    "VomitWretch": "Vomiting",
}
PECARN_FEATS_ORDERED_IAI = [
    "AbdTrauma",
    "SeatBeltSign",
    "GCSScore",
    "AbdTenderDegree",
    "ThoracicTrauma",
    "AbdomenPain",
    "DecrBreathSound",
    "VomitWretch",
]
