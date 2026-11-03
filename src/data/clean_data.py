import pandas as pd

def clean_complaints(df):

    df = df.copy()

    df["complaint_narrative"] = (
        df["complaint_narrative"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    categorical_columns = [
        "product",
        "sub_product",
        "issue",
        "sub_issue",
        "company",
        "channel",
        "state",
        "sentiment_label",
        "resolution_status"
    ]

    for column in categorical_columns:
        df[column] = df[column].astype("string").str.strip()

    df["date_received"] = pd.to_datetime(
        df["date_received"],
        dayfirst=False,
        errors="coerce"
    )

    df = df.drop_duplicates(subset=["complaint_id"])

    df = df[
        df["complaint_narrative"].str.len() > 0
    ]

    return df