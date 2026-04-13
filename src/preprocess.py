import pandas as pd

def load_data(path):
    df = pd.read_csv(path)

    # Clean column names
    df.columns = df.columns.str.strip()

    # Convert Total Charges to numeric
    if "Total Charges" in df.columns:
        df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

    # Use Churn Value as target if available
    # It is already numeric: 1 = churned, 0 = not churned
    if "Churn Value" not in df.columns:
        raise ValueError("Target column 'Churn Value' not found in dataset.")

    # Drop columns that should not be used as features
    columns_to_drop = [
        "CustomerID",
        "Churn Label",
        "Churn Score",
        "CLTV",
        "Churn Reason",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Zip Code"
    ]

    existing_cols_to_drop = [col for col in columns_to_drop if col in df.columns]
    df = df.drop(columns=existing_cols_to_drop)

    return df