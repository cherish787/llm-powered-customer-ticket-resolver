import pandas as pd
import re

def clean_text(text: str) -> str:
    """Basic text cleaning."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s?]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the support tickets dataset."""
    try:
        df = pd.read_csv(filepath)
        df["cleaned_query"] = df["query"].apply(clean_text)
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Test ingestion
    data = load_data("data/tickets.csv")
    print(data.head())
