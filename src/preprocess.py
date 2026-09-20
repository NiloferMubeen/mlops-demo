import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

RAW_DATA_PATH = "data/raw/churn.csv"
PROCESSED_DIR = Path("data/processed")


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(RAW_DATA_PATH)

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Remove customer ID
# --------------------------------------------------

df = df.drop(columns=["customer_id"])


# --------------------------------------------------
# 4. Separate features and target
# --------------------------------------------------

X = df.drop(columns=["churn"])
y = df["churn"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("churn")


# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# --------------------------------------------------
# 6. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Convert back to DataFrames
X_train_scaled = pd.DataFrame(
    X_train_scaled,
    columns=X_train.columns
)

X_test_scaled = pd.DataFrame(
    X_test_scaled,
    columns=X_test.columns
)


# --------------------------------------------------
# 7. Create processed directory
# --------------------------------------------------

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 8. Save processed datasets
# --------------------------------------------------

X_train_scaled.to_csv(
    PROCESSED_DIR / "X_train.csv",
    index=False
)

X_test_scaled.to_csv(
    PROCESSED_DIR / "X_test.csv",
    index=False
)

y_train.to_csv(
    PROCESSED_DIR / "y_train.csv",
    index=False
)

y_test.to_csv(
    PROCESSED_DIR / "y_test.csv",
    index=False
)


print("\nPreprocessing completed successfully!")

print("\nSaved files:")
print(PROCESSED_DIR / "X_train.csv")
print(PROCESSED_DIR / "X_test.csv")
print(PROCESSED_DIR / "y_train.csv")
print(PROCESSED_DIR / "y_test.csv")