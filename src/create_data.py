import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000

data = pd.DataFrame({
    "customer_id": range(1, n + 1),
    "age": np.random.randint(18, 70, n),
    "monthly_income": np.random.randint(20000, 100000, n),
    "tenure": np.random.randint(1, 61, n),
    "support_calls": np.random.randint(0, 11, n),
    "contract_months": np.random.choice(
        [6, 12, 24, 36],
        size=n
    )
})

# Create a simple probability of churn
churn_score = (
    0.08 * data["support_calls"]
    - 0.015 * data["tenure"]
    - 0.02 * data["contract_months"]
    + 0.000005 * data["monthly_income"]
)

probability = 1 / (1 + np.exp(-churn_score))

data["churn"] = np.random.binomial(1, probability)

data.to_csv("data/raw/churn.csv", index=False)

print("Dataset created successfully!")
print(f"Shape: {data.shape}")
print("\nFirst 5 rows:")
print(data.head())

print("\nChurn distribution:")
print(data["churn"].value_counts())