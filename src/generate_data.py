import numpy as np
import pandas as pd
import os

def generate_data(n_samples=2000):

    np.random.seed(42)

    data = pd.DataFrame({
        "session_duration": np.random.randint(10, 600, n_samples),
        "num_clicks": np.random.randint(1, 50, n_samples),
        "error_count": np.random.randint(0, 7, n_samples),
        "rapid_clicks": np.random.randint(0, 11, n_samples),
        "page_revisits": np.random.randint(0, 9, n_samples),
        "scroll_depth": np.random.randint(5, 101, n_samples),
        "idle_time": np.random.randint(0, 301, n_samples)
    })

    # Calculate frustration score
    score = (
        3 * data["error_count"] +
        2 * data["rapid_clicks"] +
        1.5 * data["page_revisits"] +
        0.01 * data["idle_time"] +
        0.01 * data["scroll_depth"] +
        0.005 * data["session_duration"]
    )

    # Conditional penalties
    score += np.where(data["scroll_depth"] < 20, 2, 0)
    score += np.where(data["idle_time"] > 120, 2, 0)
    score += np.where((data["session_duration"] < 30) & (data["error_count"] > 0), 2, 0)

    # Define threshold
    threshold = score.quantile(0.7)

    data["frustrated"] = (score > threshold).astype(int)

    return data


if __name__ == "__main__":

    os.makedirs("data/raw", exist_ok=True)

    df = generate_data()
    df.to_csv("data/raw/user_behavior.csv", index=False)

    print("Dataset generated successfully.")
    print(df["frustrated"].value_counts())
    print(df.head())