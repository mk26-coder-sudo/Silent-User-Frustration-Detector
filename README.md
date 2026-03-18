🚨 Silent-User-Frustration-Detector

Detect. Understand. Improve.

Overview
Tech That Understands User Behavior, One Interaction at a Time.
A Streamlit-based web application that detects silent user frustration using behavioral data like clicks, errors, idle time, and session patterns. The system uses a machine learning model to predict frustration and provides explainable insights using SHAP to help improve user experience.

🧠 Data Structures Used

1. pandas.DataFrame

📍 Used in:
Loading dataset
Training ML model
Passing user input to model

🔎 Why:
Structured format ideal for ML
Maintains feature names (important for prediction consistency)

2. list

📍 Used in:
Collecting user input values
Feature ordering for prediction

🔎 Why:

Dynamic and easy to manage
Maintains input sequence

3. dict

📍 Used in:
Mapping feature names to values (DataFrame creation)

🔎 Why:
Clear key-value structure
Helps maintain feature consistency

4. numpy array (initially)

📍 Used in:
Earlier version of input handling

🔎 Why:
Efficient numerical computation

5. matplotlib

📍 Used in:
Plotting feature importance
Visualizing SHAP outputs

🔎 Why:
Simple and effective visualization tool

6. SHAP Values (Tree Structure)

📍 Used in:
Explaining predictions

🔎 Why:
Shows contribution of each feature
Improves model transparency







