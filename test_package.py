import pandas as pd
from causalinference_azkaram.matching import PropensityScoreMatch

# Load sample data
df = pd.read_csv('datasets/healtcare_stroke_data.csv')

# One-hot encode
obj_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=obj_cols).fillna(0)

# Select columns
df_model = df[['age', 'hypertension', 'heart_disease', 'bmi', 'stroke',
               'gender_Male', 'smoking_status_smokes', 'avg_glucose_level']]

# Define variables
features = ['age', 'hypertension', 'heart_disease', 'bmi', 'gender_Male', 'avg_glucose_level']
treatment = 'smoking_status_smokes'
outcome = 'stroke'

# Run PSM
model = PropensityScoreMatch(df_model, features, treatment, outcome)

print("\n--- Results ---")
print(f"ATT : {model.ATT:.4f}")
print(f"ATE : {model.ATE:.4f}")
print(f"ATC : {model.ATC:.4f}")
print(f"\nMatched rows : {len(model.df_matched)}")
print(f"\nSMD Table:\n{model.df_smd}")

model.plot_smd()
