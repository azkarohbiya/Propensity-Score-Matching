import pandas as pd
import numpy as np
import propensio

# =============================================================================
# 1. PROPENSITY SCORE MATCHING
# =============================================================================
print("=" * 60)
print("TEST 1: PropensityScoreMatch")
print("=" * 60)

# Load built-in sample data
df = propensio.load_dataset('stroke')
obj_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=obj_cols).fillna(0)

df_model = df[['age', 'hypertension', 'heart_disease', 'bmi', 'stroke',
               'gender_Male', 'smoking_status_smokes', 'avg_glucose_level']]

features  = ['age', 'hypertension', 'heart_disease', 'bmi', 'gender_Male', 'avg_glucose_level']
treatment = 'smoking_status_smokes'
outcome   = 'stroke'

psm = propensio.PropensityScoreMatch(df_model, features, treatment, outcome)

print(f"\nATT : {psm.ATT:.4f}")
print(f"ATE : {psm.ATE:.4f}")
print(f"ATC : {psm.ATC:.4f}")
print(f"\nMatched rows : {len(psm.df_matched)}")
print(f"\nSMD Table:\n{psm.df_smd}")

psm.plot_smd()


