import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

print("Loading emissions.csv...")
df = pd.read_csv('emissions.csv')

# Drop any rows with missing values in target or features
df = df.dropna(subset=['CountryCode', 'Year', 'EmissionLevel', 'Emissions_ktCO2e'])

X = df[['CountryCode', 'Year', 'EmissionLevel']]
y = df['Emissions_ktCO2e']

# Define preprocessor
print("Building preprocessor and model pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Year']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['CountryCode', 'EmissionLevel'])
    ])

# Define pipeline
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# Fit model
print("Training RandomForestRegressor model on full dataset...")
model_pipeline.fit(X, y)

# Save model
print("Saving model pipeline to 'model_pipeline.joblib'...")
joblib.dump(model_pipeline, 'model_pipeline.joblib')
print("Model trained and saved successfully!")
