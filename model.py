# 1. Import library
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("StressLevelDataset.csv")

X = df.drop("stress_level", axis=1)
y = df["stress_level"]

preprocessor = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())            
])

X_processed = preprocessor.fit_transform(X)

# Konversi kembali ke DataFrame agar tetap punya nama kolom
feature_names = X.columns
X_processed = pd.DataFrame(X_processed, columns=feature_names)

# Split dan training
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print(model.feature_names_in_)  # ✅ Sekarang ini akan berhasil

# Evaluasi dan simpan
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Akurasi setelah preprocessing: {accuracy:.2f}")

joblib.dump(model, "stress_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")
print("Model dan preprocessing pipeline berhasil disimpan.")
