from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

app = Flask(__name__, template_folder = 'templates', static_folder = 'static', static_url_path = '/')

model = joblib.load("stress_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        features = [
            float(request.form.get("sleep_quality")),
            float(request.form.get("academic_performance")),
            float(request.form.get("study_load")),
            float(request.form.get("teacher_student_relationship")),
            float(request.form.get("future_career_concerns")),
            float(request.form.get("social_support")),
            float(request.form.get("peer_pressure")),
            float(request.form.get("bullying"))
        ]
        input_df = pd.DataFrame([features], columns=[
            "sleep_quality", "academic_performance", "study_load", "teacher_student_relationship",
            "future_career_concerns", "social_support", "peer_pressure", "bullying"
        ])
        full_df = pd.DataFrame(np.zeros((1, len(model.feature_names_in_))), columns=model.feature_names_in_)
        for col in input_df.columns:
            full_df[col] = input_df[col]

        X = preprocessor.transform(full_df)
        pred = model.predict(X)[0]
        labels = {0: "Rendah", 1: "Sedang", 2: "Tinggi"}
        prediction = labels[pred]
        return redirect(url_for('index', prediction=prediction) + "#third")
    else:
        prediction = request.args.get('prediction', default=None)
        return render_template("index.html", prediction=prediction)

@app.route('/data', methods=['GET'])
def data():
    data = []
    with open('StressLevelDataset.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        headers = next(reader)
        for row in reader:
            data.append(row)
    return render_template ('data.html', headers=headers, data=data)

# Fungsi untuk membuat visualisasi
def create_visualizations():
    df = pd.read_csv("StressLevelDataset.csv")
    X = df.drop("stress_level", axis=1)

    # Tentukan jumlah fitur
    num_features = len(X.columns)

    # Tentukan jumlah kolom subplot, misalnya 4 kolom
    num_cols = 4
    num_rows = (num_features // num_cols) + (num_features % num_cols > 0)

    # 1. Distribusi Fitur
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    axes = axes.flatten()  # Meratakan array agar lebih mudah diakses

    for i, feature in enumerate(X.columns):
        sns.histplot(df[feature], kde=True, ax=axes[i])
        axes[i].set_title(f"Distribusi {feature}")

    # Hapus subplot yang tidak digunakan (jika ada)
    for i in range(num_features, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.savefig('static/img/distribution_features.png')
    plt.close()

    # 2. Heatmap Korelasi
    corr = df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Heatmap Korelasi Fitur")
    plt.savefig('static/img/heatmap_correlation.png')
    plt.close()

    # 3. Feature Importances
    feature_importances = model.feature_importances_
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=X.columns)
    plt.title("Pentingnya Fitur dalam Model RandomForest")
    plt.xlabel("Pentingnya Fitur")
    plt.ylabel("Fitur")
    plt.savefig('static/img/feature_importance.png')
    plt.close()

    # 4. Confusion Matrix
    y_test = pd.read_csv("StressLevelDataset.csv")["stress_level"]  # Dapatkan label untuk testing
    X_test = preprocessor.transform(X)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False, xticklabels=["Low", "High"], yticklabels=["Low", "High"])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.title("Confusion Matrix")
    plt.savefig('static/img/confusion_matrix.png')
    plt.close()

@app.route('/visualization', methods=['GET'])
def visual():
    return render_template('visualization.html')


if __name__ == "__main__":
    app.run(debug=True)
