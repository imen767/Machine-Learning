from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Charger le modèle Flask simplifié
model_churn  = joblib.load('../models/random_forest_churn.joblib')
feature_cols = joblib.load('../models/feature_columns.joblib')
scaler       = joblib.load('../models/scaler_churn.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les valeurs du formulaire
        data = {
            'Frequency'              : float(request.form['frequency']),
            'MonetaryAvg'            : float(request.form['monetary_avg']),
            'AvgDaysBetweenPurchases': float(request.form['avg_days']),
            'SatisfactionScore'      : float(request.form['satisfaction']),
            'RFMSegment'             : float(request.form['rfm_segment']),
            'LoyaltyLevel'           : float(request.form['loyalty']),
            'PreferredMonth'         : float(request.form['pref_month'])
        }

        # Créer le dataframe dans le bon ordre
        input_data = pd.DataFrame([data])[feature_cols]

        # Normaliser
        input_scaled = scaler.transform(input_data)

        # Prédiction
        prediction  = model_churn.predict(input_scaled)[0]
        probability = model_churn.predict_proba(input_scaled)[0][1] * 100

        # Interprétation
        if probability >= 70:
            niveau  = "Risque élevé !"
            conseil = "Contactez ce client immédiatement avec une offre spéciale !"
        elif probability >= 40:
            niveau  = "Risque modéré"
            conseil = "Envoyez une newsletter et une promotion personnalisée"
        else:
            niveau  = "Risque faible"
            conseil = "Continuez à fidéliser ce client avec votre programme VIP"

        result = {
            'prediction' : 'CHURN ⚠️' if prediction == 1 else 'FIDÈLE ✅',
            'probability': f"{probability:.1f}%",
            'color'      : 'red' if prediction == 1 else 'green',
            'niveau'     : niveau,
            'conseil'    : conseil
        }

    except Exception as e:
        result = {
            'prediction' : 'Erreur',
            'probability': str(e),
            'color'      : 'orange',
            'niveau'     : '',
            'conseil'    : ''
        }

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)