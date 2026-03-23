import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')


def load_artifacts(models_dir):
    """Charger tous les modèles et scalers."""
    artifacts = {
        'model_churn'    : joblib.load(f'{models_dir}/random_forest_churn.joblib'),
        'model_regression': joblib.load(f'{models_dir}/random_forest_regression.joblib'),
        'scaler_churn'   : joblib.load(f'{models_dir}/scaler_churn.joblib'),
        'scaler_reg'     : joblib.load(f'{models_dir}/scaler_regression.joblib'),
        'feature_cols'   : joblib.load(f'{models_dir}/feature_columns.joblib'),
        'num_cols'       : joblib.load(f'{models_dir}/num_cols.joblib'),
    }
    print("✅ Tous les modèles chargés !")
    return artifacts


def predict_churn(client_data, artifacts):
    """
    Prédire le churn d'un client.
    
    Parameters:
        client_data : dict avec les features du client
        artifacts   : dict contenant modèle + scaler + colonnes
    
    Returns:
        dict avec prediction, probabilité et conseil
    """
    feature_cols = artifacts['feature_cols']
    scaler       = artifacts['scaler_churn']
    model        = artifacts['model_churn']

    # Créer le dataframe
    input_df = pd.DataFrame([client_data])[feature_cols]

    # Normaliser
    input_scaled = scaler.transform(input_df)

    # Prédire
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1] * 100

    # Conseil
    if probability >= 70:
        niveau  = "Risque élevé !"
        conseil = "Contactez ce client immédiatement !"
    elif probability >= 40:
        niveau  = "Risque modéré"
        conseil = "Envoyez une promotion personnalisée"
    else:
        niveau  = "Risque faible"
        conseil = "Continuez le programme de fidélité"

    return {
        'prediction' : 'CHURN' if prediction == 1 else 'FIDÈLE',
        'probability': round(probability, 1),
        'niveau'     : niveau,
        'conseil'    : conseil
    }


def predict_revenue(client_data, artifacts):
    """
    Prédire le revenu futur d'un client.

    Parameters:
        client_data : dict avec les features du client
        artifacts   : dict contenant modèle + scaler + colonnes

    Returns:
        float : montant prédit en £
    """
    num_cols = ['Frequency', 'MonetaryAvg', 'MonetaryStd',
                'TotalQuantity', 'UniqueProducts', 'Age',
                'AvgDaysBetweenPurchases', 'SupportTicketsCount',
                'SatisfactionScore', 'AvgQuantityPerTransaction']

    scaler = artifacts['scaler_reg']
    model  = artifacts['model_regression']

    # Créer le dataframe
    input_df = pd.DataFrame([client_data])
    cols_available = [c for c in num_cols if c in input_df.columns]
    input_df = input_df[cols_available]

    # Normaliser
    input_scaled = scaler.transform(input_df)

    # Prédire
    revenue = model.predict(input_scaled)[0]
    return round(revenue, 2)


def predict_batch(df, artifacts):
    """
    Prédire le churn pour un ensemble de clients.

    Parameters:
        df        : DataFrame avec les clients
        artifacts : dict contenant modèle + scaler + colonnes

    Returns:
        DataFrame avec les prédictions ajoutées
    """
    feature_cols = artifacts['feature_cols']
    scaler       = artifacts['scaler_churn']
    model        = artifacts['model_churn']

    # Aligner les colonnes
    X = df[feature_cols]

    # Normaliser
    X_scaled = scaler.transform(X)

    # Prédire
    df['Churn_Predicted']    = model.predict(X_scaled)
    df['Churn_Probability']  = model.predict_proba(X_scaled)[:, 1]

    print(f"✅ Prédictions batch terminées : {len(df)} clients")
    print(f"   Clients à risque : {df['Churn_Predicted'].sum()}")
    return df


if __name__ == '__main__':
    # Test rapide
    models_dir = r'C:\Users\rabdelmoula\Documents\Machine-Learning\projet_ml_retail\models'
    artifacts  = load_artifacts(models_dir)

    # Exemple client à risque
    client_risque = {
        'Frequency'              : 1,
        'MonetaryAvg'            : 50,
        'AvgDaysBetweenPurchases': 300,
        'SatisfactionScore'      : 1,
        'RFMSegment'             : 3,
        'LoyaltyLevel'           : 0,
        'PreferredMonth'         : 12
    }

    result = predict_churn(client_risque, artifacts)
    print(f"\n=== Test Client à Risque ===")
    print(f"Prédiction  : {result['prediction']}")
    print(f"Probabilité : {result['probability']}%")
    print(f"Conseil     : {result['conseil']}")