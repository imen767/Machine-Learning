""" import os
import sys
from src.exception import castomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','data.csv')
class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('C:\\Users\\rabdelmoula\\Documents\\Machine-Learning\\projet_ml_retail\\data\\raw\\retail_customers_COMPLETE_CATEGORICAL.csv')
            logging.info("Dataset read as pandas DataFrame")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Ingestion of data is completed")
            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)
        except Exception as e:
            raise castomException(e,sys) """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(filepath):
    """Charger les données depuis un fichier CSV."""
    df = pd.read_csv(filepath)
    print(f"✅ Données chargées : {df.shape}")
    return df


def plot_correlation_heatmap(X):
    """Afficher la heatmap de corrélation."""
    plt.figure(figsize=(20, 16))
    corr_matrix = X.select_dtypes(
        include=['float64', 'int64', 'int32']
    ).corr()
    sns.heatmap(corr_matrix, cmap='coolwarm',
                center=0, linewidths=0.5)
    plt.title('Heatmap de Corrélation')
    plt.tight_layout()
    plt.show()
    return corr_matrix


def get_high_correlation_pairs(corr_matrix, threshold=0.8):
    """Trouver les paires de features très corrélées."""
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                pairs.append({
                    'Feature 1': corr_matrix.columns[i],
                    'Feature 2': corr_matrix.columns[j],
                    'Corrélation': round(corr_matrix.iloc[i, j], 2)
                })
    df_pairs = pd.DataFrame(pairs).sort_values(
        'Corrélation', ascending=False
    )
    print(f"Paires corrélées (>{threshold}) : {len(pairs)}")
    return df_pairs


def parse_ip(ip):
    """Extraire des features depuis une adresse IP."""
    try:
        parts = ip.strip().split('.')
        first = int(parts[0])
        is_private = (
            first == 10 or
            (first == 192 and int(parts[1]) == 168) or
            (first == 172 and 16 <= int(parts[1]) <= 31)
        )
        return pd.Series([int(parts[0]), int(is_private)])
    except:
        return pd.Series([0, -1])


def print_cluster_profiles(df, target_cols):
    """Afficher le profil moyen de chaque cluster."""
    profils = df.groupby('Cluster')[target_cols].mean().round(1)
    print("📊 Profil moyen de chaque cluster :")
    print(profils)
    return profils