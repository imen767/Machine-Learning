import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


def load_raw_data(filepath):
    """Charger les données brutes."""
    df = pd.read_csv(filepath)
    print(f"✅ Données chargées : {df.shape}")
    return df


def remove_useless_columns(df):
    """Supprimer les colonnes inutiles."""
    cols_to_drop = ['Newsletter', 'CustomerID']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"✅ Colonnes supprimées : {cols_to_drop}")
    return df


def fix_aberrant_values(df):
    """Corriger les valeurs aberrantes."""
    if 'SupportTicketsCount' in df.columns:
        df['SupportTicketsCount'] = df['SupportTicketsCount'].replace(
            [-1, 999], np.nan
        )
    if 'SatisfactionScore' in df.columns:
        df['SatisfactionScore'] = df['SatisfactionScore'].replace(
            [-1, 0, 99], np.nan
        )
    print("✅ Valeurs aberrantes corrigées !")
    return df


def impute_missing_values(df):
    """Imputer les valeurs manquantes avec la médiane."""
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    print(f"✅ Valeurs manquantes imputées !")
    print(f"   Manquants restants : {df.isnull().sum().sum()}")
    return df


def parse_registration_date(df):
    """Parser la date d'inscription."""
    if 'RegistrationDate' in df.columns:
        df['RegistrationDate'] = pd.to_datetime(
            df['RegistrationDate'],
            dayfirst=True,
            errors='coerce'
        )
        df['RegYear']    = df['RegistrationDate'].dt.year
        df['RegMonth']   = df['RegistrationDate'].dt.month
        df['RegDay']     = df['RegistrationDate'].dt.day
        df['RegWeekday'] = df['RegistrationDate'].dt.weekday
        df = df.drop(columns=['RegistrationDate'])
        print("✅ RegistrationDate parsée !")
    return df


def parse_ip_features(df):
    """Extraire des features depuis LastLoginIP."""
    if 'LastLoginIP' in df.columns:
        def parse_ip(ip):
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

        df[['IP_FirstOctet', 'IP_IsPrivate']] = df['LastLoginIP'].apply(
            parse_ip
        )
        df = df.drop(columns=['LastLoginIP'])
        print("✅ LastLoginIP parsée !")
    return df


def feature_engineering(df):
    """Créer de nouvelles features."""
    if 'MonetaryTotal' in df.columns:
        df['MonetaryPerDay'] = df['MonetaryTotal'] / (
            df['Recency'] + 1
        )
        df['AvgBasketValue'] = df['MonetaryTotal'] / (
            df['Frequency'] + 1
        )
    if 'Recency' in df.columns and 'CustomerTenureDays' in df.columns:
        df['TenureRatio'] = df['Recency'] / (
            df['CustomerTenureDays'] + 1
        )
    print("✅ Feature Engineering terminé !")
    return df


def encode_categorical(df):
    """Encoder les variables catégorielles."""
    # Ordinal encoding
    ordinal_cols = [
        'RFMSegment', 'AgeCategory', 'SpendingCategory',
        'LoyaltyLevel', 'ChurnRiskCategory', 'BasketSizeCategory',
        'PreferredTimeOfDay', 'WeekendPreference', 'ProductDiversity'
    ]
    le = LabelEncoder()
    for col in ordinal_cols:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    # One-Hot encoding
    onehot_cols = [
        'CustomerType', 'FavoriteSeason',
        'Region', 'Gender', 'AccountStatus'
    ]
    onehot_cols = [c for c in onehot_cols if c in df.columns]
    df = pd.get_dummies(df, columns=onehot_cols, drop_first=True)

    # Target encoding pour Country
    if 'Country' in df.columns and 'Churn' in df.columns:
        country_means = df.groupby('Country')['Churn'].mean()
        df['Country'] = df['Country'].map(country_means)

    print("✅ Encoding terminé !")
    return df


def remove_correlated_features(df, threshold=0.8):
    """Supprimer les features trop corrélées."""
    cols_to_remove = [
        'UniqueInvoices', 'UniqueDescriptions',
        'CancelledTransactions', 'MonetaryMax',
        'MonetaryMin', 'MinQuantity',
        'WeekendPreference', 'MonetaryPerDay',
        'TotalTransactions'
    ]
    cols_to_remove = [c for c in cols_to_remove if c in df.columns]
    df = df.drop(columns=cols_to_remove)
    print(f"✅ {len(cols_to_remove)} features redondantes supprimées !")
    return df


def normalize_features(df, target_col='Churn'):
    """Normaliser les features numériques."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    num_cols = X.select_dtypes(
        include=['int64', 'float64', 'int32']
    ).columns.tolist()

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    print(f"✅ Normalisation terminée ! ({len(num_cols)} colonnes)")
    return X, y, scaler, num_cols


def split_and_save(X, y, output_dir, test_size=0.2, random_state=42):
    """Séparer et sauvegarder les données train/test."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    X_train.to_csv(f'{output_dir}/X_train.csv', index=False)
    X_test.to_csv(f'{output_dir}/X_test.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)

    print(f"✅ Train/Test sauvegardés dans {output_dir}")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    return X_train, X_test, y_train, y_test