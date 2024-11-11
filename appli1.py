import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Charger les données
data_path = r"C:\Users\AZIMUT Info\OneDrive\Desktop\spiro\df_spiro.csv"
df = pd.read_csv(data_path)

# Sélection des prédicteurs dans l'ordre spécifié et de la variable cible
X = df[['vems', 'vemscvf', 'cvf', 'aex']]
y = df['type_trouble']  # 0 = normal, 1 = obstructif, 2 = restrictif

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement du modèle
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Évaluer le modèle
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Interface Streamlit
st.title("Prédiction du Trouble Ventilatoire avec Random Forest")

st.write(f"Exactitude du modèle sur les données de test : **{accuracy:.2f}**")

st.write("""
    Cette application utilise un modèle Random Forest pour prédire le type de trouble ventilatoire en fonction de plusieurs prédicteurs :
    - **VEMS** : Volume Expiratoire Maximal par Seconde
    - **VEMS/CVF** : Ratio entre VEMS et CVF
    - **CVF** : Capacité Vitale Forcée
    - **AEX** : Autre mesure explicative
""")

# Entrée utilisateur pour les prédicteurs
vems = st.number_input("VEMS (Volume Expiratoire Maximal par Seconde)", min_value=0.0)
vemscvf = st.number_input("VEMS/CVF", min_value=0.0)
cvf = st.number_input("CVF (Capacité Vitale Forcée)", min_value=0.0)
aex = st.number_input("AEX", min_value=0.0)

# Dictionnaire pour mapper les prédictions aux libellés
labels = {0: "normal", 1: "obstructif", 2: "restrictif"}

# Bouton de prédiction
if st.button("Prédire le type de trouble"):
    # Préparation des données d'entrée
    input_data = np.array([[vems, vemscvf, cvf, aex]])

    # Prédiction avec le modèle
    prediction = model.predict(input_data)[0]
    label = labels.get(prediction, "Inconnu")

    # Afficher le résultat
    st.subheader("Résultat de la prédiction")
    st.write(f"Le type de trouble ventilatoire prédit est : **{label}**")
