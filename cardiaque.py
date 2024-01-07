#Import
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix,roc_curve,auc
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split,cross_validate


###Fonctions
def transfo_df(df):
        """Transforme les données pour les adapter à notre modèle"""
        df['cp'] = df['cp'].replace({"Angine de poitrine typique":1,"Angine de poitrine atypique":2,"Douleur non-angineuse":3,"Asymptomatique":0})
        df['fbs'] = df['fbs'].apply(lambda x: 1 if x > 120 else 0)
        df['restecg'] = df['restecg'].replace({"Normal":1,"Onde ST anormale":2,"Hypertrophie ventriculaire":0})
        df['exng'] = df['exng'].replace({'Non': 0, 'Oui': 1})
        df['slp'] = df['slp'].replace({"Pente descendante":0, "Plat":1, "Pente ascendante":2})
        df['thall'] = df['thall'].replace({"Anomalie majeure":2,"Absence":0,"Anomalie mineure":3,"Anomalie intermédiaire":1})
        return df

def one_hot_encoder(dataframe, categorical_col, drop_first=True):
    """Encode nos variables qualitatives"""
    dataframe = pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first)
    return dataframe

def encode_norme(df):
        """Encode et normalise les données"""
        df_encode = one_hot_encoder(df, qual_val_hors_binaire, drop_first=True)
        scaler = StandardScaler()
        df_encode[quant_val] = scaler.fit_transform(df_encode[quant_val])
        return(df_encode)


## Onglets
def onglet_info():
        st.title('Prédiction "Contracter une crise cardiaque"' + " qui a pour but de réaliser la tarification d'un produit d'assurance")
        st.subheader("Zeynep KAYA et Sarah BOUSTEILA")
        st.markdown("Le but de cette Webapp est d'entrer des informations sur un assuré afin de prédire s'il a de fortes chances de contracter une crise cardiaque.\n"
                "Plus précisément, l'algorithme permet d'evaluer la probabilité du rétrécissement du diamètre de l'artère coronaire qui irrigue le coeur (si elle est >= 50% on considère que l'assuré à de fortes chances d'avoir une maladie cardiovasculaire et donc susceptible de contracter une crise cardiaque).\n"
                "Les données utilisées sont celles de *heart.csv* consultables sur https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/data.")
        st.markdown("Concernant le choix du modèle nous avons choisi d'utiliser une régression logistique avec les quelques résultats de performances suivants : ")
        cv_results = cross_validate(model, X, y, cv=10, scoring=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'])
        y_pred = cross_val_predict(model, X, y, cv=10)
        fpr, tpr, thresholds = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)

        st.title('Courbe ROC avec AUC')
        st.write(f'AUC : {roc_auc:.2f}')
        fig_roc = px.area(x=fpr, y=tpr, title='Courbe ROC',
                          labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'})
        fig_roc.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
        st.plotly_chart(fig_roc)

        data = {
                'accuracy': cv_results['test_accuracy'],
                'f1': cv_results['test_f1'],
                'roc_auc': cv_results['test_roc_auc'],
                'precision': cv_results['test_precision'],
                'recall': cv_results['test_recall']}
        df = pd.DataFrame(data)
        st.title('Métriques')
        fig_metrics = px.box(df,y=['accuracy', 'f1', 'roc_auc', 'precision', 'recall'], boxmode='group', points='all', labels={'y': 'Score'},
                        title='Performances obtenues après validation croisée')
        st.plotly_chart(fig_metrics)
        conf_matrix = confusion_matrix(y, y_pred)
        st.title('Matrice de Confusion')
        st.write(conf_matrix)

def onglet_pred():
        st.title("Entrez les informations suivantes sur un assuré pour savoir s'il est fortement susceptible de contracter une crise cardiaque")
        st.write("Formulaire :")
        age = st.number_input("Age", step=1,key="age")
        cp = st.radio("Type de douleur thoracique",options=["Angine de poitrine typique","Angine de poitrine atypique","Douleur non-angineuse","Asymptomatique"],key="pain")
        trtbps = st.number_input("Pression sanguine (en mmHg)", step=1,key="trestbps")
        chol = st.number_input("Cholestérol (mg/dl)",step=1,key="chol")
        fbs = st.number_input("Taux de sucre dans le sang (mg/dl)",key="fbs")
        restecg = st.radio("Résultats de l'éléctrocardiogramme",options=["Normal","Onde ST anormale","Hypertrophie ventriculaire"],key="restecg")
        thalachh = st.number_input("Fréquence cardiaque maximale (bpm) après exercice",step=1,key="thalach")
        exang = st.radio("Angine de poitrine provoquée par l'exercice ?",options=["Oui","Non"],key="exang")
        oldpeak = st.number_input("Dépression du segment ST induite par l'exercice par rapport au repos",key="oldpeak")
        slp = st.radio("Pente du segment ST du pic d'exercice",options=["Pente descendante","Plat","Pente ascendante"],key="slp")
        caa = st.radio("Nombre de vaisseaux principaux colorés par fluorosopie",options =[0,1,2,3,4])
        thall = st.radio("Thalassémie",options=["Absence","Anomalie mineure","Anomalie intermédiaire","Anomalie majeure"])

        #Création du dataframe qui contient les données renseignées
        data = [{
            'age': age,'cp': cp,'trtbps':trtbps,'chol':chol,'fbs':fbs,'restecg':restecg,
            'thalachh':thalachh,'exng':exang,'oldpeak':oldpeak,'slp':slp,'caa':caa,'thall':thall
        }]
        if st.button("Lancer la prédiction"):
                df_formulaire = pd.DataFrame(data)
                df_test = transfo_df(df_formulaire)
                df_all_data = pd.concat([heart.drop("output", axis=1),df_test])
                df_all_data_encode = encode_norme(df_all_data)
                df_test_encode = pd.DataFrame(df_all_data_encode.tail(1))
                model.fit(X,y)
                prediction = model.predict(df_test_encode)
                col1,col2 = st.columns(2)
                if prediction[0] == 0:
                        with col1:
                                st.write("Selon le modèle, l'assuré ne fera pas de crise cardiaque.")
                        with col2:
                                st.image("images/coeur-en-bonne-sante.png")
                else:
                        with col1:
                                st.write("Attention ! Selon le modèle l'assuré fera une crise cardiaque !")
                        with col2:
                                st.image("images/coeur-mauvaise-santé.jpg")


###Lecture de la base de données
heart = pd.read_csv("data/heart.csv").drop_duplicates().drop("sex",axis=1)
###Paramètres
quant_val = ["age","trtbps","chol","thalachh","oldpeak","caa"]
qual_val = ["cp","fbs","restecg","exang","slp","thall","output"]
qual_val_hors_binaire = ["cp","restecg","slp","thall"]
heart_encode = encode_norme(heart)
y = heart_encode["output"]
X = heart_encode.drop("output", axis=1)
model = LogisticRegression(random_state=0, max_iter=3000)
st.sidebar.image("images/coeur1.jpg")

onglets = st.sidebar.radio("Onglets :",options=["Informations","Prédiction"])
if onglets == "Informations":
        onglet_info()
if onglets == "Prédiction":
        onglet_pred()
