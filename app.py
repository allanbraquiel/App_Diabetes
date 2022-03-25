import pandas as pd
import streamlit as st
from streamlit import caching
from datetime import datetime
from datetime import timedelta, date
import altair as alt
import base64
import io
from io import StringIO
import streamlit.components.v1 as components
import time
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score,cross_val_predict
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, auc
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier

st.set_page_config(
    page_title="Squad Scikit-Learn",
    layout="wide",
    page_icon=":shark",
    initial_sidebar_state="expanded",
)


# Menu Lateral
st.sidebar.subheader("Filtros para realizar a predição")

#título
st.title("Prevendo Diabetes")

#dataset
url = "https://raw.githubusercontent.com/allanbraquiel/Stack_Labs_2_Squad_Scikit-Learn/main/datasets/diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
df = pd.read_csv(url)


# Pagina Principal
# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")


dados = df.rename(columns = {'Diabetes_binary':'Diabetes', 
                                         'HighBP':'PressAlta',  
                                         'HighChol':'CholAlto',
                                         'CholCheck':'ColCheck', 
                                         'BMI':'IMC', 
                                         'Smoker':'Fumante', 
                                         'Stroke':'Derrame',
                                         'HeartDiseaseorAttack':'CoracaoEnf', 
                                         'PhysActivity':'AtivFisica', 
                                         'Fruits':'Frutas',
                                         'Veggies':"Vegetais", 
                                         'HvyAlcoholConsump':'ConsAlcool', 
                                         'AnyHealthcare':'PlSaude',
                                         'NoDocbcCost':'DespMedica', 
                                         'GenHlth':'SdGeral',
                                         'MentHlth':'SdMental',
                                         'PhysHlth':'SdFisica',
                                         'DiffWalk':'DifCaminhar', 
                                         'Sex':'Sexo',
                                         'Age':'Idade',
                                         'Education':'Educacao',
                                         'Income':'Renda' })

# atributos para serem exibidos por padrão
defaultcols = ["HighChol", "HighBP", "BMI", "Age", "Sex", "Smoker", "Stroke", "HvyAlcoholConsump", "Diabetes_binary"]

# Exibir o dataframe dos chamados
with st.expander("Descrição do dataset:", expanded=False):
    cols = st.multiselect("", df.columns.tolist(), default=defaultcols)
    # Dataframe
    st.dataframe(df[cols])


#nomedousuário
user_input = st.sidebar.text_input("Digite seu nome")

#escrevendo o nome do usuário
st.write("Paciente:", user_input)







 
    








# fonte: https://medium.com/data-hackers/desenvolvimento-de-um-aplicativo-web-utilizando-python-e-streamlit-b929888456a5