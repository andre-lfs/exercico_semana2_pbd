import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv(r'04_dados_exercicio.csv')  #raw
features = dataset.iloc[:, :-1].values  #busca com indice i localization(iloc)
print(features)
classe = dataset.iloc[:, -1].values
#Não consegui
#if(features==0):
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(features[:, 2:4])  #aqui escolhemos as colunas
features[:, 2:4] = imputer.transform(features[:, 2:4])  #aqui tb
columnTransformer = ColumnTransformer(transformers=[('encoder',
                                                     OneHotEncoder(), [1])],
                                      remainder='passthrough')
features = np.array(columnTransformer.fit_transform(features))
labelEncoder = LabelEncoder()
classe = labelEncoder.fit_transform(classe)
print('==========features==========')
print(features)
print('==========classe===========')
print(classe)
#85% treinamento

features_treinamento, features_teste, classe_treinamento, classe_teste = train_test_split(
    features, classe, test_size=0.15, random_state=1)
print('===========features_treinamento===========')
print(features_treinamento)
print('===========features_teste===========')
print(features_teste)
print('==========classe_treinamento=========')
print(classe_teste)
print('==========classe_teste==========')
print(classe_teste)
standardScaler = StandardScaler()
features_treinamento[:, 5:] = standardScaler.fit_transform(
    features_treinamento[:, 5:])
features_teste[:, 5:] = standardScaler.transform(features_teste[:, 5:])
print('============features_treinamento=========')
print(features_treinamento)
print('==========features_teste===========')
print(features_teste)
