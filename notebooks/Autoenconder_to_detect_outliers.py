# -*- coding: utf-8 -*-
"""
Created on Sun Jul  6 22:19:19 2025

@author: eosjo
"""

#%% Autoencoders -> 

#%% Import data
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers
import os
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import random
import os


# Fixar a seed para reprodutibilidade
seed = 1
np.random.seed(seed)
tf.random.set_seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)


#%% 
def path(novo_subcaminho):
    pasta_atual = os.getcwd()
    pasta_anterior = os.path.dirname(pasta_atual)
    caminho_final = os.path.join(pasta_anterior, novo_subcaminho)
    return caminho_final


caminho_arquivo1 = path("Databases\\sample_submission.csv")
df_sample_submission= pd.read_csv(caminho_arquivo1) 

caminho_arquivo2 = path("Databases\\test.csv")
df_test= pd.read_csv(caminho_arquivo2) 

caminho_arquivo3 = path("Databases\\train.csv")
caminho_arquivo3_parquet = path("notebooks\\data_no_Outliers.parquet")
caminho_arquivo_only_outliers_parquet = path("notebooks\\Outliers.parquet")

df_train_with_outliers_original= pd.read_csv(caminho_arquivo3) 
df_train_no_outliers_original = pd.read_parquet(caminho_arquivo3_parquet)
df_train_with_outliers= pd.read_csv(caminho_arquivo3) 
df_train_no_outliers= pd.read_parquet(caminho_arquivo3_parquet)
df_train_only_outliers= pd.read_parquet(caminho_arquivo_only_outliers_parquet)

#%% drop Na columns
df_train_with_outliers.dropna(axis=0, inplace=True)
df_train_no_outliers.dropna(axis=0, inplace=True)
df_train_with_outliers_original.dropna(axis=0, inplace=True)
df_train_no_outliers_original.dropna(axis=0, inplace=True)


#%% Divide dataset into target x features
colunas_para_remover = ["Personality", "id"]

df_train_no_outliers_target = df_train_no_outliers[["Personality"]]
df_train_no_outliers_features = df_train_no_outliers.drop(columns=colunas_para_remover)


df_train_with_outliers_target = df_train_with_outliers[["Personality"]]
df_train_with_outliers_features = df_train_with_outliers.drop(columns=colunas_para_remover)



#%% One hot encoding
colunas_one_hot_encoding = ["Stage_fear","Drained_after_socializing"]

df_train_no_outliers_features = pd.get_dummies(df_train_no_outliers_features, colunas_one_hot_encoding, dtype=int, drop_first=True)
df_train_with_outliers_features  = pd.get_dummies(df_train_with_outliers_features, colunas_one_hot_encoding, dtype=int, drop_first=True)

#%% normalização de dados numéricos: 
    
# Lista das colunas numéricas
colunas_normalizacao = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]

# Instanciando o scaler
scaler = StandardScaler()

# Aplicando o scaler apenas nas colunas desejadas.

df_train_no_outliers_features[colunas_normalizacao] = scaler.fit_transform(df_train_no_outliers_features[colunas_normalizacao]) 
df_train_with_outliers_features[colunas_normalizacao] = scaler.fit_transform(df_train_with_outliers_features[colunas_normalizacao]) 



#%% divisão train test

df_train_no_outliers_features_train, df_train_no_outliers_features_test, df_train_no_outliers_target_train , df_train_no_outliers_target_train_test = train_test_split(df_train_no_outliers_features, df_train_no_outliers_target , test_size=0.2, random_state=1)

#%% Autoencoders -> Criação de um autoenconder para detecção de anomalias

# Definir autoencoder
input_dim = df_train_no_outliers_features_train.shape[1]
input_layer = Input(shape=(input_dim,))
encoded = Dense(40, activation="relu")(input_layer)
encoded = Dense(7, activation="relu")(encoded)
decoded = Dense(40, activation="relu")(encoded)
decoded = Dense(input_dim, activation="linear")(decoded)

autoencoder = Model(inputs=input_layer, outputs=decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Treinar o autoencoder (só com dados normais!)
autoencoder.fit(df_train_no_outliers_features_train, df_train_no_outliers_features_train, epochs=50, batch_size=32, shuffle=True, verbose=0)


# Calcular erro de reconstrução no teste
df_train_no_outliers_features_predict = autoencoder.predict(df_train_no_outliers_features_test)
reconstruction_error = np.mean(np.square(df_train_no_outliers_features_test - df_train_no_outliers_features_predict), axis=1)
#%% 
import matplotlib.pyplot as plt

plt.hist(reconstruction_error, bins=100)
plt.title("Distribuição do Erro de Reconstrução")
plt.xlabel("Erro")
plt.ylabel("Frequência")
#plt.xlim(1e-5, 10e-5)  # ajusta o intervalo de visualização no eixo X
plt.show()

max_error = reconstruction_error.max()
#%% Definir limiar para detectar anomalias (ex: percentil 99 do erro dos normais)
limiar = np.percentile(reconstruction_error, 99)

# Teste com dados sem outliers 
df_train_no_outliers_features_predict = autoencoder.predict(df_train_with_outliers_features)
reconstruction_error_with_outliers = np.mean(np.square(df_train_with_outliers_features - df_train_no_outliers_features_predict), axis=1)

df_train_with_outliers_features["isOutlier"] = (reconstruction_error_with_outliers > limiar).astype(int)  # 1 = anomalia

#%% Remover outliers


df_train_with_outliers_features["id"] = df_train_with_outliers_original["id"]
df_train_with_outliers_features["Personality"] = df_train_with_outliers_original["Personality"]


indices_to_drop = df_train_with_outliers_features.index[df_train_with_outliers_features['isOutlier'] == 1]
df_train_without_new_outliers = df_train_with_outliers_features.drop(indices_to_drop)

df_train_without_new_outliers[colunas_normalizacao] = scaler.inverse_transform(df_train_without_new_outliers[colunas_normalizacao])

#%% Comparação de conjuntos 

only_outliers_df = df_train_with_outliers_features[~df_train_with_outliers_features['id'].isin(df_train_without_new_outliers['id'])]




# Extrair os conjuntos de IDs
ids_a = set(only_outliers_df['id'])
ids_b = set(df_train_only_outliers['id'])

# IDs em ambos
ids_em_ambos = list(ids_a & ids_b)  # interseção

# IDs só em A, não em B
ids_so_em_a = list(ids_a - ids_b)   # diferença A - B

# IDs só em B, não em A
ids_so_em_b = list(ids_b - ids_a)   # diferença B - A


