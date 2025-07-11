# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 14:13:24 2025

@author: eosjos
"""

# Bibliotecas padrão
import os
from itertools import combinations

# Bibliotecas de terceiros
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans

# K-Modes
from kmodes.kprototypes import KPrototypes



#%% Import das variáveis do modelo#%% Import data

def path(novo_subcaminho):
    pasta_atual = os.getcwd()
    pasta_anterior = os.path.dirname(pasta_atual)
    caminho_final = os.path.join(pasta_anterior, novo_subcaminho)
    return caminho_final


def inverter_clusters(df, cluster_indices):
    for x in cluster_indices:
        col_name = f"cluster{x}"
        if col_name in df.columns:
            # Garante que só 0 e 1 sejam invertidos
            df[col_name] = df[col_name].apply(lambda v: 1 if v == 0 else (0 if v == 1 else v))
        else:
            print(f"Coluna {col_name} não encontrada.")
    return df

def somar_clusters(df, inicio=0, fim=19):
    # Gera os nomes das colunas: cluster1, cluster2, ..., cluster20
    colunas_cluster = [f"cluster{i}" for i in range(inicio, fim + 1)]

    # Verifica quais colunas realmente existem no DataFrame
    colunas_existentes = [col for col in colunas_cluster if col in df.columns]

    # Soma os valores das colunas existentes
    df["cluster_soma"] = df[colunas_existentes].sum(axis=1)
    return df

def marcar_pesos_outlier(
    df: pd.DataFrame,
    personality_col: str = "Personality",
    n_clusters: int = 20,
    intro_label: str = "introvert",
    extro_label: str = "extrovert"
) -> pd.DataFrame:
    """
    Para cada cluster_i (i de 1 a n_clusters) cria a coluna peso_outlier_i:
      • 0 se (Personality == introvert  & cluster_i == 0) ou
            (Personality == extrovert & cluster_i == 1)
      • 1 caso contrário
    Retorna o DataFrame já alterado (opera in‑place e também devolve por conveniência).
    """
    # Garante comparação insensível a maiúsculas/minúsculas
    personality_series = df[personality_col].str.lower()

    for i in range(0, n_clusters + 1):
        cluster_col = f"cluster{i}"
        outlier_col = f"peso_outlier_{i}"

        if cluster_col not in df.columns:
            # Se a coluna do cluster não existir, avisa e pula
            print(f"⚠️  Coluna '{cluster_col}' não encontrada; pulando.")
            continue

        # Regra: 0 quando "bate" com o esperado, 1 quando "desvia"
        df[outlier_col] = np.where(
            ((personality_series == intro_label.lower()) & (df[cluster_col] == 0)) |
            ((personality_series == extro_label.lower()) & (df[cluster_col] == 1)),
            0,
            1
        )

    return df

def somar_outliers(df, inicio=1, fim=20):
    # Gera os nomes das colunas: cluster1, cluster2, ..., cluster20
    colunas_outlier = [f"peso_outlier_{i}" for i in range(inicio, fim + 1)]

    # Verifica quais colunas realmente existem no DataFrame
    colunas_existentes = [col for col in colunas_outlier if col in df.columns]

    # Soma os valores das colunas existentes
    df["outliers_soma"] = df[colunas_existentes].sum(axis=1)
    return df


def filtrar_outliers(df, limite, coluna="outliers_soma", ):
    n_total = len(df)
    df_filtrado = df[df[coluna] <= limite].copy()
    n_removidos = n_total - len(df_filtrado)
    return df_filtrado, n_removidos

def pegar_outliers(df,limite ,coluna="outliers_soma"):
    n_total = len(df)
    df_filtrado = df[df[coluna] > limite].copy()
    n_removidos = n_total - len(df_filtrado)
    return df_filtrado, n_removidos



def manter_colunas_em_comum(df_original, df_referencia):
    colunas_comuns = [col for col in df_original.columns if col in df_referencia.columns]
    return df_original[colunas_comuns].copy()




#%% Definitions: 

def Remove_noise_data (num_cols, cat_cols ,target_cols , df, clusters_para_inverter): 
    df_copy = df.copy()
    
    #Remove NA data and remove Y label
    df.dropna(axis=0, inplace=True)
    
    # Cria todas as combinações possíveis de pares (sem repetição)
    combinacoes = list(combinations(num_cols, 2))
    
    
    cluster_counter = 1
    print(cluster_counter)
    # Para cada par (x, y), aplica KMeans (Dados numéricos)
    for idx, (x, y) in enumerate(combinacoes, start=1):
        # Seleciona as colunas e normaliza
        X = df[[x, y]]
        X_scaled = StandardScaler().fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=[x, y], index=df.index)
        X_scaled_df["Personality"] = df["Personality"]
        
        # Índice da coluna categórica (sempre a segunda coluna, índice 1)
        categorical_index = [2]
        # Aplica KMeans
       
        # Inicializa e treina o modelo KPrototypes
        kproto = KPrototypes(n_clusters=2, init='Cao', random_state=1)
        clusters = kproto.fit_predict(X_scaled_df, categorical=categorical_index)
        # Salva os clusters com nome cluster1, cluster2, ...
        df[f"cluster{cluster_counter}"] = clusters

        #Visualização
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x=x, y=y, hue=f"cluster{ cluster_counter}", palette="Set2", s=100)
        plt.title(f"K-Means: {x} vs {y}")
        plt.xlabel(x)
        plt.ylabel(y)
        plt.grid(True)
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()
        
        cluster_counter += 1
        print(cluster_counter)
        
    # Inicializa o contador de clusters
    

    # Itera sobre todas as combinações de colunas numéricas e categóricas, usando kprototype
    for num_col in num_cols:
        for cat_col in cat_cols:
            # Cria o DataFrame X com a combinação atual
            X = df[[num_col, cat_col] + target_cols]

            # Índice da coluna categórica (sempre a segunda coluna, índice 1)
            categorical_index = [1,2]

            # Inicializa e treina o modelo KPrototypes
            kproto = KPrototypes(n_clusters=2, init='Cao', random_state=1)
            clusters = kproto.fit_predict(X, categorical=categorical_index)

            # Nome da nova coluna de cluster
            cluster_col_name = f"cluster{cluster_counter}"
            df[cluster_col_name] = clusters

            # Visualização
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df, x=num_col, y=cat_col, hue=cluster_col_name, palette="Set2", s=100)
            plt.title(f"KPrototypes com 2 Clusters: {num_col} vs {cat_col}")
            plt.xlabel(num_col)
            plt.ylabel(cat_col)
            plt.grid(True)
            plt.legend(title=f"cluster{cluster_counter}")
            plt.tight_layout()
            plt.show()

            # Incrementa o contador de clusters
            cluster_counter += 1
            print(cluster_counter)
            
    # Aplica a inversão de significado em alguns clusters, marcas pesos para outliers, e adicionar a soma desses pesos
    df = inverter_clusters(df, clusters_para_inverter)
    df= marcar_pesos_outlier(df)  
    df = somar_clusters(df)
    df = somar_outliers(df)
    
    df_comp = df[["id", "Personality", "outliers_soma", "cluster_soma"]]
    
    df_filtrado, n_removidos = filtrar_outliers(df,corte)
    print(f"Foram removidas {n_removidos} observações onde  > {corte}.")
    df_outliers, n_removidos = pegar_outliers(df,corte)
    print(f"Foram removidas {n_removidos} observações onde  < {corte}.")


    df_return = manter_colunas_em_comum(df_filtrado, df_copy)
    df_outliers = manter_colunas_em_comum(df_outliers,df_copy)
    
    return df_return,df_outliers,df_comp

#%% Definitions: 
    
num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
            "Friends_circle_size", "Post_frequency"]
cat_cols = ["Stage_fear", "Drained_after_socializing"]
target_cols = ["Personality"]
corte = 15



#%% Import Data
caminho_arquivo1 = path("Databases\\sample_submission.csv")
df_sample_submission= pd.read_csv(caminho_arquivo1) 

caminho_arquivo2 = path("Databases\\test.csv")
df_test= pd.read_csv(caminho_arquivo2) 

caminho_arquivo3 = path("Databases\\train.csv")
df_train= pd.read_csv(caminho_arquivo3) 



# Lista dos clusters que deseja inverter (Depende do random state, e de uma análise visual)
clusters_para_inverter = [2, 4, 5, 8, 9, 10, 11, 12, 13, 14]     
df_return, df_outliers, df_comp =  Remove_noise_data(num_cols, cat_cols ,target_cols , df_train, clusters_para_inverter)

#%% export data to parquet file
caminho_export1 = path("Databases\\data_no_Outliers.parquet")
caminho_export2 = path("Databases\\Outliers.parquet")
 
df_return.to_parquet(caminho_export1, index=False)
df_outliers.to_parquet(caminho_export2, index=False)



#df_comp_new = df_comp[df_comp['id'].isin(ids_so_em_b)]
