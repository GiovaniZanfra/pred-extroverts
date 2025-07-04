# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 23:16:16 2025

@author: eosjo
"""
#%% Import bibliotecas

import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # necessário para 3D
import seaborn as sns
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
import numpy as np
#%% Import das variáveis do modelo#%% Import data

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
df_train= pd.read_csv(caminho_arquivo3) 
df_original = pd.read_csv(caminho_arquivo3) 
df_train_original = pd.read_csv(caminho_arquivo3) 

#%% drop columns no related to features: 
    
df_train.dropna(axis=0, inplace=True)
df_original.dropna(axis=0, inplace=True)
df_train.drop(['id','Personality'], axis=1, inplace=True)

#%%Defina colunas numéricas e categóricas
# ────────────────────────────────────────────
# 2. Defina colunas numéricas e categóricas
# ────────────────────────────────────────────
num_cols = ["Time_spent_Alone", "Social_event_attendance","Going_outside","Friends_circle_size","Post_frequency"]
cat_cols = ["Stage_fear", "Drained_after_socializing"]

# ────────────────────────────────────────────
# 3. Crie o pré‑processador
# ────────────────────────────────────────────
preprocessador = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(sparse=False, handle_unknown="ignore"), cat_cols)
    ]
)

# ────────────────────────────────────────────
# 4. Monte o pipeline completo com PCA
# ────────────────────────────────────────────
modelo_pca = Pipeline(steps=[
    ("prep", preprocessador),
    ("pca", PCA(n_components=2, random_state=1))
])

# ────────────────────────────────────────────
# 5. Ajuste + transforme os dados
# ────────────────────────────────────────────
componentes_2d = modelo_pca.fit_transform(df_train)
df_componentes = pd.DataFrame(componentes_2d, columns=["PC1", "PC2"])
df_original["PC1"] = df_componentes["PC1"]
df_original["PC2"] = df_componentes["PC2"]


# Resultado: matriz 6 × 2
print("Componentes principais (2D):")
print(pd.DataFrame(componentes_2d, columns=["PC1", "PC2"]))

# Variância explicada
explicada = modelo_pca.named_steps["pca"].explained_variance_ratio_
print("\nVariância explicada por cada componente:", explicada.round(3))
print("Variância acumulada:", explicada.sum().round(3))

#%%
# ────────────────
# 5. Plotar gráfico
# ────────────────
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df_original, x="PC1", y="PC2", hue="Personality", palette="Set1", s=100)
plt.title("Visualização das componentes principais (PCA)")
plt.xlabel("Componente Principal 1")
plt.ylabel("Componente Principal 2")
plt.grid(True)
plt.legend(title="Classe")
plt.tight_layout()
plt.show()



#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from itertools import combinations

# Lista de variáveis numéricas
num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside",
            "Friends_circle_size", "Post_frequency"]

# Cria todas as combinações possíveis de pares (sem repetição)
combinacoes = list(combinations(num_cols, 2))

# Para cada par (x, y), aplica KMeans e salva resultado
for idx, (x, y) in enumerate(combinacoes, start=1):
    # Seleciona as colunas e normaliza
    X = df_original[[x, y]]
    X_scaled = StandardScaler().fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=[x, y], index=df_original.index)
    X_scaled_df["Personality"] = df_original["Personality"]
    
    # Índice da coluna categórica (sempre a segunda coluna, índice 1)
    categorical_index = [2]
    # Aplica KMeans
   
    # Inicializa e treina o modelo KPrototypes
    kproto = KPrototypes(n_clusters=2, init='Cao', random_state=1)
    clusters = kproto.fit_predict(X_scaled_df, categorical=categorical_index)
    # Salva os clusters com nome cluster1, cluster2, ...
    df_original[f"cluster{idx}"] = clusters

    # Visualização
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_original, x=x, y=y, hue=f"cluster{idx}", palette="Set2", s=100)
    plt.title(f"K-Means: {x} vs {y}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.grid(True)
    plt.legend(title="Cluster")
    plt.tight_layout()
    plt.show()

#%%
# Listas de colunas numéricas e categóricas
num_cols = ["Time_spent_Alone", "Social_event_attendance", "Going_outside", "Friends_circle_size", "Post_frequency"]
cat_cols = ["Stage_fear", "Drained_after_socializing"]
target_cols = ["Personality"]

# Inicializa o contador de clusters
cluster_counter = 11

# Itera sobre todas as combinações de colunas numéricas e categóricas
for num_col in num_cols:
    for cat_col in cat_cols:
        # Cria o DataFrame X com a combinação atual
        X = df_original[[num_col, cat_col] + target_cols]

        # Índice da coluna categórica (sempre a segunda coluna, índice 1)
        categorical_index = [1,2]

        # Inicializa e treina o modelo KPrototypes
        kproto = KPrototypes(n_clusters=2, init='Cao', random_state=1)
        clusters = kproto.fit_predict(X, categorical=categorical_index)

        # Nome da nova coluna de cluster
        cluster_col_name = f"cluster{cluster_counter}"
        df_original[cluster_col_name] = clusters

        # Visualização
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_original, x=num_col, y=cat_col, hue=cluster_col_name, palette="Set2", s=100)
        plt.title(f"KPrototypes com 2 Clusters: {num_col} vs {cat_col}")
        plt.xlabel(num_col)
        plt.ylabel(cat_col)
        plt.grid(True)
        plt.legend(title="Cluster")
        plt.tight_layout()
        plt.show()

        # Incrementa o contador de clusters
        cluster_counter += 1
#%%
def inverter_clusters(df, cluster_indices):
    for x in cluster_indices:
        col_name = f"cluster{x}"
        if col_name in df.columns:
            # Garante que só 0 e 1 sejam invertidos
            df[col_name] = df[col_name].apply(lambda v: 1 if v == 0 else (0 if v == 1 else v))
        else:
            print(f"Coluna {col_name} não encontrada.")
    return df

def somar_clusters(df, inicio=1, fim=20):
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

    for i in range(1, n_clusters + 1):
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


def filtrar_outliers(df, coluna="outliers_soma", limite=10):
    n_total = len(df)
    df_filtrado = df[df[coluna] <= limite].copy()
    n_removidos = n_total - len(df_filtrado)
    return df_filtrado, n_removidos

def manter_colunas_em_comum(df_original, df_referencia):
    colunas_comuns = [col for col in df_original.columns if col in df_referencia.columns]
    return df_original[colunas_comuns].copy()



#%%
# Lista dos clusters que deseja inverter
clusters_para_inverter = [2, 4, 5, 8, 9, 10, 11, 12, 13, 14]
#%%
# Aplica a inversão
df_original = inverter_clusters(df_original, clusters_para_inverter)

df_original = marcar_pesos_outlier(df_original)  # padrão: clusters 1‑20

df_original = somar_clusters(df_original)
df_original = somar_outliers(df_original)

df_comp = df_original[["id", "Personality", "outliers_soma", "cluster_soma"]]

#%%
df_original, n_removidos = filtrar_outliers(df_original)
print(f"Foram removidas {n_removidos} observações onde  > 10.")

df_original = manter_colunas_em_comum(df_original, df_train_original)

df_original.to_parquet("data_no_Outliers.parquet", index=False)





