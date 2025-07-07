# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 21:34:55 2025

@author: eosjo
"""



#%% Import data
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf # estimação do modelo logístico binário
import statsmodels.api as sm # estimação de modelos
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score, f1_score
    
from sklearn.metrics import roc_curve, auc
import seaborn as sns

#%% Import data

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
df_train= pd.read_csv(caminho_arquivo3) 
#df_train = pd.read_parquet(caminho_arquivo3_parquet)



#%% drop Na columns

df_train.dropna(axis=0, inplace=True)

#%% Inner Join between Tables
df_y_train = df_train[["Personality"]]

#%% drop Id column and  
df_train.drop(['id','Personality'], axis=1, inplace=True)

#%% One hot encoding
colunas_one_hot_encoding = ["Stage_fear", "Drained_after_socializing"]

df_x_train = pd.get_dummies(df_train, colunas_one_hot_encoding, dtype=int, drop_first=True)

df_y_train['Personality'] = np.where(df_y_train['Personality'] == 'Extrovert' , 1,0)


#%% One hot encoding
x_columns_list= list(df_x_train.columns)

formula_Logistic =[]
formula_Logistic = np.concatenate((formula_Logistic,x_columns_list),axis=0)
formula_Logistic = ' + '.join(formula_Logistic)
formula_Logistic = "Personality ~ " + formula_Logistic
print("Fórmula utilizada: ",formula_Logistic)




#%% Logistic_model

df_x_train['Personality'] = df_y_train['Personality']

model = smf.glm(formula= formula_Logistic, data=df_x_train,
                     family=sm.families.Binomial()).fit()


from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score
    
#%% Função para calculo de matriz de confusão
def matriz_confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    #cm = confusion_matrix(predicao_binaria, observado)
    #disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    #disp.plot()
    #plt.xlabel('True')
    #plt.ylabel('Classified')
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    #plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)
    f1 = f1_score(observado, predicao_binaria)
    

    #Visualizando os principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia],
                                'F1':[f1]})
    return indicadores   
    


#%% Parâmetros do modelo
model.summary()


# Adicionando os valores previstos de probabilidade na base de dados
df_x_train['predict'] = model.predict()
df_x_train['Logit'] = model.predict(linear=True)


list_metrics = []
list_cuttof = []
list_sensitividade = []
list_especificidade = []
list_acuracia = []
list_f1 =[]

 # Matriz de confusão para cutoff = cuttof
for cut in np.arange(0, 1.001, 0.05):  # vai até 1.0 inclusive

    parametros = matriz_confusao(observado=df_x_train['Personality'],
            predicts=df_x_train['predict'], 
             cutoff=cut)
    
    list_cuttof.append(cut)
    list_sensitividade.append(parametros['Sensitividade'].iloc[0])
    list_especificidade.append(parametros['Especificidade'].iloc[0])
    list_acuracia.append(parametros['Acurácia'].iloc[0])  
    list_f1.append(parametros['F1'].iloc[0])  

df_resultados = pd.DataFrame({
    'Cutoff': list_cuttof,
    'Sensitividade': list_sensitividade,
    'Especificidade': list_especificidade,
    'Acuracia': list_acuracia,
    'F1': list_f1
})
 
#%% Construção de gráficos

# Construção da curva ROC
from sklearn.metrics import roc_curve, auc

# Função 'roc_curve' do pacote 'metrics' do sklearn

fpr, tpr, thresholds =roc_curve(df_x_train['Personality'], df_x_train['predict'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()


# Plotando Sigmoide em função do logíto

plt.figure(figsize=(15,10))
sns.regplot(x=df_x_train['Logit'], y=df_x_train['predict'],
            ci=None, marker='o', logistic=True,
            scatter_kws={'color':'orange', 's':250, 'alpha':0.7},
            line_kws={'color':'darkorchid', 'linewidth':7})
plt.axhline(y = 0.5, color = 'grey', linestyle = ':')
plt.xlabel('Logito (Z)', fontsize=20)
plt.ylabel('Probabilidade de evento (P)', fontsize=20)
plt.xticks(np.arange(df_x_train['Logit'].min() - 0.01 , df_x_train['Logit'].max() + 0.01),
           fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show

