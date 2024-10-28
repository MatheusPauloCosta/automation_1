#import library
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys

#Defining a function to generate data visualization plots


def plot_pivot_table(df,value,index,func,y_label,x_label, opcao='nothing'):
    if opcao == 'nothing':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).plot(figsize=[15,5])
    elif opcao == 'sort':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).sort_values(value).plot(figsize=[15,5])
    elif opcao == 'unstack':
        pd.pivot_table(df, values=value, index=index, aggfunc=func).unstack().plot(figsize=[15,5])
    plt.xticks(rotation=10)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    return None

#
month_list = sys.argv[1:]

for month in month_list:
    #loading df
    sinasc = pd.read_csv('./input/SINASC_RO_2019_'+month+'.csv')

    
    print(sinasc.DTNASC.min(), sinasc.DTNASC.max())
    
        
    #Retrieving the last month to set as the folder name
    max_data = sinasc.DTNASC.max()[:7]
    print(max_data)
    
    #Creating a directory for data storage
    os.makedirs('./imagens/'+max_data, exist_ok=True)
    
    #Visualizations for data analysis
    
    plot_pivot_table(sinasc,'IDADEMAE','DTNASC','count','Quantidade de Nascimento','Data de Nascimento')
    plt.savefig('./imagens/'+max_data+'/quantidade_de_nascimento.png')
    
    plot_pivot_table(sinasc,'IDADEMAE',['DTNASC','SEXO'],'mean','Média Idade Mae','Data de Nascimento','unstack')
    plt.savefig('./imagens/'+max_data+'/media_idade_mae_por_sexo.png')
    
    plot_pivot_table(sinasc,'PESO',['DTNASC','SEXO'],'mean','Média Peso Bebe','Data de Nascimento','unstack')
    plt.savefig('./imagens/'+max_data+'/media_peso_bebe_por_sexo.png')
    
    plot_pivot_table(sinasc,'PESO','ESCMAE','median','PESO','ESCMAE','sort')
    plt.savefig('./imagens/'+max_data+'/peso_bebe_por_escolaridade_mae.png')
    
    plot_pivot_table(sinasc,'APGAR1','GESTACAO','mean','Apgar1','Gestação','sort')
    plt.savefig('./imagens/'+max_data+'/media_apgar1_por_gestaçao.png')
    
    plot_pivot_table(sinasc,'APGAR5','GESTACAO','mean','Apgar5','Gestaçao','sort')
    plt.savefig('./imagens/'+max_data+'/media_apgar5_por_gestaçao.png')
        
    
