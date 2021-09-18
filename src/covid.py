# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 09:45:19 2021

@author: viannaLS

Programa para executar a‌ ‌predição‌ ‌da‌ ‌incidência‌ ‌diária‌ ‌de‌ ‌COVID-19‌ nos‌
municípios‌ ‌de‌ ‌Santa‌ Catarina,‌ ‌através‌ ‌da‌ ‌execução‌ ‌de‌ ‌uma‌ ‌modelagem‌ ‌de‌ ‌dados‌ com‌
um‌ ‌algoritmo‌ ‌de‌ aprendizagem‌ ‌de‌ ‌máquina. Uma‌ ‌rede‌ ‌neural‌ ‌recorrente‌ ‌foi‌
contruída ‌para‌ ‌modelar‌ ‌um‌ ‌problema‌ ‌de‌ ‌regressão‌ ‌com‌ propósito‌ ‌preditivo,‌ ‌através‌
de‌ ‌um‌ ‌estudo‌ epidemiológico‌ ‌longitudinal‌ ‌retrospectivo‌ ‌da‌ incidência‌ ‌de‌ ‌COVID-19‌
nos‌ municípios‌ ‌analisados.‌ ‌A‌ ‌métrica‌ ‌de‌ ‌avaliação‌ ‌RMSE‌ ‌foi‌ utilizada‌ ‌para‌
avaliar‌ os‌ ‌modelos‌ ‌obtidos.‌

INPUT:
start_date: data determinada para início da série temporal
end_date: data determinada para fim da série temporal
city: cidade para avaliação individual
file: arquivo com conjunto de dados
file_mun: arquivo com relação de municípios
steps_in: quantidade de dias de entrada para modelagem
steps_out: horizonte de predição
split_size: fração dos dados de treino
epochs: épocas do modelo
batch: batches do modelo:
nodes: quantidade de neurônios de referência

OUTPUT:
RMSE do modelo construído

FILES:
acf.jpg: Gráfico de autocorrelação total
error.jpg: Gráfico erro em cada dia do horizonte de predição
error_metric.csv: Erro em cada dia do horizonte de predição
horizon.jpg: Gráficos com amostras nos diferentes horizontes de predição
incidence.jpg: Gráfico com os dados da incidência e a média móvel
pacf.jpg: Gráfico de autocorrelação parcial
predictions.csv: Predições do modelo
predictions.jpg: Gráfico com os dados da incidência, predições e suas médias
móveis
predictions_city.csv: Predições do modelo na cidade individualizada
predictions_city.jpg: Gráfico com os dados da incidência, predições e suas
médias móveis, na cidade individualizada
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import random
from math import sqrt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import RepeatVector, TimeDistributed
from sklearn.metrics import mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Define o valor da semente nos diferentes ambientes aplicados
seed_value= 12345
os.environ['PYTHONHASHSEED']=str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

# Configura sessão padrão do Keras
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                                        inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),
                            config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

def pivot_table(data, pivot_arg):
    index, columns, values, sel_condition = pivot_arg
    data_condition = data[data[values] == sel_condition]
    table_case = data_condition.pivot_table(index = index,
                                            columns = columns,
                                            values = values,
                                            aggfunc = len,
                                            fill_value = 0)
    date_min = table_case.index.min()
    date_max = table_case.index.max()
    datas = pd.date_range(start = date_min, end = date_max)
    table_case = table_case.reindex(datas, fill_value = 0)
    return table_case

def train_test_split(data, percent):
    split = int(len(data.index) * percent)
    train = data[:split]
    test = data[split:]
    return train, test

def split_sequences(data, steps_in, steps_out):
    X, y = list(), list()
    for i in range(len(data)):
        end_ix = i + steps_in
        out_end_ix = end_ix + steps_out
        if out_end_ix > len(data):
            break
        seq_x = data[i:end_ix, :]
        seq_y = data[end_ix:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def build_model(X, y, model_arg):
    epochs, batch, features, steps_in, steps_out, nodes = model_arg
    model = Sequential()
    model.add(LSTM(nodes, activation = 'relu',
                   input_shape = (steps_in, features),
                   kernel_initializer = 'glorot_normal',
                   recurrent_initializer = 'glorot_normal',
                   bias_initializer = 'random_normal',
                   kernel_regularizer = 'l2'))
    model.add(RepeatVector(steps_out))
    model.add(LSTM(nodes, activation='relu', return_sequences=True,
                    kernel_initializer = 'glorot_normal',
                    recurrent_initializer = 'glorot_normal',
                    bias_initializer = 'random_normal',
                    kernel_regularizer = 'l2'))
    model.add(TimeDistributed(Dense((int(nodes / 2)), activation = 'relu',
                                    kernel_initializer = 'glorot_normal',
                                    bias_initializer = 'random_normal',
                                    kernel_regularizer = 'l2')))
    model.add(TimeDistributed(Dense(features, activation = 'relu',
                                    kernel_initializer = 'glorot_normal',
                                    bias_initializer = 'random_normal',
                                    kernel_regularizer = 'l2')))
    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, epochs = epochs, batch_size = batch, verbose = False)  
    return model


# Variáveis de entrada
start_date = '2020-03-09'
end_date = '2021-03-30'
print('data do modelo,{}'.format(end_date), file = open('model_date.csv', 'w'))
file = 'ftp://boavista:dados_abertos@ftp.ciasc.gov.br/boavista_covid_dados_abertos.csv'
#file = 'boavista_0327.csv'
file_mun = 'data/municipio_populacao.csv'
steps_in = 7
steps_out = 14
split_size = 0.7
epochs = 300
batch = 8
nodes = 200

# Lê o arquivo do banco de dados, ajusta das datas e nomina as colunas
dataset = pd.read_csv(file, sep = ';',
                      usecols = ['obito',
                                 'data_obito',
                                 'data_resultado',
                                 'municipio_notificacao',
                                 'classificacao'])
dataset['data_obito'] = pd.to_datetime(dataset['data_obito']).dt.date
dataset['data_resultado'] = pd.to_datetime(dataset['data_resultado']).dt.date
dataset.columns = ['obito',
                   'data_obito',
                   'data_resultado',
                   'municipio',
                   'classificacao']

# Tabula os dados
sel_condition = 'CONFIRMADO'
pivot_arg = ['data_resultado',
             'municipio',
             'classificacao',
             sel_condition]
table = pivot_table(dataset, pivot_arg)

# Seleciona os municípios catarinenses da origem do paciente
mun = pd.read_csv(file_mun, sep = ',')
table = table.loc[:, table.columns.isin(mun.municipio)]
table_rolling = table.rolling(window = 7).mean()

# Limita período dos dados
table_filtered = table[(table.index >= start_date) & (table.index <= end_date)]
table_ma = table_rolling[(table_rolling.index >= start_date) &
                         (table_rolling.index <= end_date)]

# Visualização dos dados: evolução dos casos
plt.figure(figsize = (18, 9))
plt.title('Inicidência de COVID-19 em Santa Catarina')
plt.plot(np.sum(table_ma.transpose()), label = 'Média móvel', color = 'r')
plt.plot(np.sum(table_filtered.transpose()), label = 'Incidência diária',
         color = 'b')
plt.legend(loc = 'upper left')
plt.savefig('static/incidence.jpg', dpi = 600)

# Divide dados de treino e teste
train, test = train_test_split(table_filtered, split_size)
X_train, y_train = split_sequences(train.values, steps_in, steps_out)
X_test, y_test = split_sequences(test.values, steps_in, steps_out)
if (X_test.size == 0 or y_test.size == 0): raise Exception('Not enough data!')

# Constroi o modelo
features = X_train.shape[2]
model_arg = [epochs, batch, features, steps_in, steps_out, nodes]
model = build_model(X_train, y_train, model_arg)

# Salva modelo e tabelas
model.save('model_covid')
save('X_test.npy', X_test)
save('y_test.npy', y_test)
save('test_predictions.npy', test_predictions)
table_filtered.to_csv('table_filtered.csv')