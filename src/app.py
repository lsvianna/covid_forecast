# -*- coding: utf-8 -*-

from flask import Flask, request, render_template
import datetime
import pandas as pd
import os
from tensorflow import keras

try:
    os.remove('static/incidence.jpg')
    os.remove('static/predictions_city.jpg')
except: None

# Variáveis para o app
today = datetime.datetime.now() - datetime.timedelta(days = 1)
mun_table = pd.read_csv('data/municipio_populacao.csv', sep = ',')
municipios = mun_table.municipio.sort_values()

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def index():
    try:
        data = ''
        municipio = ''
        rmse = float()
        media_antes = float()
        media_depois = float()
        variacao = float()
        if request.method == 'POST' and 'data' in request.form:
            data = request.form.get('data')
            municipio = request.form.get('municipio')
            rmse, media_antes, media_depois = pred_covid(data, municipio)
            variacao = ((media_depois - media_antes) / media_antes) * 100

    except Exception as e:
        return render_template('500.html', error = str(e))

    return render_template('index.html',
							today = today,
							municipios = municipios,
							data = data,
							municipio = municipio,
							rmse = round(rmse, 2),
							media_antes = round(media_antes, 0),
							media_depois = round(media_depois, 0),
							variacao = round(variacao, 2))

def pred_covid(data, municipio):
    # Lê modelos e tabelas
    model = keras.models.load_model('model_covid')
    X_test = load('X_test.npy')
    y_test = load('y_test.npy')
    test_predictions = load('test_predictions.npy')
    table_filtered = pd.read_csv('table_filtered.csv')
    
    # Faz as predições de teste
    test_predictions = model.predict(X_test);

    # Avalia o resultado geral do modelo
    rmse = sqrt(mean_squared_error(y_test.flatten(), test_predictions.flatten()))

    # Constrói predição futura
    prediction = model.predict(table_filtered.iloc[-7:].values.reshape(1, 7, -1))[0]
    datas = pd.date_range(start = table_filtered.index.max(),
                          periods = steps_out + 1)[1:]
    predictions = pd.DataFrame(np.round(prediction, 0), index = datas,
                               columns = table_filtered.columns).astype(int)

    table_case = table_filtered.append(predictions)
    table_case_ma = table_case.rolling(window = 7).mean()
    table_city = table_case.loc[:, city]
    table_city_ma = table_city.rolling(window = 7).mean()

    # Visualização dos dados: predição futura município selecionado
    plt.figure(figsize = (10, 6))
    plt.title('Incidência de COVID-19 no Município Selecionado')
    plt.plot([], [], ' ', label = city)
    plt.plot(table_city.transpose(),
             label = 'Incidência diária (valores reais)', color = 'b')
    plt.plot(table_city_ma.transpose(),
             label = 'Média móvel (valores reais)', color = 'r')
    plt.plot(table_city[-steps_out:].transpose(),
             label = 'Incidência diária (valores preditos)', color = 'g')
    plt.plot(table_city_ma[-steps_out:].transpose(),
             label = 'Média móvel (valores preditos)', color = 'orange')
    plt.legend(loc = 'upper left')
    plt.vlines(table_filtered.index.max(), table_city.min(),
               table_city.max(), linestyles = 'dotted')
    plt.text(table_filtered.index.max(), table_city.max(),
             table_filtered.index.max().strftime('%Y-%m-%d'), fontsize = 8)
    plt.savefig('static/predictions_city.jpg', dpi = 600)

    return rmse, table_city_ma.iloc[-(steps_out + 1)], table_city_ma.iloc[-1]


@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r