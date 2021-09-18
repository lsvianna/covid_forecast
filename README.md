# Programa para predição de casos de COVID-19 nos municípios de Santa Catarina

Programa para executar a predição da incidência diária de COVID-19 nos municípios de Santa Catarina, através da execução de uma modelagem de dados com um algoritmo de aprendizagem de máquina. Uma rede neural recorrente foi contruída para modelar um problema de regressão com propósito preditivo, através de um estudo epidemiológico longitudinal retrospectivo da incidência de COVID-19 nos municípios analisados. A métrica de avaliação RMSE foi utilizada para avaliar os modelos obtidos. <br/>

**INPUT:** <br/>
start_date: data determinada para início da série temporal <br/>
end_date: data determinada para fim da série temporal <br/>
city: cidade para avaliação individual <br/>
file: arquivo com conjunto de dados <br/>
file_mun: arquivo com relação de municípios <br/>
steps_in: quantidade de dias de entrada para modelagem <br/>
steps_out: horizonte de predição <br/>
split_size: fração dos dados de treino <br/>
epochs: épocas do modelo <br/>
batch: batches do modelo: <br/>
nodes: quantidade de neurônios de referência <br/>

**OUTPUT:** <br/>
RMSE do modelo construído <br/>

**OUTPUT FILES:** <br/>
acf.jpg: Gráfico de autocorrelação total <br/>
error.jpg: Gráfico erro em cada dia do horizonte de predição <br/>
error_metric.csv: Erro em cada dia do horizonte de predição <br/>
horizon.jpg: Gráficos com amostras nos diferentes horizontes de predição <br/>
incidence.jpg: Gráfico com os dados da incidência e a média móvel <br/>
pacf.jpg: Gráfico de autocorrelação parcial <br/>
predictions.csv: Predições do modelo <br/>
predictions.jpg: Gráfico com os dados da incidência, predições e suas médias móveis <br/>
predictions_city.csv: Predições do modelo na cidade individualizada <br/>
predictions_city.jpg: Gráfico com os dados da incidência, predições e suas médias móveis, na cidade individualizada <br/>
