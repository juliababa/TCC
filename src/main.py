import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error


df_notimecollumn = pd.read_csv("data/2023_cleaned/INMET_CO_DF_A001_BRASILIA_01-01-2023_A_31-12-2023.CSV", encoding="utf-8", decimal=".", sep=";", converters={'Data': str, 'Hora UTC': str})

#mudando tipo para string
df_notimecollumn[['Data', 'Hora UTC']] = df_notimecollumn[['Data', 'Hora UTC']].astype("string")

df_notimecollumn = df_notimecollumn.drop(columns=['Hora UTC'])


column_name = 'TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)'

# print(df_notimecollumn.info())
# print(df_notimecollumn.info())
df_notimecollumn.groupby(['Data'], as_index= True).mean()

print(df_notimecollumn.info())

#Converte Data
df_notimecollumn['Data']=pd.to_datetime(df_notimecollumn['Data'])
df_notimecollumn.set_index('Data', inplace=True)
df_notimecollumn = df_notimecollumn.resample("D").last()

print(df_notimecollumn)


#Teste
plt.plot(df_notimecollumn[column_name], label='Previsão')
plt.savefig('data.png')

# Verificar se há valores nulos
print(df_notimecollumn.isnull().sum())

# Tratar valores nulos
df_notimecollumn = df_notimecollumn.ffill()

# Divisão dos dados em treinamento (80%) e teste (20%)
train_size = int(len(df_notimecollumn) * 0.8)
train, test = df_notimecollumn[:train_size], df_notimecollumn[train_size:]

# Ajuste do modelo ARIMA
model = ARIMA(train[column_name], order=(1, 0, 0), trend='ct')
model_fit = model.fit()

# Fazer previsões
forecast = model_fit.forecast(steps=len(test))

# Visualização das previsões
plt.figure(figsize=(12,6))
plt.plot(train.index, train[column_name], label='Treinamento')
plt.plot(test.index, test[column_name], label='Teste')
plt.plot(test.index, forecast, label='Previsão')
plt.ylabel('Temperatura')
plt.legend(loc='upper left')

# Cálculo erro quadrático
mse = mean_squared_error(test[column_name], forecast)
rmse = mse**0.5 
print(f'MSE: {mse}')
print(f'RMSE: {rmse}')
plt.fill_between(test.index, (forecast-rmse), (forecast+rmse), alpha=.3, color='r', zorder=20)

# Salva o gráfico como arquivo de imagem
plt.savefig('forecast.png')





