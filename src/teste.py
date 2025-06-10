import requests
import json
import re


# Exemplo: Código IBGE de Brasília
codigo_ibge = "5300108"
url = f"https://apiprevmet3.inmet.gov.br/previsao/{codigo_ibge}"

# Fazendo a requisição
response = requests.get(url)
data = response.json()

# Salvando em um arquivo local (nomeie como quiser)
with open("inmet_brasilia.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("Arquivo salvo como inmet_brasilia.json")

import json
import pandas as pd

# Carregar o arquivo salvo
with open('inmet_brasilia.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extrair dados da cidade pelo código
cidade = "5300108"
resultados = []

for data_dia, periodos in data[cidade].items():
    if re.match(r"\d{2}/\d{2}/\d{4}", data_dia):
        for periodo, info in periodos.items():
            if isinstance(info, dict):
                linha = {'data': data_dia, 'periodo': periodo}
                linha.update(info)
                resultados.append(linha)

df = pd.DataFrame(resultados)
print(df[['data', 'periodo', 'temp_max', 'temp_min', 'umidade_max', 'umidade_min']])
