'''Esercizio 3.2 – CSV e analisi temporale
Leggi un file CSV (puoi crearne uno semplice con colonne Data, Temperatura, Umidità).
- Converte la colonna Data in formato datetime,
- Traccia un grafico a linee della temperatura nel tempo,
- Calcola la media settimanale e disegnala sopra al grafico.'''

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Creazione del CSV e lettura
data = {'Data': ['02.08.2024', '03.08.2024', '04.08.2024'],
        'Temperatura': [35, 34, 30],
        'Umidità': [50, 43, 39]}

df = pd.DataFrame(data, columns=['Data', 'Temperatura', 'Umidità'])
df.to_csv('my_data.csv')

pd.read_csv('my_data.csv', header=None)

# 1) Conversione della colonna in formato Date Time
df['Data'] = pd.to_datetime(df['Data'])
#print(df.dtypes)

# 2) Grafico a linee della temperatura
x = df['Data']
y = df['Temperatura']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(x, y, color='red', linewidth=3)
plt.show()

