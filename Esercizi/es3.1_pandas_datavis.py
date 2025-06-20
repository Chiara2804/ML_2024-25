'''Esercizio 3.1 – Dataset fittizio studenti
Crea un DataFrame Pandas con i seguenti campi: Nome, Corso, Esame, Voto, CFU.
Scrivi codice per:
- Visualizzare i voti medi per corso,
- Visualizzare con un barplot i CFU medi per esame,
- Visualizzare la distribuzione dei voti con un histogram o un boxplot.'''

import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

# Creazione del dataframe
data = {'Nome': ['Chiara', 'Federico', 'Alice', 'Elisa'],
        'Corso': ['Ingegneria', 'Economia', 'Ingegneria', 'Lettere'],
        'Esame': ['ML', 'Matematica', 'ML', 'Latino'],
        'Voto': [28, 30, 25, 18],
        'CFU': [9, 12, 9, 12]}
df = pd.DataFrame(data, columns=['Nome', 'Corso', 'Esame', 'Voto', 'CFU'])

print(df)

# 1) Voti medi per corso
media_voti_per_corso = df.groupby("Corso")["Voto"].mean()
print(media_voti_per_corso)

# 2) Visualizzazione dei CFU medi per esame con barplot
sns.barplot(x="Esame", y="CFU", data=df)
plt.title("CFU per esame")
plt.show()

# 3) Visualizzazione della distribuzione dei voti
sns.boxplot(x="Esame", y="Voto", data=df)
plt.title("Distribuzione dei voti")
plt.show()