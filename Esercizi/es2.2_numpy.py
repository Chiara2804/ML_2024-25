'''Esercizio 2.2 - Operazioni su array
Crea due array NumPy di lunghezza 100 con numeri interi casuali tra 0 e 50.
Calcola:
- L'array differenza,
- L'array con solo i valori comuni (intersezione),
- La correlazione tra i due array.'''

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Creazione degli array
first = np.random.randint(0, 50, 100)
second = np.random.randint(0, 50, 100)
print("Primo array")
print(first)
print("Secondo array")
print(second)

# 1) Array differenza
difference = np.subtract(second, first)
print("Array differenza")
print(difference)

# 2) Array intersezione
intersection = np.intersect1d(first, second)
print("Array intersezione")
print(intersection)

# 3) Correlazione tra i due array
corr_matrix = np.corrcoef(first, second)
print(f"Correlazione tra i due array {corr_matrix}")

sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
plt.title("Heatmap della correlazione")
plt.show()
