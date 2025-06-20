'''Esercizio 2.1 – Matrici casuali
Genera una matrice 5x5 con numeri casuali tra 0 e 100. Calcola:
- La somma di ogni riga,
- Il massimo di ogni colonna,
- La matrice trasposta.'''

import numpy as np

# Creazione della matrice
random_array = np.random.randint(0, 100, 25)
print(random_array)
random_array = random_array.reshape(5, 5)
print("La mia matrice ")
print(random_array)

# 1) Somma di ogni riga
sum_rows = random_array.sum(axis=1)
print(f"Somma di ogni riga {sum_rows}")

# 2) Massimo per ogni colonna
col = random_array.max(axis=0)
print(f"Massimo per ogni colonna {col}")

# 3) Matrice trasposta
new_matrix = random_array.T
print("Matrice trasposta")
print(new_matrix)