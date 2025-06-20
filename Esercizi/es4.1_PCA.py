'''Esercizio 4.1 - PCA su dataset sintetico
Crea un dataset NumPy di dimensioni 100x5 con dati casuali correlati.
- Standardizza i dati,
- Applica PCA per ridurre a 2 componenti principali,
- Visualizza i punti nel nuovo piano 2D,
- Calcola la varianza spiegata da ogni componente.'''

import numpy as np
import matplotlib.pyplot as plt

# Creazione del dataset
random_array = np.random.randint(0, 100, 500)
random_array = random_array.reshape(100, 5)
print(random_array)

# 1) Applico PCA per ridurre a 2 componenti principali ì
# - Standardizzazione dei dati -> Z-score
X_mean = random_array.mean(axis=0) # media per colonna
X_std = random_array.std(axis=0) # deviazione standard per colonna

X_scaled = (random_array - X_mean) / X_std # standardizzazione

# - Matrice di covarianza
cov_matrix = np.cov(X_scaled, rowvar=False)

# - Decomposizione in autovalori e autovettori
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# - Ordinamento
# Ordina indici in base agli autovalori decrescenti
idx = np.argsort(eigenvalues)[::-1]

# Riordina autovalori e autovettori
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Proiezioni
k = 2 # quante proeizioni
PCs = eigenvectors[:, :k]  # prende le prime k colonne (i vettori principali)

X_pca = X_scaled @ PCs  # prodotto matrice * autovettori

plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Proiezione PCA")
plt.grid(True)
plt.show()


# 3) Calcolo varianza spiegata da ogni componente
explained_variance = eigenvalues / eigenvalues.sum()
print(explained_variance)


