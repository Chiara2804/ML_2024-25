''' Esercizio 1 - Regressione lineare con rumore non gaussiano
Obiettivo: Analizzare l'effetto di rumore non gaussiano sul modello.

Task:
- Genera dati 1D con una retta y=3x+2, ma aggiungi rumore uniforme anziché gaussiano.
- Applica LinearRegression e visualizza i risultati.
- Confronta l'errore MSE con il caso gaussiano.'''

import numpy as np
import matplotlib.pyplot as plt


def r_squared(y, y_pred):
    y = np.array(y)  # Convert y to a NumPy array
    y_pred = np.array(y_pred)  # Convert y_pred to a NumPy array
    ss_tot = np.sum((y - np.mean(y))**2)
    ss_res = np.sum((y - y_pred)**2)
    r_squared = 1 - (ss_res / ss_tot)
    return r_squared

def K_fold(X):
    k = 3
    fold_size = len(X) // k

    # shuffle the data
    np.random.seed(0)
    shuffled_indices = np.random.permutation(len(X))
    X_shuffled = X[shuffled_indices]
    y_shuffled = y[shuffled_indices]

    # per ogni fold
    mse_folds_ols = []
    r2_folds_ols = []

    for i in range(k-1):
        # divido i dati in train e test
        start = i * fold_size
        end = (i+1) * fold_size
        test_indices = np.arange(start, end) # indici dei dati di test
        train_indices = np.concatenate((np.arange(0, start), np.arange(end, n_samples))) # indici dei dati di training

        X_train = X_shuffled[train_indices]
        y_train = y_shuffled[train_indices]
        X_test = X_shuffled[test_indices]
        y_test = y_shuffled[test_indices]

        # standardizzo i dati (tutti)
        X_train_mean = np.mean(X_train, axis=0)
        X_train_std = np.std(X_train, axis=0)
        X_train = (X_train - X_train_mean) / X_train_std
        X_test = (X_test - X_train_mean) / X_train_std

        beta_ols = r_squared(X_train, y_train) # fit model

        y_pred_ols = X_test @ beta_ols[1:] + beta_ols[0] # predict on test set

        # metriche
        mse_ols = np.mean((y_test - y_pred_ols)**2)
        r2_ols = 1 - np.sum((y_test - y_pred_ols)**2) / np.sum((y_test - np.mean(y_test))**2)

        mse_folds_ols.append(mse_ols)
        r2_folds_ols.append(r2_ols)

    return mse_folds_ols, r2_folds_ols

#def MCCV():



# 1) Genera dati 1D con una retta y=3x+2, ma aggiungi rumore uniforme anziché gaussiano
np.random.seed(0)
n_samples = 100

# Generazione dei dati x
x = np.linspace(0, 5, 10)

# Rumore uniforme
rumore = np.random.uniform(-1, 1, size=x.shape)

# Calcolo di y con la retta y = 3x + 2 + rumore
y = 3 * x + 2 + rumore

# Visualizzazione
plt.scatter(x, y, color='blue', label='Dati con rumore uniforme')
plt.plot(np.sort(x), 3*np.sort(x) + 2, color='red', label='y = 3x + 2 (senza rumore)')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Dati sintetici con rumore uniforme")
plt.grid(True)
plt.show()


# 2) Applica LinearRegression e visualizza i risultati

# Trovo i parametri: beta = (X^TX)^-1 X^T y
X = np.vstack((np.ones_like(x), x)).T # costruisco X: aggiungo a x una colonna di 1 e metto x in colonna

beta = np.linalg.inv(X.T @ X) @ X.T @ y # Calcolo beta

y_pred = X @ beta # Predizione

# DA FARE - Valutazione parametri
print(f"R^2: {r_squared(y, y_pred)}")
print(K_fold(X))




# Visualizzazione
plt.scatter(x, y, label="Dati con rumore")
plt.plot(x, y_pred, color='red', label="Retta stimata (OLS)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Regressione lineare - OLS matriciale")
plt.show()

# Stampo parametri trovati
print(f"Intercetta (β₀): {beta[0]}")
print(f"Coefficiente (β₁): {beta[1]}")

# DA FARE - Regolarizzazione se serve ...