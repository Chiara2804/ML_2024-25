'''Esercizio 4.2 – PCA su dataset Iris (o simile)
Usa il dataset Iris da sklearn.datasets.load_iris().
- Applica PCA e riduci a 2 componenti,
- Visualizza i dati con un scatterplot a colori secondo la specie,
- Analizza quanta varianza viene spiegata dalla PCA.'''

from sklearn.datasets import load_iris

iris = load_iris()

