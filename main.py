import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import Bayes
import knn

# definicion de porcentaje de elementos en conjunto de prueba
percentage_test = 0.1

# carga del conjunto de datos digits
X, y = load_digits(return_X_y=True)

# separacion de subconjuntos disjuntos de entrenamiento y prueba
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=percentage_test, stratify=y)

predictions = knn.knn(X_tr, X_tr, y_tr)
accuracy = knn.accuracy(predictions, y_tr)
print(accuracy, "% of accuracy training with KNN")

predictions = knn.knn(X_te, X_tr, y_tr)
accuracy = knn.accuracy(predictions, y_te)
print(accuracy, "% of accuracy test with KNN")

predictions = Bayes.Bayes(X_tr, X_tr, y_tr)
accuracy = Bayes.accuracy(predictions, y_tr)
print(accuracy, "% of accuracy training with Bayes")

predictions = Bayes.Bayes(X_te, X_tr, y_tr)
precision = Bayes.accuracy(predictions, y_te)
print(accuracy, "% of accuracy test with Bayes")

# Visualización del primer individuo del test
plt.gray()
plt.matshow(np.reshape(X_te[0], (8, 8)))
plt.show()
