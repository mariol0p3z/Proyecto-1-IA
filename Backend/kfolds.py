import numpy as np

def dividir_kfolds(X, y, k = 5, shuffle = True, semilla = 42):
    n_muestras = len(X)
    indices = np.arange(n_muestras)

    if shuffle:
        np.random.seed(semilla)
        np.random.shuffle(indices)

    #Calcular el tamaño de cada fold
    tamanio_fold = n_muestras // k
    folds = []

    for i in range(k):
        #Indices para el fold actual
        inicio_test = i *tamanio_fold
        fin_test = (i+1) * tamanio_fold if i < k - 1 else n_muestras
        indices_test = indices[inicio_test:fin_test]

        #Indices para el entrenamiento (todos excepto los del fold actual)
        indices_train = np.concatenate([indices[:inicio_test], indices[fin_test:]])
        folds.append((indices_train, indices_test))
    
    return folds

def obtener_datos_fold(X, y, indices_train, indices_test):
    X_train = [X[i] for i in indices_train]
    X_test = [X[i] for i in indices_test]
    y_train = y[indices_train]
    y_test = y[indices_test]

    return X_train, X_test, y_train, y_test