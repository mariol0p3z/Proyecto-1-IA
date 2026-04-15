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

if __name__ == "__main__":
    print("="*80)
    print("PRUEBA DE K-FOLDS")
    print("="*80)
    
    # Datos de ejemplo
    X = list(range(20))  # [0, 1, 2, ..., 19]
    y = np.array(['A'] * 10 + ['B'] * 10)
    
    print(f"\nDataset de ejemplo: {len(X)} muestras")
    print(f"X: {X}")
    print(f"y: {y}\n")
    
    # Dividir en 5 folds
    k = 5
    folds = dividir_kfolds(X, y, k=k, shuffle=False)
    
    print(f"Dividido en {k} folds:\n")
    
    for i, (indices_train, indices_test) in enumerate(folds, 1):
        print(f"Fold {i}:")
        print(f"  Train: {len(indices_train)} muestras - índices {indices_train[:5]}...{indices_train[-5:]}")
        print(f"  Test:  {len(indices_test)} muestras - índices {list(indices_test)}")
        
        # Verificar que no hay solapamiento
        assert len(set(indices_train) & set(indices_test)) == 0, "¡Error! Hay solapamiento"
        
        # Verificar que cubren todo el dataset
        assert len(indices_train) + len(indices_test) == len(X), "¡Error! No cubren todo"
        
        print(f"  ✓ Sin solapamiento, cubre todo el dataset")
        print()
    
    # Probar con shuffle
    print("\n" + "="*80)
    print("PRUEBA CON SHUFFLE")
    print("="*80 + "\n")
    
    folds_shuffle = dividir_kfolds(X, y, k=k, shuffle=True, semilla=42)
    
    for i, (indices_train, indices_test) in enumerate(folds_shuffle, 1):
        print(f"Fold {i}:")
        print(f"  Test: índices {list(indices_test)}")
    
    print("\n✓ K-Folds funcionando correctamente")

