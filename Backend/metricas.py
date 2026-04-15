import numpy as np

def calcular_metricas(y_real, y_pred, clases):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)

    resultados = {}
    for clase in clases:
        #TP = True Positive, FP = False Positive, TN = True Negative, FN = False Negative
        tp = sum((y_real == clase) & (y_pred == clase))
        fp = sum((y_real != clase) & (y_pred == clase))
        fn = sum((y_real == clase) & (y_pred != clase))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        resultados[clase] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'support': np.sum(y_real == clase)
        }

    accuracy = np.sum(y_real == y_pred) / len(y_real)
    macro_f1 = np.mean([resultados[clase]['f1_score'] for clase in clases])

    resultados['accuracy'] = accuracy
    resultados['macro_f1'] = macro_f1

    return resultados

def matriz_confusion(y_real, y_pred, clases):
    y_real = np.array(y_real)
    y_pred = np.array(y_pred)
    
    n_clases = len(clases)
    matriz = np.zeros((n_clases, n_clases), dtype=int)

    #Mapeo clase -> indice
    clase_a_indice = {clase: indice for indice, clase in enumerate(clases)}

    for real, pred in zip(y_real, y_pred):
        i = clase_a_indice[real]
        j = clase_a_indice[pred]
        matriz[i][j] += 1

    return matriz

def imprimir_metricas(metricas, clases):
    print("Metricas por clase:")
    for clase in clases:
        m = metricas[clase]
        print(f"{clase:<25} {m['precision']:>12.3f} {m['recall']:>12.3f} {m['f1_score']:>12.3f} {m['support']:>12}")
    
    print(f"\n{'Accuracy':<25} {metricas['accuracy']:>12.3f}")
    print(f"{'Macro F1-Score':<25} {metricas['macro_f1']:>12.3f}")

def imprimir_matriz_confusion(matriz, clases):
    print("\nMatriz de Confusión:")
    print("Filas = Clase Real | Columnas = Clase Predicha\n")

    #Imprimir encabezados de columnas
    header = "Real \\ Pred".ljust(25)
    for i, clase in enumerate(clases):
        header += f"{i:>8}"
    print(header)

    #Imprimir filas
    for i, clase in enumerate(clases):
        fila = f"{i}. {clase[:20]}".ljust(25)
        for j in range(len(clases)):
            fila += f"{matriz[i][j]:>8}"
        print(fila)
    
    #Leyenda
    print("\nLeyenda de clases:")
    for i, clase in enumerate(clases):
        print(f"{i}. {clase}")

# ===== PRUEBAS =====
if __name__ == "__main__":
    print("="*80)
    print("PRUEBA DE MÉTRICAS")
    print("="*80)
    
    # Datos de ejemplo
    y_real = np.array([
        'Technical issue', 'Technical issue', 'Billing inquiry',
        'Billing inquiry', 'Product inquiry', 'Cancellation request',
        'Refund request', 'Technical issue', 'Billing inquiry',
        'Product inquiry'
    ])
    
    y_predicho = np.array([
        'Technical issue', 'Technical issue', 'Billing inquiry',
        'Refund request', 'Product inquiry', 'Cancellation request',
        'Refund request', 'Technical issue', 'Billing inquiry',
        'Technical issue'
    ])
    
    clases = [
        'Refund request',
        'Technical issue',
        'Cancellation request',
        'Product inquiry',
        'Billing inquiry'
    ]
    
    # Calcular métricas
    metricas = calcular_metricas(y_real, y_predicho, clases)
    imprimir_metricas(metricas, clases)
    
    # Calcular y mostrar matriz de confusión
    matriz = matriz_confusion(y_real, y_predicho, clases)
    imprimir_matriz_confusion(matriz, clases)