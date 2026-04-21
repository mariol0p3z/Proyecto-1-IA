dataset = 'data/bitext_customer_support.csv'
columna_categorica = 'category'
columna_texto = 'instruction'

categorias = [
    'ACCOUNT',
    'CANCEL',
    'CONTACT',
    'DELIVERY',
    'FEEDBACK',
    'INVOICE',
    'ORDER',
    'PAYMENT',
    'REFUND',
    'SHIPPING',
    'SUBSCRIPTION'
]

laplace_alpha = 1.0
k_folds = 5
semilla = 42

models_dir = 'models'