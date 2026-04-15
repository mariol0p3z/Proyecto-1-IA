dataset = 'data/customer_support_tickets.csv'
columna_categorica = 'Ticket Type'
columna_texto = 'Ticket Description'

categorias = [
    'Refund request',
    'Technical issue',
    'Cancellation request',
    'Product inquiry',
    'Billing inquiry'
]

laplace_alpha = 1.0
k_folds = 5
semilla = 42

models_dir = 'models'