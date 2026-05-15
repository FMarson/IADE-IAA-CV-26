import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# -------------------------------------------------
# Dataset pequeno e ruidoso
# -------------------------------------------------
print("Gerando dataset pequeno e ruidoso...")
# X, y = make_classification(
#     n_samples=120,          # número total de amostras
#     n_features=20,          # número total de atributos
#     n_informative=5,        # número de atributos realmente informativos
#     n_redundant=5,          # número de atributos redundantes (cópias lineares)
#     n_clusters_per_class=2, # cada classe tem 2 clusters internos
#     flip_y=0.15,            # ruído nos rótulos
#     class_sep=0.8,          # classes pouco separadas
#     random_state=42         # fixa a aleatoriedade
# )

X, y = make_classification(
    n_samples=120,          # número total de amostras
    n_features=20,          # número total de atributos
    n_informative=5,        # número de atributos realmente informativos
    n_redundant=5,          # número de atributos redundantes (cópias lineares)
    n_clusters_per_class=3, # cada classe tem 1 cluster interno
    flip_y=0.15,            # ruído nos rótulos
    class_sep=0.8,          # classes pouco separadas
    random_state=42         # fixa a aleatoriedade
)

# -------------------------------------------------
# Separação treino e teste
# -------------------------------------------------
print("Separando dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------------------------
# Definir grade inicial de valores de C (escala logarítmica)
# -------------------------------------------------
C_values = [0.01, 0.1, 1, 10, 100]

param_grid = {'C': C_values}

# -------------------------------------------------
# Validação cruzada para escolher C
# -------------------------------------------------
print("Realizando validação cruzada...")
svm = SVC(kernel='linear')

grid = GridSearchCV(
    svm,
    param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',
    return_train_score=True
)

grid.fit(X_train, y_train)

# -------------------------------------------------
# Resultados da validação cruzada
# -------------------------------------------------
print("Resultados da validação cruzada:")
for mean, std, params in zip(
    grid.cv_results_['mean_test_score'],
    grid.cv_results_['std_test_score'],
    grid.cv_results_['params']
):
    print(f"C = {params['C']:<6} | Acurácia média = {mean:.3f} ± {std:.3f}")

print("\nMelhor valor de C:", grid.best_params_['C'])

# -------------------------------------------------
# Tabela em Python puro (percentual)
# -------------------------------------------------
print("\nTabela de acurácia (%)")
print("C\t\tTreino (%)\tValidação (%)")
print("-" * 45)

for C, train, test in zip(
    C_values,
    grid.cv_results_['mean_train_score'],
    grid.cv_results_['mean_test_score']
):
    print(f"{C:<8}\t{train*100:>6.2f}\t\t{test*100:>6.2f}")

# -------------------------------------------------
# Treino do modelo final com o melhor C
# -------------------------------------------------
best_svm = grid.best_estimator_
best_svm.fit(X_train, y_train)

# -------------------------------------------------
# Avaliação final no conjunto de teste
# -------------------------------------------------
y_pred = best_svm.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print("Acurácia no conjunto de teste:", round(test_accuracy, 3))

# -------------------------------------------------
# Gráficos da variação da acurácia
# -------------------------------------------------
mean_test = grid.cv_results_['mean_test_score']
mean_train = grid.cv_results_['mean_train_score']

# -------------------------------------------------
# Plotar gráficos
# -------------------------------------------------
plt.figure(figsize=(8, 5))
plt.semilogx(C_values, mean_train, marker='o', label='Acurácia (treino)')
plt.semilogx(C_values, mean_test, marker='s', label='Acurácia (validação)')

plt.xlabel('Valor de C (escala log)')
plt.ylabel('Acurácia')
plt.title('Variação da acurácia em função de C')
plt.legend()
plt.grid(True)

plt.savefig('./SVM/svm_accuracy.png')