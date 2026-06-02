import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df = pd.read_csv(r'C:\Users\zazaq\Analise_Pred_Indu\dataset\ai4i2020.csv')

df['Temp_diff [K]'] = df['Process temperature [K]'] - df['Air temperature [K]']


X = df[['Rotational speed [rpm]', 'Temp_diff [K]', 'Tool wear [min]']]
y = df['Torque [Nm]']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Amostras de treino: {len(X_train)}")
print(f"Amostras de teste:  {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print(f"\nErro Quadrático Médio (MSE): {mse:.2f}")
print(f"Raiz do MSE         (RMSE): {np.sqrt(mse):.2f} Nm")
print(f"Coeficiente de Determinação (R2): {r2:.2f}")

nova_operacao = [[1500, 9, 100]]
torque_previsto = model.predict(nova_operacao)
print(f"\nPrevisão de torque para a nova operação: {torque_previsto[0]:.2f} Nm")

nomes        = ['Rotational speed [rpm]', 'Temp_diff [K]', 'Tool wear [min]']
coeficientes = model.coef_
intercepto   = model.intercept_

print("\nCoeficientes (peso de cada variável):")
for nome, valor in zip(nomes, coeficientes):
    print(f"  {nome}: {valor:.6f}")
print(f"Intercepto (termo constante): {intercepto:.4f}")

print(f"\nFormula aproximada do modelo:")
print(
    f"Torque [Nm] = {intercepto:.2f} "
    f"+ ({coeficientes[0]:.4f} * Rotational_speed) "
    f"+ ({coeficientes[1]:.4f} * Temp_diff) "
    f"+ ({coeficientes[2]:.4f} * Tool_wear)"
)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='steelblue', s=10, label='Operações')
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color='red', linewidth=2, linestyle='--', label='Previsão perfeita'
)
plt.title('Torque Real vs Torque Previsto')
plt.xlabel('Torque Real (Nm)')
plt.ylabel('Torque Previsto (Nm)')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('real_vs_previsto.png', dpi=150)
plt.show()

residuos = y_test - y_pred

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(residuos, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
plt.axvline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Distribuição dos Resíduos')
plt.xlabel('Resíduo (Nm)')
plt.ylabel('Frequência')
plt.grid(True, alpha=0.4)

plt.subplot(1, 2, 2)
plt.scatter(y_pred, residuos, alpha=0.3, color='coral', s=10)
plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
plt.title('Resíduos vs Valores Previstos')
plt.xlabel('Torque Previsto (Nm)')
plt.ylabel('Resíduo (Nm)')
plt.grid(True, alpha=0.4)

plt.suptitle('Análise dos Resíduos do Modelo', fontsize=13)
plt.tight_layout()
plt.savefig('residuos.png', dpi=150)
plt.show()

importancias = np.abs(coeficientes) / np.sum(np.abs(coeficientes)) * 100

plt.figure(figsize=(8, 4))
bars = plt.barh(nomes, importancias, color=['steelblue', 'coral', 'orange'], alpha=0.85)
for bar, val in zip(bars, importancias):
    plt.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%', va='center', fontsize=11
    )
plt.title('Importância Relativa das Variáveis')
plt.xlabel('Contribuição (%)')
plt.grid(axis='x', alpha=0.4)
plt.tight_layout()
plt.savefig('importancia_variaveis.png', dpi=150)
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(
    df['Rotational speed [rpm]'], df['Torque [Nm]'],
    alpha=0.2, color='steelblue', s=8, label='Dados reais'
)

x_line = np.linspace(df['Rotational speed [rpm]'].min(),
                    df['Rotational speed [rpm]'].max(), 100)
y_line = intercepto + coeficientes[0] * x_line
plt.plot(x_line, y_line, color='red', linewidth=2, label='Tendência linear')

plt.title('Velocidade de Rotação vs Torque (correlação = -0.88)')
plt.xlabel('Velocidade de Rotação (rpm)')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('rpm_vs_torque.png', dpi=150)
plt.show()