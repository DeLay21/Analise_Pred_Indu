import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import ExponentialSmoothing

df = pd.read_csv(
    r'.\Analise_Pred_Indu\dataset\ai4i2020.csv',
    index_col='UDI'
)
coluna = 'Tool wear [min]'
serie  = df[coluna]

diff  = serie.diff()
ciclo_id = 0
ciclos   = []

for i in range(len(serie)):
    if i > 0 and diff.iloc[i] < 0:
        ciclo_id += 1
    ciclos.append(ciclo_id)

df['ciclo'] = ciclos
ultimo_ciclo = df['ciclo'].max()

print(f"Total de ciclos detectados: {ultimo_ciclo + 1}")
print(f"Ciclos históricos completos: {ultimo_ciclo}")

historicos  = df[df['ciclo'] < ultimo_ciclo]
ciclo_atual = df[df['ciclo'] == ultimo_ciclo][coluna].reset_index(drop=True)

print(f"\nCiclo atual em andamento: ciclo {ultimo_ciclo}")
print(f"  Operação atual:  {len(ciclo_atual)}")
print(f"  Desgaste atual:  {ciclo_atual.iloc[-1]} min")


taxas = []
for cid in range(ultimo_ciclo):
    c = historicos[historicos['ciclo'] == cid][coluna].values
    if len(c) > 1:
        taxas.append(np.polyfit(range(len(c)), c, 1)[0])

taxa_media = np.mean(taxas)
print(f"\nTaxa média histórica: {taxa_media:.3f} min/operação")

resumo       = historicos.groupby('ciclo')[coluna].last()
ciclo_ref    = resumo.idxmax()
serie_treino = historicos[historicos['ciclo'] == ciclo_ref][coluna].reset_index(drop=True)

print(f"Ciclo de referência: {ciclo_ref} (desgaste final: {serie_treino.iloc[-1]} min)")

model     = ExponentialSmoothing(serie_treino, trend='add', seasonal=None)
fit_model = model.fit()

ciclos_teste     = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
maes             = []
rmses            = []
erros_max        = []
resultados_teste = []

print("\n=== Avaliação da Precisão do Modelo ===")
print(f"{'Ciclo':>6} | {'Ops':>5} | {'MAE':>8} | {'RMSE':>8} | {'Erro máx':>9}")
print("-" * 48)

for cid in ciclos_teste:
    ciclo_data = historicos[historicos['ciclo'] == cid][coluna].reset_index(drop=True)
    meio       = len(ciclo_data) // 2
    entrada    = ciclo_data.iloc[:meio]
    real       = ciclo_data.iloc[meio:].values

    forecast_r = fit_model.forecast(len(real))
    off        = entrada.iloc[-1] - forecast_r.iloc[0]
    previsao   = (forecast_r + off).values
    min_len    = min(len(previsao), len(real))

    mae  = np.mean(np.abs(real[:min_len] - previsao[:min_len]))
    rmse = np.sqrt(np.mean((real[:min_len] - previsao[:min_len])**2))
    emax = np.max(np.abs(real[:min_len] - previsao[:min_len]))

    maes.append(mae)
    rmses.append(rmse)
    erros_max.append(emax)
    resultados_teste.append((cid, len(ciclo_data), mae, rmse, emax, entrada, real, previsao))

    print(f"{cid:>6} | {len(ciclo_data):>5} | {mae:>7.2f}m | {rmse:>7.2f}m | {emax:>8.2f}m")

print("-" * 48)
print(f"{'Média':>6} | {'':>5} | {np.mean(maes):>7.2f}m | {np.mean(rmses):>7.2f}m |")

LIMITE_CRITICO = 200
desgaste_atual = ciclo_atual.iloc[-1]
op_atual       = len(ciclo_atual)
ops_restantes  = int((LIMITE_CRITICO - desgaste_atual) / taxa_media)
n_forecast     = max(ops_restantes + 20, 50)

forecast_raw      = fit_model.forecast(n_forecast)
offset            = desgaste_atual - forecast_raw.iloc[0]
forecast_ajustado = forecast_raw + offset
forecast_ajustado.index = range(op_atual, op_atual + n_forecast)

cruzamento = forecast_ajustado[forecast_ajustado >= LIMITE_CRITICO]
op_limite  = cruzamento.index[0] if len(cruzamento) > 0 else None
faltam     = op_limite - op_atual if op_limite else None

if faltam:
    print(f"\n⚠️  Previsão: ferramenta atingirá {LIMITE_CRITICO} min em ~{faltam} operações")
    print(f"   (na operação {op_limite} do ciclo atual)")
else:
    print(f"\n✅ Ferramenta não deve atingir {LIMITE_CRITICO} min nas próximas {n_forecast} operações")

period = max(5, len(serie_treino) // 10)
decomposition = seasonal_decompose(serie_treino, model='additive', period=period)
fig1 = decomposition.plot()
fig1.set_size_inches(12, 8)
plt.suptitle(f'Decomposição da Série — Ciclo de Referência ({ciclo_ref})', y=1.01)
plt.tight_layout()
plt.savefig('decomposicao.png', dpi=150, bbox_inches='tight')
plt.show()

fig2, ax = plt.subplots(figsize=(12, 6))
ax.plot(ciclo_atual.index, ciclo_atual.values,
        label='Desgaste atual (ciclo em andamento)', color='steelblue', linewidth=2)
ax.plot(forecast_ajustado.index, forecast_ajustado.values,
        label='Previsão', linestyle='--', color='orange', linewidth=2)
ax.axhline(y=LIMITE_CRITICO, color='red', linestyle=':', linewidth=2,
           label=f'Limite crítico ({LIMITE_CRITICO} min)')
if op_limite:
    ax.axvline(x=op_limite, color='red', linestyle='--', alpha=0.4)
    ax.annotate(
        f'Troca em ~{faltam} ops',
        xy=(op_limite, LIMITE_CRITICO),
        xytext=(op_limite + 2, LIMITE_CRITICO - 20),
        arrowprops=dict(arrowstyle='->', color='red'),
        color='red', fontsize=10
    )
ax.set_title('Previsão de Vida Útil da Ferramenta (Ciclo Atual)')
ax.set_xlabel('Operação dentro do ciclo')
ax.set_ylabel('Desgaste (min)')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('previsao_desgaste.png', dpi=150)
plt.show()


fig3, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()

for i, (cid, n_ops, mae, rmse, emax, entrada, real, previsao) in enumerate(resultados_teste[:4]):
    ax      = axes[i]
    x_ent   = range(len(entrada))
    x_real  = range(len(entrada), len(entrada) + len(real))
    min_len = min(len(real), len(previsao))

    ax.plot(x_ent,  entrada.values,          color='steelblue', linewidth=2, label='Entrada (histórico)')
    ax.plot(x_real, real,                    color='green',     linewidth=2, label='Real (gabarito)')
    ax.plot(x_real[:min_len], previsao[:min_len],
            color='orange', linestyle='--',  linewidth=2, label='Previsão')
    ax.axhline(y=LIMITE_CRITICO, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_title(f'Ciclo {cid}  |  MAE: {mae:.1f} min  |  RMSE: {rmse:.1f} min')
    ax.set_xlabel('Operação')
    ax.set_ylabel('Desgaste (min)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.4)

plt.suptitle('Validação do Modelo — Previsão vs Real (4 ciclos de teste)', fontsize=13)
plt.tight_layout()
plt.savefig('validacao_modelo.png', dpi=150)
plt.show()


fig4, axes = plt.subplots(1, 3, figsize=(14, 5))
ciclos_labels = [f'C{c[0]}' for c in resultados_teste]

axes[0].bar(ciclos_labels, maes, color='steelblue', alpha=0.8)
axes[0].axhline(np.mean(maes), color='red', linestyle='--', linewidth=1.5,
                label=f'Média: {np.mean(maes):.2f}')
axes[0].set_title('MAE por ciclo de teste')
axes[0].set_ylabel('Erro (min)')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.4)

axes[1].bar(ciclos_labels, rmses, color='coral', alpha=0.8)
axes[1].axhline(np.mean(rmses), color='red', linestyle='--', linewidth=1.5,
                label=f'Média: {np.mean(rmses):.2f}')
axes[1].set_title('RMSE por ciclo de teste')
axes[1].set_ylabel('Erro (min)')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.4)

axes[2].bar(ciclos_labels, erros_max, color='orange', alpha=0.8)
axes[2].axhline(np.mean(erros_max), color='red', linestyle='--', linewidth=1.5,
                label=f'Média: {np.mean(erros_max):.2f}')
axes[2].set_title('Erro máximo por ciclo de teste')
axes[2].set_ylabel('Erro (min)')
axes[2].legend()
axes[2].grid(axis='y', alpha=0.4)

plt.suptitle('Resumo das Métricas de Precisão do Modelo', fontsize=13)
plt.tight_layout()
plt.savefig('metricas_precisao.png', dpi=150)
plt.show()