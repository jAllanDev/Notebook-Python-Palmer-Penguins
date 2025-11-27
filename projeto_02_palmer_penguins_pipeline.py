# %%
"""
Projeto 02 - Esteira de Machine Learning
Base: Palmer Penguins (UCI / palmerpenguins)
Formato: script com marcações de células (compatível com Jupytext / VSCode / Colab quando convertido a .ipynb)
Instruções: salve este arquivo como .py ou converta/execute como notebook. Para gerar .ipynb com jupytext:
  pip install jupytext
  jupytext --to notebook Projeto02_Palmer_Penguins_Pipeline.py

Resumo das etapas implementadas:
- Carregamento da base (tenta seaborn -> github raw -> palmerpenguins pip)
- Estatísticas descritivas
- Transformação de colunas (engenharia de features | encoding)
- Transformação de linhas (tratamento de outliers e missing)
- Split treino/val/test (estratificado)
- Treinamento de um RandomForestClassifier
- Avaliação: matriz de confusão, acurácia, relatório
- Exemplo de predição e exportação do modelo
"""

# %%
# Dependências
# Execute: pip install -r requirements.txt
# requirements.txt suggestion:
# pandas
# numpy
# scikit-learn
# matplotlib
# seaborn
# joblib
# jupytext (opcional)

import os
import io
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import joblib

# %%
# 1) Carregar a base de dados
# Tentaremos múltiplas fontes para robustez: seaborn, raw github, pacote palmerpenguins

def load_penguins():
    # 1) seaborn
    try:
        df = sns.load_dataset('penguins')
        if df is not None and not df.empty:
            print('Carregado via seaborn.load_dataset')
            return df
    except Exception:
        pass

    # 2) tentar raw CSV hospedado (fallback)
    urls = [
        'https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv',
        'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv'
    ]
    for url in urls:
        try:
            df = pd.read_csv(url)
            print(f'Carregado via CSV raw: {url}')
            return df
        except Exception:
            pass

    # 3) última tentativa: instalar palmerpenguins (pypi)
    try:
        import palmerpenguins
        # pacote palmerpenguins não necessariamente disponibiliza df direto; mas tentamos
    except Exception:
        pass

    raise RuntimeError('Não foi possível carregar o dataset. Verifique conexão ou instale seaborn/palmerpenguins.')


penguins = load_penguins()

# Exibir primeiras linhas
print('\nPrimeiras 5 linhas:')
print(penguins.head())

# %%
# 2) Estatísticas descritivas gerais
print('\nResumo estatístico (numérico):')
print(penguins.describe(include=[np.number]))

print('\nResumo (categórico):')
print(penguins.describe(include=['object', 'category']))

# Visualização rápida
plt.figure(figsize=(8,5))
sns.pairplot(penguins.dropna(), hue='species')
plt.suptitle('Pairplot (somente linhas sem NA)')
plt.tight_layout()
plt.show()

# %%
# 3) Transformações de coluna(s)
# - Criar relacionamento: bill_ratio = bill_length_mm / bill_depth_mm
# - Tratar categorias: transformar island e sex em dummies/label encoding se necessário

df = penguins.copy()

# Coluna criada
df['bill_ratio'] = df['bill_length_mm'] / (df['bill_depth_mm'] + 1e-8)  # evitar divisão por zero

# Exemplo de transformação: agrupar species raras? (não aplicável: 3 espécies)

# Mostrar mudança
print('\nColunas após engenharia de features:')
print(df.columns.tolist())

# %%
# 4) Transformação de linhas
# - Remover duplicatas
# - Tratar outliers: remover linhas com flipper_length_mm fora de 3 desvios-padrão
# - Tratar missing: imputar valores numéricos com mediana e categóricos com moda

# 4.1 remover duplicatas
initial_len = len(df)
df = df.drop_duplicates()
print(f'Linhas removidas (duplicatas): {initial_len - len(df)}')

# 4.2 outlier removal (flipper_length_mm)
if 'flipper_length_mm' in df.columns:
    flipper = df['flipper_length_mm']
    mean = flipper.mean()
    std = flipper.std()
    cutoff = 3 * std
    mask = (flipper >= mean - cutoff) & (flipper <= mean + cutoff)
    removed = (~mask).sum()
    df = df[mask | flipper.isna()]  # manter NA pra depois imputar
    print(f'Linhas removidas por outlier (flipper_length_mm): {removed}')

# 4.3 imputação
# Numéricos -> mediana
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for c in num_cols:
    med = df[c].median()
    df[c] = df[c].fillna(med)

# Categóricos -> moda
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
for c in cat_cols:
    mode = df[c].mode()
    if not mode.empty:
        df[c] = df[c].fillna(mode[0])

print('\nApós tratamento de linhas:')
print(df.info())

# %%
# 5) Preparar features e target
# Vamos prever 'species' (Adelie, Chinstrap, Gentoo)

target = 'species'
features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g', 'bill_ratio', 'island', 'sex']

# Encoding: label encode target, one-hot for island and sex
le = LabelEncoder()
df['species_label'] = le.fit_transform(df[target])

# One-hot for island and sex
df_enc = pd.get_dummies(df[features], drop_first=True)

X = df_enc
y = df['species_label']

print('\nFeature matrix shape:', X.shape)
print('Target distribution:')
print(y.value_counts())

# %%
# 6) Criar splits: train / val / test (60/20/20) - estratificado
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
relative_val_size = 0.25  # 0.25 * 0.8 = 0.2 => total val 20%
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=relative_val_size, stratify=y_temp, random_state=42)

print('\nTamanhos:')
print('Treino:', X_train.shape)
print('Validação:', X_val.shape)
print('Teste:', X_test.shape)

# %%
# 7) Escalonamento (aplicado apenas às colunas numéricas dentro X)
numeric_features = [c for c in X.columns if any(n in c for n in ['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g','bill_ratio'])]
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_val[numeric_features] = scaler.transform(X_val[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# %%
# 8) Treinamento do modelo
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(X_train, y_train)

# %%
# 9) Avaliação

y_pred_val = clf.predict(X_val)
acc_val = accuracy_score(y_val, y_pred_val)
cm_val = confusion_matrix(y_val, y_pred_val)
print(f'Accuracy (val): {acc_val:.4f}')
print('\nConfusion Matrix (val):')
print(cm_val)
print('\nClassification Report (val):')
print(classification_report(y_val, y_pred_val, target_names=le.classes_))

# Mostrar matriz de confusão plotada
plt.figure(figsize=(6,5))
sns.heatmap(cm_val, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel('Predito')
plt.ylabel('Verdadeiro')
plt.title('Matriz de Confusão - Val')
plt.show()

# Avaliar no teste
y_pred_test = clf.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
cm_test = confusion_matrix(y_test, y_pred_test)
print(f'Accuracy (test): {acc_test:.4f}')
print('\nConfusion Matrix (test):')
print(cm_test)

# %%
# 10) Exemplo de predição usando um novo registro (usar média/valores plausíveis)
sample = X_test.iloc[0:1].copy()
print('\nAmostra de features (antes inverse transform):')
print(sample)

pred_label = clf.predict(sample)[0]
pred_species = le.inverse_transform([pred_label])[0]
print(f'Predição de exemplo (classe): {pred_species}')

# Mostrar probabilidade
probs = clf.predict_proba(sample)[0]
prob_df = pd.DataFrame({'species': le.classes_, 'prob': probs})
print('\nProbabilidades por espécie:')
print(prob_df.sort_values('prob', ascending=False))

# %%
# 11) Salvar modelo e artefatos
os.makedirs('artifacts', exist_ok=True)
joblib.dump(clf, 'artifacts/rf_penguins.joblib')
joblib.dump(scaler, 'artifacts/scaler.joblib')
joblib.dump(le, 'artifacts/label_encoder.joblib')

print('\nModelos e artefatos salvos em ./artifacts')

# %%
# 12) Observações finais e instruções para reproduzir
print('\nObservações:')
print('- Este notebook converte facilmente para .ipynb com jupytext.')
print('- Para acelerar o vídeo, execute os blocos de treinamento com menos árvores (ex: n_estimators=50) e depois volte ao valor final ao commitar o código.')
print('- README.md deve incluir passos: instalar dependências, converter se necessário, rodar notebook, links para vídeo e github.')

# Fim
