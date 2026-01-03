# -*- coding: utf-8 -*-
"""
Pipeline de Machine Learning para Previsao de Obesidade
"""

import pandas as pd
import numpy as np
import sys
import io
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configurar encoding para evitar erros
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("="*80)
print("PIPELINE DE MACHINE LEARNING - PREVISAO DE OBESIDADE")
print("="*80)

# 1. CARREGAMENTO DOS DADOS
print("\n[1/7] Carregando dados...")
df = pd.read_csv('Obesity.csv')
print(f"[OK] Dataset carregado: {df.shape[0]} registros, {df.shape[1]} features")

# 2. ANALISE EXPLORATORIA
print("\n[2/7] Analise Exploratoria dos Dados...")
print("\n--- Informacoes Gerais ---")
print(df.info())
print("\n--- Primeiras linhas ---")
print(df.head())
print("\n--- Estatisticas descritivas ---")
print(df.describe())
print("\n--- Valores nulos ---")
print(df.isnull().sum())
print("\n--- Distribuicao da variavel alvo (Obesity) ---")
print(df['Obesity'].value_counts())

# Renomear coluna alvo para facilitar
df.rename(columns={'Obesity': 'Obesity_level'}, inplace=True)

# 3. FEATURE ENGINEERING
print("\n[3/7] Feature Engineering...")

# Criar copia para transformacoes
df_processed = df.copy()

# 3.1 Criacao de novas features
print("   -> Criando novas features...")

# IMC (Indice de Massa Corporal)
df_processed['BMI'] = df_processed['Weight'] / (df_processed['Height'] ** 2)

# Relacao peso/idade
df_processed['Weight_Age_Ratio'] = df_processed['Weight'] / df_processed['Age']

# Categoria de idade
df_processed['Age_Category'] = pd.cut(df_processed['Age'],
                                       bins=[0, 25, 35, 50, 100],
                                       labels=['Young', 'Adult', 'Middle_Age', 'Senior'])

# Score de habitos saudaveis (quanto maior, mais saudavel)
df_processed['Healthy_Score'] = (
    df_processed['FCVC'] +  # Consumo de vegetais
    df_processed['CH2O'] +  # Consumo de agua
    df_processed['FAF'] -   # Atividade fisica
    (df_processed['TUE'] / 2)  # Menos tempo em tecnologia
)

# Score de risco (quanto maior, mais risco)
df_processed['Risk_Score'] = (
    df_processed['FAVC'].map({'yes': 1, 'no': 0}) +  # Comida calorica
    df_processed['family_history'].map({'yes': 1, 'no': 0}) +  # Historico familiar
    (df_processed['NCP'] > 3).astype(int) +  # Muitas refeicoes
    df_processed['SMOKE'].map({'yes': 1, 'no': 0})  # Fumar
)

print(f"[OK] Features criadas: BMI, Weight_Age_Ratio, Age_Category, Healthy_Score, Risk_Score")

# 3.2 Encoding de variaveis categoricas
print("   -> Codificando variaveis categoricas...")

label_encoders = {}
categorical_columns = ['Gender', 'family_history', 'FAVC', 'CAEC', 'SMOKE',
                       'SCC', 'CALC', 'MTRANS', 'Age_Category']

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[col] = le.fit_transform(df_processed[col])
    label_encoders[col] = le

print(f"[OK] {len(categorical_columns)} variaveis categoricas codificadas")

# 3.3 Separacao de features e target
X = df_processed.drop('Obesity_level', axis=1)
y = df_processed['Obesity_level']

# Encoding do target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

print(f"[OK] Target encoding: {len(le_target.classes_)} classes")
print(f"   Classes: {list(le_target.classes_)}")

# 3.4 Normalizacao
print("   -> Normalizando features numericas...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"[OK] Features normalizadas: {X_scaled.shape[1]} features")

# 4. DIVISAO TREINO/TESTE
print("\n[4/7] Dividindo dados em treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"[OK] Treino: {X_train.shape[0]} amostras | Teste: {X_test.shape[0]} amostras")

# 5. TREINAMENTO DE MODELOS
print("\n[5/7] Treinamento e Avaliacao de Modelos...")

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\n   Treinando {name}...")

    # Treinar modelo
    model.fit(X_train, y_train)

    # Predicoes
    y_pred = model.predict(X_test)

    # Metricas
    accuracy = accuracy_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'predictions': y_pred
    }

    print(f"   [OK] Acuracia no teste: {accuracy:.4f}")
    print(f"   [OK] Cross-validation: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 6. SELECAO DO MELHOR MODELO
print("\n[6/7] Selecao do Melhor Modelo...")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']
best_accuracy = results[best_model_name]['accuracy']

print(f"\n{'='*80}")
print(f"*** MELHOR MODELO: {best_model_name}")
print(f"*** Acuracia: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"{'='*80}")

# Relatorio de classificacao detalhado
y_pred_best = results[best_model_name]['predictions']
print("\n--- Relatorio de Classificacao ---")
print(classification_report(y_test, y_pred_best,
                          target_names=le_target.classes_,
                          zero_division=0))

# Matriz de confusao
print("\n--- Matriz de Confusao ---")
cm = confusion_matrix(y_test, y_pred_best)
print(cm)

# 7. SALVAMENTO DOS ARTEFATOS
print("\n[7/7] Salvando artefatos do modelo...")

# Salvar modelo treinado
joblib.dump(best_model, 'modelo_obesidade.pkl')
print("[OK] Modelo salvo: modelo_obesidade.pkl")

# Salvar encoders e scaler
joblib.dump(label_encoders, 'label_encoders.pkl')
print("[OK] Label encoders salvos: label_encoders.pkl")

joblib.dump(le_target, 'target_encoder.pkl')
print("[OK] Target encoder salvo: target_encoder.pkl")

joblib.dump(scaler, 'scaler.pkl')
print("[OK] Scaler salvo: scaler.pkl")

# Salvar features para referencia
feature_names = X.columns.tolist()
joblib.dump(feature_names, 'feature_names.pkl')
print("[OK] Feature names salvas: feature_names.pkl")

# Salvar dados processados para o dashboard
df_processed['Obesity_level'] = y
df_processed.to_csv('dados_processados.csv', index=False)
print("[OK] Dados processados salvos: dados_processados.csv")

# Feature Importance (se disponivel)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\n--- Top 10 Features Mais Importantes ---")
    print(feature_importance.head(10))

    feature_importance.to_csv('feature_importance.csv', index=False)
    print("[OK] Feature importance salva: feature_importance.csv")

print("\n" + "="*80)
print("*** PIPELINE CONCLUIDA COM SUCESSO!")
print(f"*** Modelo final: {best_model_name} com {best_accuracy*100:.2f}% de acuracia")
print("*** Todos os artefatos foram salvos e estao prontos para deploy")
print("="*80)

# Resumo final
print("\n*** RESUMO DOS MODELOS:")
print("-" * 80)
for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{name:25s} | Acuracia: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)")
print("-" * 80)
