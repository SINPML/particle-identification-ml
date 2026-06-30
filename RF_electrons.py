import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('clas12_electron_id_features.csv')

print("=" * 50)
print("Исходная информация о датасете")
print("=" * 50)
print(f"Размер датасета: {df.shape}")
print(f"\nЦелевая переменная распределение:")
print(df['is_electron_mc'].value_counts())
print(f"Доля электронов: {df['is_electron_mc'].mean()*100:.2f}%")
print(f"Доля не-электронов: {(1-df['is_electron_mc'].mean())*100:.2f}%")

exclude_cols = ['is_electron_mc', 'mc_pid','rec_pid']
mc_fields = [col for col in df.columns if col.startswith('mc_')]
exclude_cols.extend(mc_fields)

feature_cols = [col for col in df.columns if col not in exclude_cols]


if 'p' not in df.columns:
    df['p'] = np.sqrt(df['px']**2 + df['py']**2 + df['pz']**2)
    print("\nСоздана переменная p")

if 'pt' not in df.columns:
    df['pt'] = np.sqrt(df['px']**2 + df['py']**2)
    print("Создана переменная pt")

if 'theta' not in df.columns:
    df['theta'] = np.arccos(df['pz'] / df['p']) * 180 / np.pi
    print("Создана переменная theta")

if 'phi' not in df.columns:
    df['phi'] = np.arctan2(df['py'], df['px']) * 180 / np.pi
    print("Создана переменная phi")

if 'E_over_P' not in df.columns and 'calo_energy' in df.columns:
    df['E_over_P'] = df['calo_energy'] / df['p']
    df['E_over_P'] = df['E_over_P'].replace([np.inf, -np.inf], 0).fillna(0)
    print("Создана переменная E_over_P")

if 'scint_beta' not in df.columns and 'scint_path' in df.columns and 'scint_time' in df.columns:
    df['scint_beta'] = df['scint_path'] / (df['scint_time'] * 29.979)
    df['scint_beta'] = df['scint_beta'].replace([np.inf, -np.inf], 0).fillna(0)
    print("Создана переменная scint_beta")

if 'track_chi2_per_NDF' not in df.columns and 'track_chi2' in df.columns and 'track_NDF' in df.columns:
    df['track_chi2_per_NDF'] = df['track_chi2'] / df['track_NDF']
    df['track_chi2_per_NDF'] = df['track_chi2_per_NDF'].replace([np.inf, -np.inf], 0).fillna(0)
    print("Создана переменная track_chi2_per_NDF")

if 'electron_score_basic' not in df.columns and 'E_over_P' in df.columns and 'htcc_nphe' in df.columns:
    df['electron_score_basic'] = df['E_over_P'] * np.log(1 + df['htcc_nphe'])
    df['electron_score_basic'] = df['electron_score_basic'].replace([np.inf, -np.inf], 0).fillna(0)
    print("Создана переменная electron_score_basic")

feature_cols = [col for col in df.columns if col not in exclude_cols]


df_clean = df.dropna(subset=feature_cols + ['is_electron_mc'])
print(f"\nПосле удаления NaN: {df_clean.shape}")
print(f"Электронов было: {df['is_electron_mc'].sum()}, стало: {df_clean['is_electron_mc'].sum()}")
print(f"Потеряно электронов: {df['is_electron_mc'].sum() - df_clean['is_electron_mc'].sum()}")
print(f"Потеряно не-электронов: {(1-df['is_electron_mc']).sum() - (1-df_clean['is_electron_mc']).sum()}")


X = df_clean[feature_cols]
y = df_clean['is_electron_mc']

X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)

# Масштабирование (опционально для Random Forest, но полезно для корреляций)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)

print(f"\nРазмер обучающей выборки: {X_train_scaled.shape}")
print(f"Размер тестовой выборки: {X_test_scaled.shape}")

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)






train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(
        n_estimators=100,      
        max_depth=15,          
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    ),
    X_train, y_train,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_scores.mean(axis=1), 'o-', label='Train')
plt.plot(train_sizes, test_scores.mean(axis=1), 'o-', label='Validation')
plt.xlabel('Размер выборки')
plt.ylabel('Точность')
plt.title('Кривые обучения Random Forest')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('learning_curve.png', dpi=150)
plt.show()

print(f"Итоговая точность на валидации: {test_scores.mean(axis=1)[-1]:.3f}")
print("=" * 50)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_pred_proba = rf.predict_proba(X_test)[:, 1]

print("\n" + "=" * 50)
print("Оценка качества модели")
print("=" * 50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Не-электрон', 'Электрон']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

print("\n" + "=" * 50)
print("Создание большой матрицы корреляций")
print("=" * 50)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

top_features = feature_importance.head(30)['feature'].tolist()
print(f"\nТоп-30 наиболее важных признаков:")
for i, (feat, imp) in enumerate(zip(feature_importance.head(30)['feature'], 
                                     feature_importance.head(30)['importance']), 1):
    print(f"{i:2d}. {feat:<30} {imp:.4f}")


fig, ax = plt.subplots(figsize=(20, 18))


correlation_with_target = X_train_scaled[top_features].corrwith(y_train).sort_values(ascending=False)
print("\nКорреляция признаков с целевой переменной (на обучающей выборке):")
print(correlation_with_target.round(3))

corr_matrix = X_train_scaled[top_features].corr()


mask = np.triu(np.ones_like(corr_matrix, dtype=bool))


sns.heatmap(corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8, "label": "Корреляция"},
            annot_kws={'size': 8})

plt.title('Матрица корреляций топ-30 признаков\n(можно скроллить при сохранении)', fontsize=16, pad=20)
plt.xticks(rotation=90, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('correlation_matrix_large.png', dpi=150, bbox_inches='tight')
print("\n✓ Матрица корреляций сохранена как 'correlation_matrix_large.png'")
plt.show()

print("\n" + "=" * 50)
print("Визуализация одного дерева из Random Forest")
print("=" * 50)


tree_to_plot = rf.estimators_[0]

plt.figure(figsize=(30, 20))

plot_tree(tree_to_plot,
          max_depth=4,
          feature_names=feature_cols,
          class_names=['Не-электрон', 'Электрон'],
          filled=True,
          rounded=True,
          fontsize=8,
          proportion=True)

plt.title('Визуализация одного дерева из Random Forest (глубина 4)', 
          fontsize=16, pad=20)
plt.tight_layout()
plt.savefig('decision_tree_visualization.png', dpi=150, bbox_inches='tight')
print("✓ Визуализация дерева сохранена как 'decision_tree_visualization.png'")
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Важность признаков (топ-20)
importance_sorted = feature_importance.head(20)
axes[0, 0].barh(range(len(importance_sorted)), importance_sorted['importance'], color='steelblue')
axes[0, 0].set_yticks(range(len(importance_sorted)))
axes[0, 0].set_yticklabels(importance_sorted['feature'])
axes[0, 0].set_xlabel('Важность')
axes[0, 0].set_title('Топ-20 наиболее важных признаков')
axes[0, 0].invert_yaxis()

# Confusion Matrix (нормализованная)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', 
            xticklabels=['Не-электрон', 'Электрон'],
            yticklabels=['Не-электрон', 'Электрон'],
            ax=axes[1, 0])
axes[1, 0].set_xlabel('Predicted')
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_title('Normalized Confusion Matrix')

# Распределение предсказанных вероятностей
axes[1, 1].hist(y_pred_proba[y_test == 0], bins=50, alpha=0.6, label='Не-электроны', color='red')
axes[1, 1].hist(y_pred_proba[y_test == 1], bins=50, alpha=0.6, label='Электроны', color='blue')
axes[1, 1].set_xlabel('Предсказанная вероятность быть электроном')
axes[1, 1].set_ylabel('Частота')
axes[1, 1].set_title('Распределение предсказанных вероятностей')
axes[1, 1].legend()
axes[1, 1].axvline(x=0.5, color='black', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('additional_plots.png', dpi=150, bbox_inches='tight')
print("✓ Дополнительные графики сохранены как 'additional_plots.png'")
plt.show()

# 4. АНАЛИЗ ОШИБОК
print("\n" + "=" * 50)
print("Анализ ошибок модели")
print("=" * 50)

# Находим ошибочно классифицированные примеры
misclassified_idx = np.where(y_test != y_pred)[0]
print(f"\nКоличество ошибочно классифицированных примеров: {len(misclassified_idx)}")
print(f"Доля ошибок: {len(misclassified_idx)/len(y_test)*100:.2f}%")

# Анализ важности признаков для правильных и неправильных предсказаний
print("\nВажность признаков (топ-10):")
for i, (feat, imp) in enumerate(zip(feature_importance.head(10)['feature'], 
                                     feature_importance.head(10)['importance']), 1):
    print(f"{i:2d}. {feat:<30} {imp:.4f}")

# Сохраняем модель
import joblib
joblib.dump(rf, 'random_forest_electron_model.pkl')
print("\n✓ Модель сохранена как 'random_forest_electron_model.pkl'")

print("\n" + "=" * 50)
print("Все визуализации сохранены:")
print("1. correlation_matrix_large.png - большая матрица корреляций")
print("2. decision_tree_visualization.png - визуализация дерева")
print("3. additional_plots.png - дополнительные графики")
print("4. random_forest_electron_model.pkl - сохраненная модель")
print("=" * 50)