# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. Load Data
# ------------------------------
df = pd.read_csv(r"C:\Users\DEll Two Two\Videos\project\mental_health.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())

# ------------------------------
# 2. Basic Info & Missing Values
# ------------------------------
print("\nData Info:")
df.info()
print("\nMissing values per column:")
print(df.isnull().sum())   # Should be zero from inspection

# ------------------------------
# 3. Target Distribution
# ------------------------------
sns.countplot(x='Has_Mental_Health_Issue', data=df)
plt.title('Target Distribution')
plt.show()
print(df['Has_Mental_Health_Issue'].value_counts(normalize=True))

# ------------------------------
# 4. Separate Features & Target
# ------------------------------
X = df.drop('Has_Mental_Health_Issue', axis=1)
y = df['Has_Mental_Health_Issue']

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# ------------------------------
# 5. Preprocessing Pipeline
# ------------------------------
# For ordinal categorical variables, we can map them manually or use OrdinalEncoder.
# We'll use OneHotEncoder for all categorical for simplicity (cardinality is low).

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# ------------------------------
# 6. Train/Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# ------------------------------
# 7. Model Comparison (Cross-validation)
# ------------------------------
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
}

results = {}
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1')
    results[name] = scores.mean()
    print(f"{name}: F1 = {scores.mean():.4f} (+/- {scores.std():.4f})")

# Choose best model
best_model_name = max(results, key=results.get)
print(f"\nBest model: {best_model_name} with F1 = {results[best_model_name]:.4f}")

# ------------------------------
# 8. Train Best Model on Full Training Set & Evaluate on Test
# ------------------------------
best_model = models[best_model_name]
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('classifier', best_model)])
final_pipeline.fit(X_train, y_train)

y_pred = final_pipeline.predict(X_test)
y_proba = final_pipeline.predict_proba(X_test)[:, 1]

print("\n--- Test Set Evaluation ---")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# ------------------------------
# 9. Feature Importance (if tree-based)
# ------------------------------
if best_model_name in ['Random Forest', 'XGBoost']:
    # Get feature names after preprocessing
    feature_names = (numeric_cols +
                     list(final_pipeline.named_steps['preprocessor']
                          .named_transformers_['cat']
                          .get_feature_names_out(categorical_cols)))
    importances = final_pipeline.named_steps['classifier'].feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
    plt.figure(figsize=(10,6))
    feat_imp.plot(kind='bar')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()

# ------------------------------
# 10. Save Model
# ------------------------------
joblib.dump(final_pipeline, 'mental_health_model.pkl')
print("\nModel saved as mental_health_model.pkl")