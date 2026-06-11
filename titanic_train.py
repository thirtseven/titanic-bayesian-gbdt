import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Titanic Survival Prediction - Bayesian Optimized GBDT")
print("="*60)

# Load data
print("\n[1/6] Loading data...")
train_data = pd.read_csv('/kaggle/input/competitions/titanic/train.csv')
test_data = pd.read_csv('/kaggle/input/competitions/titanic/test.csv')
print(f"Train shape: {train_data.shape}")
print(f"Test shape: {test_data.shape}")

# Feature engineering
print("\n[2/6] Feature engineering...")
def create_features(df):
    data = df.copy()
    
    # Fill missing values
    data['Age'] = data['Age'].fillna(data['Age'].median())
    data['Fare'] = data['Fare'].fillna(data['Fare'].median())
    data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
    
    # Encode categorical
    data['Sex'] = (data['Sex'] == 'male').astype(int)
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Family features
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    
    # Age bins
    data['AgeBin'] = pd.cut(data['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0,1,2,3,4])
    data['AgeBin'] = data['AgeBin'].astype(int)
    
    # Fare bins
    data['FareBin'] = pd.qcut(data['Fare'], q=4, labels=[0,1,2,3])
    data['FareBin'] = data['FareBin'].astype(int)
    
    # Title extraction
    if 'Name' in data.columns:
        data['Title'] = data['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {
            'Mr': 0, 'Miss': 1, 'Mrs': 2, 'Master': 3,
            'Dr': 4, 'Rev': 4, 'Col': 4, 'Major': 4, 'Mlle': 1,
            'Ms': 1, 'Mme': 2, 'Don': 4, 'Dona': 4, 'Lady': 4,
            'Countess': 4, 'Jonkheer': 4, 'Sir': 4, 'Capt': 4
        }
        data['Title'] = data['Title'].map(title_map).fillna(4)
    
    return data

train_df = create_features(train_data)
test_df = create_features(test_data)

feature_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
                'Embarked', 'FamilySize', 'IsAlone', 'AgeBin', 'FareBin', 'Title']

X = train_df[feature_cols]
y = train_df['Survived']
X_test_final = test_df[feature_cols]

print(f"Features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")

# Split data
print("\n[3/6] Splitting data...")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# Train model with Bayesian optimized parameters
print("\n[4/6] Training GradientBoosting model...")
print("Parameters:")
print("  - n_estimators: 209")
print("  - max_depth: 3")
print("  - learning_rate: 0.0478")
print("  - subsample: 0.6")

model = GradientBoostingClassifier(
    n_estimators=209,
    max_depth=3,
    learning_rate=0.047763382446662,
    subsample=0.6,
    random_state=42,
    verbose=1
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n[5/6] Evaluating model...")
y_train_pred = model.predict(X_train_scaled)
y_val_pred = model.predict(X_val_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
val_acc = accuracy_score(y_val, y_val_pred)

print(f"Train Accuracy: {train_acc:.4f}")
print(f"Validation Accuracy: {val_acc:.4f}")

# Cross validation
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\nClassification Report (Validation):")
print(classification_report(y_val, y_val_pred, target_names=['Died', 'Survived']))

# Feature importance
print("Feature Importance:")
importances = model.feature_importances_
for feat, imp in sorted(zip(feature_cols, importances), key=lambda x: -x[1]):
    print(f"  {feat}: {imp:.4f}")

# Generate predictions
print("\n[6/6] Generating predictions...")
predictions = model.predict(X_test_scaled)

# Create submission
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
submission.to_csv('/kaggle/working/submission.csv', index=False)

# Save model
joblib.dump(model, '/kaggle/working/titanic_model.pkl')
joblib.dump(scaler, '/kaggle/working/titanic_scaler.pkl')
joblib.dump(feature_cols, '/kaggle/working/titanic_features.pkl')

print(f"\nSubmission saved: /kaggle/working/submission.csv")
print(f"Model saved: /kaggle/working/titanic_model.pkl")
print(f"Samples: {len(submission)}")
print(f"Predicted survival rate: {predictions.mean():.2%}")
print("\nPreview:")
print(submission.head(10))
print("\n" + "="*60)
print("Done!")
print("="*60)
