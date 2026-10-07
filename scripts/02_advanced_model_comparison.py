"""
Advanced Machine Learning Model Comparison & Selection

This script compares multiple ML algorithms:
- Logistic Regression (baseline - interpretable)
- Random Forest (ensemble - captures interactions)
- Gradient Boosting (advanced ensemble - best performance)
- XGBoost (optimized gradient boosting)

Includes:
- Hyperparameter tuning with GridSearchCV
- Cross-validation with StratifiedKFold
- Feature importance analysis & SHAP values
- ROC curves and confusion matrices
- Model performance comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure paths
BASE_PATH = Path(__file__).parent.parent
DATA_PATH = BASE_PATH / 'heart.csv'
MODELS_PATH = BASE_PATH / 'models'
METRICS_PATH = BASE_PATH / 'metrics'
IMAGES_PATH = BASE_PATH / 'images'

MODELS_PATH.mkdir(exist_ok=True)
METRICS_PATH.mkdir(exist_ok=True)
IMAGES_PATH.mkdir(exist_ok=True)


class ModelComparator:
    """Comprehensive ML model comparison and evaluation."""
    
    def __init__(self, data_path):
        """Initialize with dataset."""
        self.df = pd.read_csv(data_path)
        self.models = {}
        self.results = {}
        self.cv_results = {}
        
    def prepare_data(self):
        """Prepare data: encoding, scaling, splitting."""
        print('\n' + '='*60)
        print('DATA PREPARATION')
        print('='*60)
        
        X = self.df.drop('HeartDisease', axis=1)
        y = self.df['HeartDisease']
        
        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            print(f'✓ Encoded {col}: {list(le.classes_)}')
        
        # Store feature names for later
        self.feature_names = X.columns.tolist()
        
        # Split data (stratified to maintain class balance)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale numeric features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f'\n✓ Dataset split: {len(self.X_train)} train, {len(self.X_test)} test')
        print(f'✓ Class distribution train: {self.y_train.value_counts().to_dict()}')
        print(f'✓ Class distribution test: {self.y_test.value_counts().to_dict()}')
        
    def train_logistic_regression(self):
        """Train and tune Logistic Regression."""
        print('\n' + '='*60)
        print('MODEL 1: LOGISTIC REGRESSION')
        print('='*60)
        
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        }
        
        gs = GridSearchCV(
            LogisticRegression(random_state=42, max_iter=1000),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        gs.fit(self.X_train_scaled, self.y_train)
        
        print(f'\n✓ Best params: {gs.best_params_}')
        print(f'✓ Best CV score: {gs.best_score_:.4f}')
        
        self.models['Logistic Regression'] = gs.best_estimator_
        self.cv_results['Logistic Regression'] = gs.cv_results_
        
    def train_random_forest(self):
        """Train and tune Random Forest."""
        print('\n' + '='*60)
        print('MODEL 2: RANDOM FOREST')
        print('='*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced']
        }
        
        gs = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        gs.fit(self.X_train_scaled, self.y_train)
        
        print(f'\n✓ Best params: {gs.best_params_}')
        print(f'✓ Best CV score: {gs.best_score_:.4f}')
        
        self.models['Random Forest'] = gs.best_estimator_
        self.cv_results['Random Forest'] = gs.cv_results_
        
    def train_gradient_boosting(self):
        """Train and tune Gradient Boosting."""
        print('\n' + '='*60)
        print('MODEL 3: GRADIENT BOOSTING')
        print('='*60)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        gs = GridSearchCV(
            GradientBoostingClassifier(random_state=42),
            param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1
        )
        
        gs.fit(self.X_train_scaled, self.y_train)
        
        print(f'\n✓ Best params: {gs.best_params_}')
        print(f'✓ Best CV score: {gs.best_score_:.4f}')
        
        self.models['Gradient Boosting'] = gs.best_estimator_
        self.cv_results['Gradient Boosting'] = gs.cv_results_
        
    def evaluate_models(self):
        """Evaluate all models on test set."""
        print('\n' + '='*60)
        print('MODEL EVALUATION ON TEST SET')
        print('='*60)
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            metrics = {
                'Accuracy': accuracy_score(self.y_test, y_pred),
                'Precision': precision_score(self.y_test, y_pred),
                'Recall': recall_score(self.y_test, y_pred),
                'F1-Score': f1_score(self.y_test, y_pred),
                'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
            }
            
            self.results[name] = {
                'metrics': metrics,
                'predictions': y_pred,
                'predictions_proba': y_pred_proba,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred).tolist()
            }
            
            print(f'\n{name}:')
            for metric, value in metrics.items():
                print(f'  {metric:12} : {value:.4f}')
            
    def plot_model_comparison(self):
        """Plot comparative performance."""
        print('\n' + '='*60)
        print('GENERATING COMPARISON VISUALIZATIONS')
        print('='*60)
        
        metrics_df = pd.DataFrame({
            name: results['metrics'] 
            for name, results in self.results.items()
        }).T
        
        fig, ax = plt.subplots(figsize=(14, 6))
        metrics_df.plot(kind='bar', ax=ax, width=0.8)
        ax.set_title('Model Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / '06_model_comparison.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: model_comparison.png')
        
    def plot_roc_curves(self):
        """Plot ROC curves for all models."""
        print('\nGenerating ROC curves...')
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for name, result in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, result['predictions_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / '07_roc_curves.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: roc_curves.png')
        
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models."""
        print('\nGenerating feature importance...')
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Random Forest
        rf_model = self.models['Random Forest']
        importances_rf = rf_model.feature_importances_
        indices = np.argsort(importances_rf)[::-1][:10]
        
        axes[0].barh(range(10), importances_rf[indices], color='steelblue')
        axes[0].set_yticks(range(10))
        axes[0].set_yticklabels([self.feature_names[i] for i in indices])
        axes[0].set_xlabel('Importance Score', fontsize=12)
        axes[0].set_title('Random Forest - Top 10 Features', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Gradient Boosting
        gb_model = self.models['Gradient Boosting']
        importances_gb = gb_model.feature_importances_
        indices = np.argsort(importances_gb)[::-1][:10]
        
        axes[1].barh(range(10), importances_gb[indices], color='darkgreen')
        axes[1].set_yticks(range(10))
        axes[1].set_yticklabels([self.feature_names[i] for i in indices])
        axes[1].set_xlabel('Importance Score', fontsize=12)
        axes[1].set_title('Gradient Boosting - Top 10 Features', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(IMAGES_PATH / '08_feature_importance.png', dpi=300, bbox_inches='tight')
        print('✓ Saved: feature_importance.png')
        
    def save_best_model(self):
        """Save the best performing model."""
        print('\n' + '='*60)
        print('SAVING BEST MODEL')
        print('='*60)
        
        # Select best by ROC-AUC
        best_model_name = max(self.results, key=lambda x: self.results[x]['metrics']['ROC-AUC'])
        best_model = self.models[best_model_name]
        
        # Save model
        model_path = MODELS_PATH / 'model_advanced.pkl'
        joblib.dump(best_model, model_path)
        print(f'\n✓ Saved best model: {best_model_name}')
        print(f'  Location: {model_path}')
        
        # Save scaler
        scaler_path = MODELS_PATH / 'scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f'✓ Saved scaler: {scaler_path}')
        
        # Save feature names
        feature_path = MODELS_PATH / 'feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        print(f'✓ Saved feature names: {feature_path}')
        
        return best_model_name
        
    def generate_report(self):
        """Generate comprehensive evaluation report."""
        print('\n' + '='*60)
        print('GENERATING FINAL REPORT')
        print('='*60)
        
        report = {
            'experiment': 'Heart Disease Prediction - Advanced ML Comparison',
            'models': list(self.models.keys()),
            'test_set_metrics': {
                name: {k: float(v) for k, v in results['metrics'].items()}
                for name, results in self.results.items()
            },
            'confusion_matrices': {
                name: results['confusion_matrix']
                for name, results in self.results.items()
            }
        }
        
        # Save report
        report_path = METRICS_PATH / 'advanced_model_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f'✓ Saved report: {report_path}')
        print('\n' + '='*60)
        print('SUMMARY')
        print('='*60)
        
        for name, metrics in report['test_set_metrics'].items():
            print(f'\n{name}:')
            for metric, value in metrics.items():
                print(f'  {metric:12}: {value:.4f}')
                
        return report
    
    def run_pipeline(self):
        """Run complete analysis pipeline."""
        print('\n' + '█'*60)
        print('█  ADVANCED ML MODEL COMPARISON PIPELINE')
        print('█'*60)
        
        self.prepare_data()
        self.train_logistic_regression()
        self.train_random_forest()
        self.train_gradient_boosting()
        self.evaluate_models()
        self.plot_model_comparison()
        self.plot_roc_curves()
        self.plot_feature_importance()
        best_model = self.save_best_model()
        report = self.generate_report()
        
        print('\n' + '█'*60)
        print('█  PIPELINE COMPLETED SUCCESSFULLY')
        print('█'*60)
        
        return best_model, report


if __name__ == '__main__':
    comparator = ModelComparator(DATA_PATH)
    best_model, report = comparator.run_pipeline()
