"""
12_retrospective_analysis.py

Retrospective analysis module for ICU patient monitoring.
Provides comprehensive analysis of patient trajectories, model performance,
and clinical outcomes for research and quality improvement.

Features:
- Patient trajectory analysis and visualization
- Clinical outcome prediction and validation
- Model performance analysis across different patient cohorts
- Risk factor identification and analysis
- Clinical decision support insights
- Export capabilities for research and reporting

Usage:
    python 12_retrospective_analysis.py --analyze --patient_id 12345
    python 12_retrospective_analysis.py --cohort_analysis --cohort "post_surgical"
    python 12_retrospective_analysis.py --generate_report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.calibration import calibration_curve
import config
import utils
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RetrospectiveAnalyzer:
    """Retrospective analysis for ICU patient monitoring"""
    
    def __init__(self):
        self.patient_data = None
        self.models = {}
        self.load_data()
        self.load_models()
    
    def load_data(self):
        """Load patient data for analysis"""
        try:
            if config.FEATURES_FILE.exists():
                self.patient_data = pd.read_parquet(config.FEATURES_FILE)
                logger.info(f"Loaded patient data: {len(self.patient_data)} records")
            else:
                logger.warning("Features file not found")
                self.patient_data = pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            self.patient_data = pd.DataFrame()
    
    def load_models(self):
        """Load trained models for analysis"""
        try:
            # Load scaler
            scaler_path = config.MODELS_DIR / "scaler.pkl"
            if scaler_path.exists():
                self.models['scaler'] = joblib.load(scaler_path)
            
            # Load models
            model_files = {
                'rf': config.RANDOM_FOREST_MODEL,
                'xgboost': config.XGBOOST_MODEL,
                'deep': config.DEEP_MODEL
            }
            
            for name, path in model_files.items():
                if path.exists():
                    if name == 'deep':
                        import tensorflow as tf
                        self.models[name] = tf.keras.models.load_model(path)
                    else:
                        self.models[name] = joblib.load(path)
                    logger.info(f"Loaded {name} model")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
    
    def analyze_patient_trajectory(self, patient_id: str) -> Dict:
        """Analyze individual patient trajectory"""
        
        if self.patient_data.empty or 'subject_id' not in self.patient_data.columns:
            return {"error": "No patient data available"}
        
        patient_df = self.patient_data[self.patient_data['subject_id'] == patient_id].copy()
        if patient_df.empty:
            return {"error": f"Patient {patient_id} not found"}
        
        # Sort by time
        if 'window_start' in patient_df.columns:
            patient_df = patient_df.sort_values('window_start')
        
        analysis = {
            'patient_id': patient_id,
            'total_records': len(patient_df),
            'time_span': self._calculate_time_span(patient_df),
            'vital_trends': self._analyze_vital_trends(patient_df),
            'risk_progression': self._analyze_risk_progression(patient_df),
            'clinical_events': self._identify_clinical_events(patient_df),
            'outcome_prediction': self._predict_outcome(patient_df)
        }
        
        return analysis
    
    def _calculate_time_span(self, patient_df: pd.DataFrame) -> Dict:
        """Calculate time span of patient data"""
        if 'window_start' in patient_df.columns:
            start_time = pd.to_datetime(patient_df['window_start']).min()
            end_time = pd.to_datetime(patient_df['window_start']).max()
            duration = (end_time - start_time).total_seconds() / 3600  # hours
        else:
            start_time = end_time = None
            duration = 0
        
        return {
            'start_time': start_time.isoformat() if start_time else None,
            'end_time': end_time.isoformat() if end_time else None,
            'duration_hours': duration
        }
    
    def _analyze_vital_trends(self, patient_df: pd.DataFrame) -> Dict:
        """Analyze trends in vital signs"""
        vital_cols = [col for col in patient_df.columns if any(vital in col.lower() 
                     for vital in ['hr', 'resp', 'sbp', 'dbp', 'temp', 'spo2', 'mbp'])]
        
        trends = {}
        for col in vital_cols:
            if col in patient_df.columns:
                values = patient_df[col].dropna()
                if len(values) > 1:
                    # Calculate trend (slope)
                    x = np.arange(len(values))
                    slope = np.polyfit(x, values, 1)[0]
                    
                    trends[col] = {
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable',
                        'slope': float(slope)
                    }
        
        return trends
    
    def _analyze_risk_progression(self, patient_df: pd.DataFrame) -> Dict:
        """Analyze risk score progression over time"""
        if 'deterioration_label' in patient_df.columns:
            risk_scores = patient_df['deterioration_label'].values
            progression = {
                'initial_risk': float(risk_scores[0]) if len(risk_scores) > 0 else 0,
                'final_risk': float(risk_scores[-1]) if len(risk_scores) > 0 else 0,
                'max_risk': float(np.max(risk_scores)),
                'risk_changes': int(np.sum(np.diff(risk_scores) != 0)),
                'deterioration_episodes': int(np.sum(risk_scores))
            }
        else:
            progression = {'error': 'No risk data available'}
        
        return progression
    
    def _identify_clinical_events(self, patient_df: pd.DataFrame) -> List[Dict]:
        """Identify significant clinical events"""
        events = []
        
        # Identify vital sign abnormalities
        vital_cols = [col for col in patient_df.columns if any(vital in col.lower() 
                     for vital in ['hr', 'resp', 'sbp', 'dbp', 'temp', 'spo2', 'mbp'])]
        
        for col in vital_cols:
            if col in patient_df.columns:
                values = patient_df[col].dropna()
                if len(values) > 0:
                    # Define normal ranges
                    normal_ranges = {
                        'hr': (60, 100),
                        'resp': (12, 20),
                        'sbp': (90, 140),
                        'dbp': (60, 90),
                        'temp': (97, 99),
                        'spo2': (95, 100),
                        'mbp': (70, 105)
                    }
                    
                    vital_type = col.split('_')[0].lower()
                    if vital_type in normal_ranges:
                        min_val, max_val = normal_ranges[vital_type]
                        abnormal_indices = np.where((values < min_val) | (values > max_val))[0]
                        
                        for idx in abnormal_indices:
                            events.append({
                                'type': f'{vital_type.upper()} abnormality',
                                'value': float(values.iloc[idx]),
                                'normal_range': f'{min_val}-{max_val}',
                                'severity': 'high' if values.iloc[idx] < min_val * 0.8 or values.iloc[idx] > max_val * 1.2 else 'moderate',
                                'timestamp': patient_df.iloc[idx]['window_start'] if 'window_start' in patient_df.columns else None
                            })
        
        return events
    
    def _predict_outcome(self, patient_df: pd.DataFrame) -> Dict:
        """Predict patient outcome using available models"""
        if not self.models or patient_df.empty:
            return {'error': 'No models or data available'}
        
        # Prepare features
        feature_cols = [col for col in patient_df.columns 
                       if col not in ['subject_id', 'window_start', 'deterioration_label']]
        
        if not feature_cols:
            return {'error': 'No features available'}
        
        # Use latest record for prediction
        latest_features = patient_df[feature_cols].iloc[-1:].fillna(0)
        
        predictions = {}
        for model_name, model in self.models.items():
            if model_name == 'scaler':
                continue
            
            try:
                if model_name == 'deep':
                    if 'scaler' in self.models:
                        features_scaled = self.models['scaler'].transform(latest_features)
                    else:
                        features_scaled = latest_features.values
                    pred = model.predict(features_scaled)[0][0]
                else:
                    pred = model.predict_proba(latest_features)[0][1]
                
                predictions[model_name] = float(pred)
            except Exception as e:
                logger.error(f"Error predicting with {model_name}: {e}")
                predictions[model_name] = 0.0
        
        # Ensemble prediction
        if predictions:
            ensemble_pred = np.mean(list(predictions.values()))
        else:
            ensemble_pred = 0.0
        
        return {
            'individual_predictions': predictions,
            'ensemble_prediction': float(ensemble_pred),
            'risk_level': self._classify_risk_level(ensemble_pred)
        }
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify risk level based on score"""
        if risk_score >= 0.8:
            return 'CRITICAL'
        elif risk_score >= 0.6:
            return 'HIGH'
        elif risk_score >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def analyze_cohort(self, cohort_criteria: Dict) -> Dict:
        """Analyze a specific patient cohort"""
        
        if self.patient_data.empty:
            return {'error': 'No patient data available'}
        
        # Filter patients based on criteria
        cohort_df = self.patient_data.copy()
        
        # Apply filters
        if 'age_range' in cohort_criteria:
            min_age, max_age = cohort_criteria['age_range']
            # Assuming age is in static features
            cohort_df = cohort_df[cohort_df.get('age', 0).between(min_age, max_age)]
        
        if 'risk_level' in cohort_criteria:
            risk_level = cohort_criteria['risk_level']
            if 'deterioration_label' in cohort_df.columns:
                if risk_level == 'high':
                    cohort_df = cohort_df[cohort_df['deterioration_label'] > 0.7]
                elif risk_level == 'low':
                    cohort_df = cohort_df[cohort_df['deterioration_label'] < 0.3]
        
        # Analyze cohort
        analysis = {
            'cohort_size': len(cohort_df),
            'unique_patients': cohort_df['subject_id'].nunique() if 'subject_id' in cohort_df.columns else 0,
            'outcome_distribution': self._analyze_outcome_distribution(cohort_df),
            'risk_factors': self._identify_risk_factors(cohort_df),
            'model_performance': self._evaluate_model_performance(cohort_df)
        }
        
        return analysis
    
    def _analyze_outcome_distribution(self, cohort_df: pd.DataFrame) -> Dict:
        """Analyze outcome distribution in cohort"""
        if 'deterioration_label' in cohort_df.columns:
            outcomes = cohort_df['deterioration_label'].value_counts()
            return {
                'deterioration_rate': float(outcomes.get(1, 0) / len(cohort_df)),
                'stable_rate': float(outcomes.get(0, 0) / len(cohort_df)),
                'total_patients': len(cohort_df)
            }
        else:
            return {'error': 'No outcome data available'}
    
    def _identify_risk_factors(self, cohort_df: pd.DataFrame) -> Dict:
        """Identify key risk factors in cohort"""
        # This is a simplified analysis - in practice, you'd use more sophisticated methods
        numeric_cols = cohort_df.select_dtypes(include=[np.number]).columns
        risk_factors = {}
        
        for col in numeric_cols:
            if col not in ['subject_id', 'deterioration_label']:
                correlation = cohort_df[col].corr(cohort_df.get('deterioration_label', 0))
                if not np.isnan(correlation) and abs(correlation) > 0.1:
                    risk_factors[col] = {
                        'correlation': float(correlation),
                        'mean_value': float(cohort_df[col].mean()),
                        'std_value': float(cohort_df[col].std())
                    }
        
        # Sort by absolute correlation
        risk_factors = dict(sorted(risk_factors.items(), 
                                 key=lambda x: abs(x[1]['correlation']), 
                                 reverse=True))
        
        return risk_factors
    
    def _evaluate_model_performance(self, cohort_df: pd.DataFrame) -> Dict:
        """Evaluate model performance on cohort"""
        if not self.models or 'deterioration_label' not in cohort_df.columns:
            return {'error': 'No models or outcome data available'}
        
        # Prepare features
        feature_cols = [col for col in cohort_df.columns 
                       if col not in ['subject_id', 'window_start', 'deterioration_label']]
        
        if not feature_cols:
            return {'error': 'No features available'}
        
        X = cohort_df[feature_cols].fillna(0)
        y = cohort_df['deterioration_label']
        
        performance = {}
        for model_name, model in self.models.items():
            if model_name == 'scaler':
                continue
            
            try:
                if model_name == 'deep':
                    if 'scaler' in self.models:
                        X_scaled = self.models['scaler'].transform(X)
                    else:
                        X_scaled = X.values
                    y_pred = model.predict(X_scaled).ravel()
                else:
                    y_pred = model.predict_proba(X)[:, 1]
                
                auc = roc_auc_score(y, y_pred)
                performance[model_name] = {
                    'auc': float(auc),
                    'predictions': y_pred.tolist()
                }
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {e}")
                performance[model_name] = {'error': str(e)}
        
        return performance
    
    def generate_visualizations(self, analysis_results: Dict, output_dir: Path = None):
        """Generate visualization plots for analysis results"""
        
        if output_dir is None:
            output_dir = config.FIGURES_DIR
        
        output_dir.mkdir(exist_ok=True)
        
        # Patient trajectory visualization
        if 'patient_id' in analysis_results:
            self._plot_patient_trajectory(analysis_results, output_dir)
        
        # Cohort analysis visualization
        if 'cohort_size' in analysis_results:
            self._plot_cohort_analysis(analysis_results, output_dir)
        
        # Model performance visualization
        if 'model_performance' in analysis_results:
            self._plot_model_performance(analysis_results['model_performance'], output_dir)
    
    def _plot_patient_trajectory(self, analysis: Dict, output_dir: Path):
        """Plot patient trajectory visualization"""
        patient_id = analysis['patient_id']
        
        # This would create a comprehensive trajectory plot
        # For now, we'll create a simple summary plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Patient {patient_id} Trajectory Analysis', fontsize=16)
        
        # Risk progression
        if 'risk_progression' in analysis:
            risk_data = analysis['risk_progression']
            axes[0, 0].bar(['Initial', 'Final', 'Max'], 
                          [risk_data.get('initial_risk', 0), 
                           risk_data.get('final_risk', 0), 
                           risk_data.get('max_risk', 0)])
            axes[0, 0].set_title('Risk Progression')
            axes[0, 0].set_ylabel('Risk Score')
        
        # Vital trends
        if 'vital_trends' in analysis:
            vital_data = analysis['vital_trends']
            if vital_data:
                trends = [data['trend'] for data in vital_data.values()]
                trend_counts = pd.Series(trends).value_counts()
                axes[0, 1].pie(trend_counts.values, labels=trend_counts.index, autopct='%1.1f%%')
                axes[0, 1].set_title('Vital Sign Trends')
        
        # Clinical events
        if 'clinical_events' in analysis:
            events = analysis['clinical_events']
            event_types = [event['type'] for event in events]
            if event_types:
                event_counts = pd.Series(event_types).value_counts()
                axes[1, 0].bar(event_counts.index, event_counts.values)
                axes[1, 0].set_title('Clinical Events')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Outcome prediction
        if 'outcome_prediction' in analysis:
            pred_data = analysis['outcome_prediction']
            if 'individual_predictions' in pred_data:
                models = list(pred_data['individual_predictions'].keys())
                scores = list(pred_data['individual_predictions'].values())
                axes[1, 1].bar(models, scores)
                axes[1, 1].set_title('Model Predictions')
                axes[1, 1].set_ylabel('Risk Score')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'patient_{patient_id}_trajectory.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_cohort_analysis(self, analysis: Dict, output_dir: Path):
        """Plot cohort analysis visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Cohort Analysis', fontsize=16)
        
        # Outcome distribution
        if 'outcome_distribution' in analysis:
            outcome_data = analysis['outcome_distribution']
            if 'error' not in outcome_data:
                labels = ['Stable', 'Deterioration']
                sizes = [outcome_data.get('stable_rate', 0), outcome_data.get('deterioration_rate', 0)]
                axes[0, 0].pie(sizes, labels=labels, autopct='%1.1f%%')
                axes[0, 0].set_title('Outcome Distribution')
        
        # Risk factors
        if 'risk_factors' in analysis:
            risk_data = analysis['risk_factors']
            if risk_data:
                factors = list(risk_data.keys())[:10]  # Top 10
                correlations = [risk_data[factor]['correlation'] for factor in factors]
                axes[0, 1].barh(factors, correlations)
                axes[0, 1].set_title('Top Risk Factors')
                axes[0, 1].set_xlabel('Correlation with Outcome')
        
        # Model performance
        if 'model_performance' in analysis:
            perf_data = analysis['model_performance']
            if 'error' not in perf_data:
                models = list(perf_data.keys())
                aucs = [perf_data[model].get('auc', 0) for model in models]
                axes[1, 0].bar(models, aucs)
                axes[1, 0].set_title('Model Performance (AUC)')
                axes[1, 0].set_ylabel('AUC Score')
        
        # Cohort size
        axes[1, 1].text(0.5, 0.5, f"Cohort Size: {analysis.get('cohort_size', 0)}\n"
                                   f"Unique Patients: {analysis.get('unique_patients', 0)}", 
                       ha='center', va='center', fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        axes[1, 1].set_title('Cohort Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'cohort_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_model_performance(self, performance: Dict, output_dir: Path):
        """Plot model performance visualization"""
        if 'error' in performance:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # AUC comparison
        models = list(performance.keys())
        aucs = [performance[model].get('auc', 0) for model in models]
        
        axes[0].bar(models, aucs)
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_ylabel('AUC Score')
        axes[0].set_ylim(0, 1)
        
        # ROC curves (if predictions available)
        if any('predictions' in performance[model] for model in models):
            from sklearn.metrics import roc_curve
            # This would plot ROC curves for each model
            axes[1].text(0.5, 0.5, 'ROC Curves\n(Implementation needed)', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_title('ROC Curves')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'model_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, analysis_results: Dict, output_file: Path = None) -> str:
        """Generate comprehensive analysis report"""
        
        if output_file is None:
            output_file = config.RESULTS_DIR / f"retrospective_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Generate visualizations
        self.generate_visualizations(analysis_results)
        
        # Save analysis results
        with open(output_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {output_file}")
        return str(output_file)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Retrospective Analysis for ICU Monitoring")
    parser.add_argument("--analyze", action="store_true", help="Analyze patient trajectory")
    parser.add_argument("--cohort_analysis", action="store_true", help="Analyze patient cohort")
    parser.add_argument("--generate_report", action="store_true", help="Generate analysis report")
    parser.add_argument("--patient_id", type=str, help="Patient ID for analysis")
    parser.add_argument("--cohort", type=str, help="Cohort name for analysis")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = RetrospectiveAnalyzer()
    
    if args.analyze and args.patient_id:
        logger.info(f"Analyzing patient trajectory for {args.patient_id}")
        analysis = analyzer.analyze_patient_trajectory(args.patient_id)
        
        # Generate visualizations
        analyzer.generate_visualizations(analysis)
        
        # Print summary
        print(f"\n=== Patient {args.patient_id} Analysis ===")
        print(f"Total records: {analysis.get('total_records', 0)}")
        print(f"Time span: {analysis.get('time_span', {}).get('duration_hours', 0):.1f} hours")
        print(f"Clinical events: {len(analysis.get('clinical_events', []))}")
        
        if 'outcome_prediction' in analysis:
            pred = analysis['outcome_prediction']
            print(f"Ensemble prediction: {pred.get('ensemble_prediction', 0):.3f}")
            print(f"Risk level: {pred.get('risk_level', 'Unknown')}")
    
    elif args.cohort_analysis:
        logger.info("Performing cohort analysis")
        
        # Define cohort criteria
        cohort_criteria = {}
        if args.cohort == "post_surgical":
            cohort_criteria = {'risk_level': 'high'}
        elif args.cohort == "elderly":
            cohort_criteria = {'age_range': (65, 100)}
        
        analysis = analyzer.analyze_cohort(cohort_criteria)
        
        # Generate visualizations
        analyzer.generate_visualizations(analysis)
        
        # Print summary
        print(f"\n=== Cohort Analysis ===")
        print(f"Cohort size: {analysis.get('cohort_size', 0)}")
        print(f"Unique patients: {analysis.get('unique_patients', 0)}")
        
        if 'outcome_distribution' in analysis:
            outcome = analysis['outcome_distribution']
            if 'error' not in outcome:
                print(f"Deterioration rate: {outcome.get('deterioration_rate', 0):.1%}")
    
    elif args.generate_report:
        logger.info("Generating comprehensive analysis report")
        
        # Perform comprehensive analysis
        if not analyzer.patient_data.empty:
            # Analyze all patients
            all_patients = analyzer.patient_data['subject_id'].unique() if 'subject_id' in analyzer.patient_data.columns else []
            
            analysis_results = {
                'analysis_date': datetime.now().isoformat(),
                'total_patients': len(all_patients),
                'total_records': len(analyzer.patient_data),
                'cohort_analysis': analyzer.analyze_cohort({}),
                'sample_patient_analysis': analyzer.analyze_patient_trajectory(all_patients[0]) if len(all_patients) > 0 else {}
            }
            
            # Generate report
            report_file = analyzer.generate_report(analysis_results)
            print(f"Analysis report generated: {report_file}")
        else:
            print("No data available for analysis")

if __name__ == "__main__":
    main()
