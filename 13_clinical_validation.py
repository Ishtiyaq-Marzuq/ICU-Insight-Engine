"""
13_clinical_validation.py

Clinical validation and protocol generation for ICU patient monitoring system.
Provides validation against clinical standards, protocol generation for deployment,
and compliance tracking for regulatory requirements.

Features:
- Clinical validation against established standards
- Protocol generation for hospital deployment
- Compliance tracking and reporting
- Clinical trial design and management
- Regulatory documentation generation
- Quality assurance and validation testing

Usage:
    python 13_clinical_validation.py --validate --protocol
    python 13_clinical_validation.py --generate_trial_protocol
    python 13_clinical_validation.py --compliance_report
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import argparse
import logging
from dataclasses import dataclass, asdict
from enum import Enum
import config
import utils

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationStatus(Enum):
    """Validation status levels"""
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"

class ComplianceLevel(Enum):
    """Compliance levels"""
    FULLY_COMPLIANT = "FULLY_COMPLIANT"
    PARTIALLY_COMPLIANT = "PARTIALLY_COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"

@dataclass
class ValidationResult:
    """Individual validation result"""
    test_name: str
    status: ValidationStatus
    score: float
    threshold: float
    details: str
    recommendations: List[str]
    clinical_significance: str

@dataclass
class ClinicalProtocol:
    """Clinical protocol definition"""
    protocol_id: str
    title: str
    version: str
    effective_date: datetime
    scope: str
    objectives: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    monitoring_procedures: List[str]
    safety_measures: List[str]
    data_collection: List[str]
    reporting_requirements: List[str]

class ClinicalValidator:
    """Clinical validation and protocol generation system"""
    
    def __init__(self):
        self.validation_results = []
        self.clinical_standards = self._load_clinical_standards()
        self.regulatory_requirements = self._load_regulatory_requirements()
    
    def _load_clinical_standards(self) -> Dict:
        """Load clinical standards and guidelines"""
        return {
            'vital_signs': {
                'heart_rate': {'normal': (60, 100), 'critical_low': 40, 'critical_high': 150},
                'systolic_bp': {'normal': (90, 140), 'critical_low': 70, 'critical_high': 180},
                'diastolic_bp': {'normal': (60, 90), 'critical_low': 40, 'critical_high': 110},
                'oxygen_saturation': {'normal': (95, 100), 'critical_low': 90, 'critical_high': 100},
                'temperature': {'normal': (97, 99), 'critical_low': 95, 'critical_high': 102},
                'respiratory_rate': {'normal': (12, 20), 'critical_low': 8, 'critical_high': 30}
            },
            'alert_response_times': {
                'critical': 2,  # minutes
                'high': 5,
                'medium': 15,
                'low': 30
            },
            'model_performance': {
                'minimum_auc': 0.75,
                'minimum_precision': 0.70,
                'minimum_recall': 0.70,
                'maximum_false_positive_rate': 0.10
            },
            'data_quality': {
                'minimum_completeness': 0.85,
                'maximum_missing_rate': 0.15,
                'maximum_outlier_rate': 0.05
            }
        }
    
    def _load_regulatory_requirements(self) -> Dict:
        """Load regulatory requirements"""
        return {
            'fda_requirements': {
                'clinical_evidence': True,
                'safety_data': True,
                'effectiveness_data': True,
                'risk_benefit_analysis': True,
                'post_market_surveillance': True
            },
            'hipaa_compliance': {
                'data_encryption': True,
                'access_controls': True,
                'audit_logging': True,
                'data_retention': True,
                'breach_notification': True
            },
            'ce_marking': {
                'medical_device_directive': True,
                'risk_management': True,
                'clinical_evaluation': True,
                'quality_management': True
            }
        }
    
    def validate_model_performance(self, model_results: Dict) -> List[ValidationResult]:
        """Validate model performance against clinical standards"""
        results = []
        standards = self.clinical_standards['model_performance']
        
        # AUC validation
        auc = model_results.get('auc', 0)
        auc_result = ValidationResult(
            test_name="Model AUC",
            status=ValidationStatus.PASS if auc >= standards['minimum_auc'] else ValidationStatus.FAIL,
            score=auc,
            threshold=standards['minimum_auc'],
            details=f"Model AUC: {auc:.3f}, Required: {standards['minimum_auc']}",
            recommendations=["Improve model training" if auc < standards['minimum_auc'] else "Model performance acceptable"],
            clinical_significance="AUC below threshold may lead to poor clinical decision support"
        )
        results.append(auc_result)
        
        # Precision validation
        precision = model_results.get('precision', 0)
        precision_result = ValidationResult(
            test_name="Model Precision",
            status=ValidationStatus.PASS if precision >= standards['minimum_precision'] else ValidationStatus.FAIL,
            score=precision,
            threshold=standards['minimum_precision'],
            details=f"Model Precision: {precision:.3f}, Required: {standards['minimum_precision']}",
            recommendations=["Reduce false positives" if precision < standards['minimum_precision'] else "Precision acceptable"],
            clinical_significance="Low precision may cause alert fatigue"
        )
        results.append(precision_result)
        
        # Recall validation
        recall = model_results.get('recall', 0)
        recall_result = ValidationResult(
            test_name="Model Recall",
            status=ValidationStatus.PASS if recall >= standards['minimum_recall'] else ValidationStatus.FAIL,
            score=recall,
            threshold=standards['minimum_recall'],
            details=f"Model Recall: {recall:.3f}, Required: {standards['minimum_recall']}",
            recommendations=["Improve sensitivity" if recall < standards['minimum_recall'] else "Recall acceptable"],
            clinical_significance="Low recall may miss critical patient deterioration"
        )
        results.append(recall_result)
        
        return results
    
    def validate_alert_system(self, alert_data: Dict) -> List[ValidationResult]:
        """Validate alert system performance"""
        results = []
        standards = self.clinical_standards['alert_response_times']
        
        # Response time validation
        for alert_level, max_time in standards.items():
            actual_time = alert_data.get(f'{alert_level}_response_time', 0)
            status = ValidationStatus.PASS if actual_time <= max_time else ValidationStatus.FAIL
            
            result = ValidationResult(
                test_name=f"{alert_level.title()} Alert Response Time",
                status=status,
                score=actual_time,
                threshold=max_time,
                details=f"Response time: {actual_time} minutes, Required: ≤{max_time} minutes",
                recommendations=["Improve response time" if actual_time > max_time else "Response time acceptable"],
                clinical_significance=f"Delayed response to {alert_level} alerts may compromise patient safety"
            )
            results.append(result)
        
        # Alert accuracy validation
        false_positive_rate = alert_data.get('false_positive_rate', 0)
        fp_threshold = self.clinical_standards['model_performance']['maximum_false_positive_rate']
        
        fp_result = ValidationResult(
            test_name="False Positive Rate",
            status=ValidationStatus.PASS if false_positive_rate <= fp_threshold else ValidationStatus.FAIL,
            score=false_positive_rate,
            threshold=fp_threshold,
            details=f"False positive rate: {false_positive_rate:.3f}, Required: ≤{fp_threshold}",
            recommendations=["Reduce false positives" if false_positive_rate > fp_threshold else "False positive rate acceptable"],
            clinical_significance="High false positive rate causes alert fatigue and reduces clinical trust"
        )
        results.append(fp_result)
        
        return results
    
    def validate_data_quality(self, data_metrics: Dict) -> List[ValidationResult]:
        """Validate data quality metrics"""
        results = []
        standards = self.clinical_standards['data_quality']
        
        # Completeness validation
        completeness = data_metrics.get('completeness', 0)
        completeness_result = ValidationResult(
            test_name="Data Completeness",
            status=ValidationStatus.PASS if completeness >= standards['minimum_completeness'] else ValidationStatus.FAIL,
            score=completeness,
            threshold=standards['minimum_completeness'],
            details=f"Data completeness: {completeness:.3f}, Required: ≥{standards['minimum_completeness']}",
            recommendations=["Improve data collection" if completeness < standards['minimum_completeness'] else "Data completeness acceptable"],
            clinical_significance="Incomplete data may lead to inaccurate predictions"
        )
        results.append(completeness_result)
        
        # Missing data validation
        missing_rate = data_metrics.get('missing_rate', 0)
        missing_result = ValidationResult(
            test_name="Missing Data Rate",
            status=ValidationStatus.PASS if missing_rate <= standards['maximum_missing_rate'] else ValidationStatus.FAIL,
            score=missing_rate,
            threshold=standards['maximum_missing_rate'],
            details=f"Missing data rate: {missing_rate:.3f}, Required: ≤{standards['maximum_missing_rate']}",
            recommendations=["Improve data collection processes" if missing_rate > standards['maximum_missing_rate'] else "Missing data rate acceptable"],
            clinical_significance="High missing data rate reduces model reliability"
        )
        results.append(missing_result)
        
        # Outlier validation
        outlier_rate = data_metrics.get('outlier_rate', 0)
        outlier_result = ValidationResult(
            test_name="Outlier Rate",
            status=ValidationStatus.PASS if outlier_rate <= standards['maximum_outlier_rate'] else ValidationStatus.WARNING,
            score=outlier_rate,
            threshold=standards['maximum_outlier_rate'],
            details=f"Outlier rate: {outlier_rate:.3f}, Required: ≤{standards['maximum_outlier_rate']}",
            recommendations=["Review data quality controls" if outlier_rate > standards['maximum_outlier_rate'] else "Outlier rate acceptable"],
            clinical_significance="High outlier rate may indicate data quality issues"
        )
        results.append(outlier_result)
        
        return results
    
    def validate_clinical_safety(self, safety_metrics: Dict) -> List[ValidationResult]:
        """Validate clinical safety measures"""
        results = []
        
        # Adverse event rate
        adverse_events = safety_metrics.get('adverse_events', 0)
        total_patients = safety_metrics.get('total_patients', 1)
        adverse_rate = adverse_events / total_patients
        
        adverse_result = ValidationResult(
            test_name="Adverse Event Rate",
            status=ValidationStatus.PASS if adverse_rate <= 0.05 else ValidationStatus.FAIL,
            score=adverse_rate,
            threshold=0.05,
            details=f"Adverse event rate: {adverse_rate:.3f}, Required: ≤0.05",
            recommendations=["Investigate adverse events" if adverse_rate > 0.05 else "Adverse event rate acceptable"],
            clinical_significance="High adverse event rate indicates safety concerns"
        )
        results.append(adverse_result)
        
        # System downtime
        downtime_hours = safety_metrics.get('downtime_hours', 0)
        total_hours = safety_metrics.get('total_hours', 1)
        uptime_rate = 1 - (downtime_hours / total_hours)
        
        uptime_result = ValidationResult(
            test_name="System Uptime",
            status=ValidationStatus.PASS if uptime_rate >= 0.99 else ValidationStatus.FAIL,
            score=uptime_rate,
            threshold=0.99,
            details=f"System uptime: {uptime_rate:.3f}, Required: ≥0.99",
            recommendations=["Improve system reliability" if uptime_rate < 0.99 else "System uptime acceptable"],
            clinical_significance="System downtime may compromise patient monitoring"
        )
        results.append(uptime_result)
        
        return results
    
    def generate_clinical_protocol(self, protocol_type: str = "icu_monitoring") -> ClinicalProtocol:
        """Generate clinical protocol for deployment"""
        
        if protocol_type == "icu_monitoring":
            protocol = ClinicalProtocol(
                protocol_id="ICU-MON-001",
                title="AI-Driven Real-Time Multi-Modal ICU Patient Monitoring Protocol",
                version="1.0",
                effective_date=datetime.now(),
                scope="This protocol defines the implementation and use of AI-driven real-time monitoring for ICU patients to detect clinical deterioration early and support timely medical intervention.",
                objectives=[
                    "Implement real-time monitoring of vital signs and clinical parameters",
                    "Provide early warning of patient deterioration",
                    "Support clinical decision-making with AI-driven insights",
                    "Improve patient outcomes through timely intervention",
                    "Reduce ICU length of stay and mortality rates"
                ],
                inclusion_criteria=[
                    "Adult patients (≥18 years) admitted to ICU",
                    "Expected ICU stay ≥24 hours",
                    "Consent for AI monitoring participation",
                    "Availability of required monitoring equipment"
                ],
                exclusion_criteria=[
                    "Patients with do-not-resuscitate orders",
                    "Patients with known AI system contraindications",
                    "Patients unable to provide informed consent",
                    "Pregnant patients (if applicable to monitoring parameters)"
                ],
                monitoring_procedures=[
                    "Continuous vital signs monitoring (heart rate, blood pressure, oxygen saturation, temperature, respiratory rate)",
                    "Real-time data collection from monitoring equipment",
                    "AI model prediction every 15 minutes",
                    "Alert generation based on risk stratification",
                    "Clinical response to alerts within defined timeframes",
                    "Documentation of all alerts and responses"
                ],
                safety_measures=[
                    "Redundant monitoring systems for critical parameters",
                    "Manual override capability for all AI recommendations",
                    "Regular calibration of monitoring equipment",
                    "Backup power systems for continuous operation",
                    "Regular system maintenance and updates",
                    "Staff training on AI system operation"
                ],
                data_collection=[
                    "Demographic and clinical characteristics",
                    "Vital signs and laboratory values",
                    "Medication administration records",
                    "Clinical interventions and responses",
                    "Patient outcomes and complications",
                    "System performance metrics"
                ],
                reporting_requirements=[
                    "Daily monitoring reports",
                    "Weekly performance summaries",
                    "Monthly quality assurance reviews",
                    "Quarterly safety assessments",
                    "Annual effectiveness evaluations",
                    "Adverse event reporting within 24 hours"
                ]
            )
        
        return protocol
    
    def generate_trial_protocol(self) -> Dict:
        """Generate clinical trial protocol"""
        
        trial_protocol = {
            "trial_id": "ICU-AI-MON-2024-001",
            "title": "Randomized Controlled Trial of AI-Driven Real-Time Multi-Modal ICU Patient Monitoring",
            "phase": "Phase III",
            "design": "Randomized, controlled, parallel-group trial",
            "primary_endpoint": "ICU mortality rate",
            "secondary_endpoints": [
                "ICU length of stay",
                "Time to detection of clinical deterioration",
                "False positive alert rate",
                "Clinician satisfaction scores",
                "Cost-effectiveness analysis"
            ],
            "sample_size": {
                "total_patients": 1000,
                "intervention_group": 500,
                "control_group": 500,
                "power": 0.80,
                "alpha": 0.05,
                "effect_size": 0.15
            },
            "randomization": {
                "method": "Block randomization",
                "stratification": ["Age group", "ICU type", "Admission diagnosis"],
                "allocation_ratio": "1:1"
            },
            "intervention": {
                "description": "AI-driven real-time monitoring with automated alerts",
                "duration": "Entire ICU stay",
                "components": [
                    "Continuous vital signs monitoring",
                    "AI risk assessment every 15 minutes",
                    "Automated alert generation",
                    "Clinical decision support recommendations"
                ]
            },
            "control": {
                "description": "Standard ICU monitoring without AI assistance",
                "components": [
                    "Standard vital signs monitoring",
                    "Manual chart review",
                    "Standard alert systems",
                    "Routine clinical assessment"
                ]
            },
            "inclusion_criteria": [
                "Age ≥18 years",
                "ICU admission expected ≥24 hours",
                "Informed consent obtained",
                "No contraindications to monitoring"
            ],
            "exclusion_criteria": [
                "DNR/DNI orders",
                "Pregnancy",
                "Known AI system contraindications",
                "Inability to provide consent"
            ],
            "outcome_measures": {
                "primary": "ICU mortality (30-day)",
                "secondary": [
                    "ICU length of stay",
                    "Hospital length of stay",
                    "Time to clinical deterioration",
                    "Alert response time",
                    "False positive rate",
                    "Clinician workload",
                    "Patient satisfaction"
                ]
            },
            "statistical_analysis": {
                "primary_analysis": "Intention-to-treat",
                "statistical_tests": [
                    "Chi-square test for mortality",
                    "t-test for continuous outcomes",
                    "Kaplan-Meier survival analysis",
                    "Cox proportional hazards model"
                ],
                "interim_analysis": "At 50% enrollment",
                "stopping_rules": "Futility or efficacy boundaries"
            },
            "safety_monitoring": {
                "data_safety_monitoring_board": "Independent DSMB",
                "adverse_event_reporting": "Within 24 hours",
                "serious_adverse_events": "Immediate reporting",
                "safety_interim_analysis": "Every 6 months"
            },
            "regulatory_considerations": {
                "fda_approval": "Required for medical device",
                "institutional_review_board": "Local IRB approval required",
                "informed_consent": "Written consent required",
                "data_privacy": "HIPAA compliance required"
            }
        }
        
        return trial_protocol
    
    def generate_compliance_report(self, validation_results: List[ValidationResult]) -> Dict:
        """Generate compliance report"""
        
        total_tests = len(validation_results)
        passed_tests = sum(1 for result in validation_results if result.status == ValidationStatus.PASS)
        failed_tests = sum(1 for result in validation_results if result.status == ValidationStatus.FAIL)
        warning_tests = sum(1 for result in validation_results if result.status == ValidationStatus.WARNING)
        
        compliance_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if compliance_rate >= 0.95:
            compliance_level = ComplianceLevel.FULLY_COMPLIANT
        elif compliance_rate >= 0.80:
            compliance_level = ComplianceLevel.PARTIALLY_COMPLIANT
        else:
            compliance_level = ComplianceLevel.NON_COMPLIANT
        
        report = {
            "report_date": datetime.now().isoformat(),
            "compliance_level": compliance_level.value,
            "compliance_rate": compliance_rate,
            "test_summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests
            },
            "validation_results": [asdict(result) for result in validation_results],
            "recommendations": self._generate_recommendations(validation_results),
            "next_steps": self._generate_next_steps(compliance_level, validation_results)
        }
        
        return report
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        failed_tests = [result for result in validation_results if result.status == ValidationStatus.FAIL]
        warning_tests = [result for result in validation_results if result.status == ValidationStatus.WARNING]
        
        for result in failed_tests:
            recommendations.extend(result.recommendations)
        
        for result in warning_tests:
            recommendations.extend(result.recommendations)
        
        # Add general recommendations
        if failed_tests:
            recommendations.append("Conduct comprehensive system review before deployment")
            recommendations.append("Implement additional testing and validation")
        
        if warning_tests:
            recommendations.append("Monitor warning areas closely during initial deployment")
            recommendations.append("Develop mitigation strategies for identified risks")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _generate_next_steps(self, compliance_level: ComplianceLevel, validation_results: List[ValidationResult]) -> List[str]:
        """Generate next steps based on compliance level"""
        next_steps = []
        
        if compliance_level == ComplianceLevel.FULLY_COMPLIANT:
            next_steps.extend([
                "Proceed with clinical trial initiation",
                "Submit regulatory applications",
                "Begin pilot deployment planning",
                "Finalize clinical protocols"
            ])
        elif compliance_level == ComplianceLevel.PARTIALLY_COMPLIANT:
            next_steps.extend([
                "Address failed validation tests",
                "Implement additional safety measures",
                "Conduct limited pilot study",
                "Revise protocols based on findings"
            ])
        else:
            next_steps.extend([
                "Conduct comprehensive system redesign",
                "Implement additional validation testing",
                "Address all compliance issues",
                "Re-evaluate deployment timeline"
            ])
        
        return next_steps

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Clinical Validation and Protocol Generation")
    parser.add_argument("--validate", action="store_true", help="Run clinical validation")
    parser.add_argument("--protocol", action="store_true", help="Generate clinical protocol")
    parser.add_argument("--generate_trial_protocol", action="store_true", help="Generate trial protocol")
    parser.add_argument("--compliance_report", action="store_true", help="Generate compliance report")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize validator
    validator = ClinicalValidator()
    
    if args.validate:
        logger.info("Running clinical validation...")
        
        # Sample validation data (in practice, this would come from actual system metrics)
        model_results = {
            'auc': 0.85,
            'precision': 0.78,
            'recall': 0.82,
            'f1': 0.80
        }
        
        alert_data = {
            'critical_response_time': 1.5,
            'high_response_time': 4.0,
            'medium_response_time': 12.0,
            'low_response_time': 25.0,
            'false_positive_rate': 0.08
        }
        
        data_metrics = {
            'completeness': 0.92,
            'missing_rate': 0.08,
            'outlier_rate': 0.03
        }
        
        safety_metrics = {
            'adverse_events': 2,
            'total_patients': 100,
            'downtime_hours': 2,
            'total_hours': 8760
        }
        
        # Run validations
        validation_results = []
        validation_results.extend(validator.validate_model_performance(model_results))
        validation_results.extend(validator.validate_alert_system(alert_data))
        validation_results.extend(validator.validate_data_quality(data_metrics))
        validation_results.extend(validator.validate_clinical_safety(safety_metrics))
        
        # Print results
        print("\n=== Clinical Validation Results ===")
        for result in validation_results:
            status_icon = "✅" if result.status == ValidationStatus.PASS else "⚠️" if result.status == ValidationStatus.WARNING else "❌"
            print(f"{status_icon} {result.test_name}: {result.status.value}")
            print(f"   Score: {result.score:.3f}, Threshold: {result.threshold}")
            print(f"   Details: {result.details}")
            if result.recommendations:
                print(f"   Recommendations: {', '.join(result.recommendations)}")
            print()
    
    if args.protocol:
        logger.info("Generating clinical protocol...")
        protocol = validator.generate_clinical_protocol()
        
        # Save protocol
        protocol_file = config.RESULTS_DIR / f"clinical_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(protocol_file, 'w') as f:
            json.dump(asdict(protocol), f, indent=2, default=str)
        
        print(f"Clinical protocol generated: {protocol_file}")
        print(f"Protocol ID: {protocol.protocol_id}")
        print(f"Title: {protocol.title}")
        print(f"Version: {protocol.version}")
    
    if args.generate_trial_protocol:
        logger.info("Generating clinical trial protocol...")
        trial_protocol = validator.generate_trial_protocol()
        
        # Save trial protocol
        trial_file = config.RESULTS_DIR / f"trial_protocol_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(trial_file, 'w') as f:
            json.dump(trial_protocol, f, indent=2, default=str)
        
        print(f"Clinical trial protocol generated: {trial_file}")
        print(f"Trial ID: {trial_protocol['trial_id']}")
        print(f"Design: {trial_protocol['design']}")
        print(f"Sample Size: {trial_protocol['sample_size']['total_patients']} patients")
    
    if args.compliance_report:
        logger.info("Generating compliance report...")
        
        # Run validation first
        model_results = {'auc': 0.85, 'precision': 0.78, 'recall': 0.82}
        alert_data = {'critical_response_time': 1.5, 'false_positive_rate': 0.08}
        data_metrics = {'completeness': 0.92, 'missing_rate': 0.08, 'outlier_rate': 0.03}
        safety_metrics = {'adverse_events': 2, 'total_patients': 100, 'downtime_hours': 2, 'total_hours': 8760}
        
        validation_results = []
        validation_results.extend(validator.validate_model_performance(model_results))
        validation_results.extend(validator.validate_alert_system(alert_data))
        validation_results.extend(validator.validate_data_quality(data_metrics))
        validation_results.extend(validator.validate_clinical_safety(safety_metrics))
        
        # Generate compliance report
        compliance_report = validator.generate_compliance_report(validation_results)
        
        # Save report
        report_file = config.RESULTS_DIR / f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(compliance_report, f, indent=2, default=str)
        
        print(f"Compliance report generated: {report_file}")
        print(f"Compliance Level: {compliance_report['compliance_level']}")
        print(f"Compliance Rate: {compliance_report['compliance_rate']:.1%}")
        print(f"Tests Passed: {compliance_report['test_summary']['passed']}/{compliance_report['test_summary']['total_tests']}")

if __name__ == "__main__":
    main()
