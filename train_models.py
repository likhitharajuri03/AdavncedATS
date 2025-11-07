"""
Training script for all ML models
Run this script to train and save models before starting the application
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from models.resume_classifier import train_classifier_on_dataset
from models.skill_extractor import SkillExtractor
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = ['models', 'data', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created/verified directory: {directory}")

def prepare_dataset(csv_path: str):
    """
    Prepare and validate the dataset
    """
    logger.info(f"Loading dataset from {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        logger.info(f"Dataset loaded successfully: {len(df)} rows, {len(df.columns)} columns")
        
        # Display column names
        logger.info(f"Columns: {list(df.columns)}")
        
        # Check for required columns
        required_columns = ['Job Title', 'Description', 'IT Skills', 'Soft Skills']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")
            logger.info("Attempting to map columns...")
            
            # Try to find similar column names
            column_mapping = {}
            for req_col in missing_columns:
                for df_col in df.columns:
                    if req_col.lower() in df_col.lower():
                        column_mapping[df_col] = req_col
                        logger.info(f"Mapping '{df_col}' to '{req_col}'")
            
            if column_mapping:
                df = df.rename(columns=column_mapping)
        
        # Handle missing values
        for col in df.columns:
            missing_count = df[col].isna().sum()
            if missing_count > 0:
                logger.info(f"Column '{col}' has {missing_count} missing values")
        
        # Save processed dataset
        processed_path = csv_path.replace('.csv', '_processed.csv')
        df.to_csv(processed_path, index=False)
        logger.info(f"Processed dataset saved to {processed_path}")
        
        return df, processed_path
        
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        raise

def train_resume_classifier(data_path: str, output_path: str):
    """
    Train the resume classification model
    """
    logger.info("=" * 50)
    logger.info("TRAINING RESUME CLASSIFIER")
    logger.info("=" * 50)
    
    try:
        classifier, results = train_classifier_on_dataset(data_path, output_path)
        
        # Print detailed results
        logger.info("\n" + "=" * 50)
        logger.info("TRAINING RESULTS")
        logger.info("=" * 50)
        
        for model_name, metrics in results.items():
            logger.info(f"\n{model_name}:")
            logger.info(f"  Accuracy:  {metrics['accuracy']*100:.2f}%")
            logger.info(f"  Precision: {metrics['precision']*100:.2f}%")
            logger.info(f"  Recall:    {metrics['recall']*100:.2f}%")
            logger.info(f"  F1 Score:  {metrics['f1']*100:.2f}%")
        
        best_model = max(results, key=lambda x: results[x]['f1'])
        logger.info(f"\n🏆 Best Model: {best_model}")
        logger.info(f"   F1 Score: {results[best_model]['f1']*100:.2f}%")
        
        # Check if accuracy exceeds paper's requirement
        if results[best_model]['accuracy'] > 0.99:
            logger.info(f"✅ ACHIEVEMENT: Accuracy {results[best_model]['accuracy']*100:.2f}% exceeds paper requirement (99%)")
        else:
            logger.warning(f"⚠️  Accuracy {results[best_model]['accuracy']*100:.2f}% below paper requirement (99%)")
        
        logger.info(f"\n✅ Model saved to: {output_path}")
        return classifier, results
        
    except Exception as e:
        logger.error(f"Error training classifier: {str(e)}")
        raise

def build_skill_database(data_path: str, output_path: str):
    """
    Build and save skill extraction database
    """
    logger.info("=" * 50)
    logger.info("BUILDING SKILL DATABASE")
    logger.info("=" * 50)
    
    try:
        extractor = SkillExtractor(skills_database_path=data_path)
        
        logger.info(f"Technical Skills: {len(extractor.technical_skills)}")
        logger.info(f"Soft Skills: {len(extractor.soft_skills)}")
        
        # Save skill database
        skill_data = {
            'technical_skills': list(extractor.technical_skills),
            'soft_skills': list(extractor.soft_skills)
        }
        
        joblib.dump(skill_data, output_path)
        logger.info(f"✅ Skill database saved to: {output_path}")
        
        # Sample some skills
        logger.info("\nSample Technical Skills:")
        for skill in list(extractor.technical_skills)[:20]:
            logger.info(f"  - {skill}")
        
        return extractor
        
    except Exception as e:
        logger.error(f"Error building skill database: {str(e)}")
        raise

def validate_models(classifier_path: str, skills_path: str):
    """
    Validate that models can be loaded
    """
    logger.info("=" * 50)
    logger.info("VALIDATING MODELS")
    logger.info("=" * 50)
    
    try:
        # Test classifier loading
        from models.resume_classifier import ResumeClassifier
        classifier = ResumeClassifier()
        classifier.load_model(classifier_path)
        logger.info("✅ Resume Classifier loaded successfully")
        
        # Test skill database loading
        skill_data = joblib.load(skills_path)
        logger.info(f"✅ Skill Database loaded: {len(skill_data['technical_skills'])} technical, {len(skill_data['soft_skills'])} soft skills")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model validation failed: {str(e)}")
        return False

def main():
    """
    Main training pipeline
    """
    logger.info("=" * 60)
    logger.info("AI RESUME TRACKER - MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Configuration
    DATASET_PATH = 'data/jobs_dataset.csv'
    CLASSIFIER_OUTPUT = 'models/resume_classifier.pkl'
    SKILLS_OUTPUT = 'models/skill_database.pkl'
    
    try:
        # Step 1: Create directories
        logger.info("\nStep 1: Creating directories...")
        create_directories()
        
        # Step 2: Prepare dataset
        logger.info("\nStep 2: Preparing dataset...")
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset not found at {DATASET_PATH}")
            logger.info("Please place your jobs_dataset.csv in the data/ directory")
            return
        
        df, processed_path = prepare_dataset(DATASET_PATH)
        
        # Step 3: Train classifier
        logger.info("\nStep 3: Training resume classifier...")
        classifier, results = train_resume_classifier(processed_path, CLASSIFIER_OUTPUT)
        
        # Step 4: Build skill database
        logger.info("\nStep 4: Building skill database...")
        extractor = build_skill_database(processed_path, SKILLS_OUTPUT)
        
        # Step 5: Validate models
        logger.info("\nStep 5: Validating models...")
        if validate_models(CLASSIFIER_OUTPUT, SKILLS_OUTPUT):
            logger.info("\n" + "=" * 60)
            logger.info("✅ ALL MODELS TRAINED AND VALIDATED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"\nModels saved:")
            logger.info(f"  - Classifier: {CLASSIFIER_OUTPUT}")
            logger.info(f"  - Skills DB:  {SKILLS_OUTPUT}")
            logger.info("\nYou can now start the application with: python backend/app.py")
        else:
            logger.error("\n❌ Model validation failed. Please check the errors above.")
        
    except Exception as e:
        logger.error(f"\n❌ Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()