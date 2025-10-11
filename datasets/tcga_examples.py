"""
TCGA Cohort Loader - Example Usage and Documentation

This module demonstrates how to use the TCGA Cohort Loader system
to download, process, and integrate TCGA data with existing multimodal datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Import TCGA modules
from datasets.tcga_cohort_loader import TCGACohortLoader
from datasets.tcga_integration import TCGAIntegrationManager
from datasets.tcga_preprocessing import TCGADataPreprocessor


def example_basic_usage():
    """Example of basic TCGA data loading."""
    logging.info("=== Basic TCGA Data Loading Example ===")
    
    # Initialize TCGA loader
    loader = TCGACohortLoader(data_dir="./tcga_data", cache_dir="./tcga_cache")
    
    # Get available projects
    available_projects = list(loader.TCGA_PROJECTS.keys())
    logging.info(f"Available TCGA projects: {available_projects}")
    
    # Example: Get project information for LIHC
    project_id = 'lihc'
    try:
        project_info = loader.get_project_info(project_id)
        logging.info(f"Retrieved project info for {project_id}")
        
        # Get available data types
        data_types = loader.get_available_data_types(project_id)
        logging.info(f"Available data types for {project_id}: {data_types}")
        
    except Exception as e:
        logging.info(f"Error retrieving project info: {e}")


def example_data_download():
    """Example of downloading TCGA data."""
    logging.info("\n=== TCGA Data Download Example ===")
    
    loader = TCGACohortLoader(data_dir="./tcga_data", cache_dir="./tcga_cache")
    
    # Download data for LIHC project
    project_id = 'lihc'
    data_types = ['Clinical', 'Gene Expression Quantification']
    
    try:
        logging.info(f"Downloading data for {project_id}...")
        # Note: This would actually download data from TCGA API
        # cohort_data = loader.download_cohort_data(project_id, data_types)
        logging.info(f"Data download completed for {project_id}")
        
    except Exception as e:
        logging.info(f"Error downloading data: {e}")


def example_data_processing():
    """Example of processing TCGA data."""
    logging.info("\n=== TCGA Data Processing Example ===")
    
    # Create sample data for demonstration
    sample_data = create_sample_tcga_data()
    
    # Initialize preprocessor
    preprocessor = TCGADataPreprocessor()
    
    # Process clinical data
    if 'clinical' in sample_data:
        logging.info("Processing clinical data...")
        processed_clinical = preprocessor.preprocess_clinical_data(sample_data['clinical'])
        logging.info(f"Clinical data processed: {processed_clinical.shape}")
    
    # Process mRNA data
    if 'mrna' in sample_data:
        logging.info("Processing mRNA data...")
        processed_mrna = preprocessor.preprocess_mrna_data(sample_data['mrna'])
        logging.info(f"mRNA data processed: {processed_mrna.shape}")
    
    # Process stage data
    if 'stages' in sample_data:
        logging.info("Processing stage data...")
        processed_stages = preprocessor.preprocess_stage_data(sample_data['stages'])
        logging.info(f"Stage data processed: {processed_stages.shape}")


def example_integration():
    """Example of integrating TCGA data with existing system."""
    logging.info("\n=== TCGA Integration Example ===")
    
    sample_data = create_sample_tcga_data()
    
    save_sample_data(sample_data, 'lihc')
    
    integration_manager = TCGAIntegrationManager(
        tcga_data_dir="./tcga_data",
        target_data_dir="./data"
    )
    
    project_id = 'lihc'
    success = integration_manager.prepare_tcga_data_for_existing_structure(project_id)
    
    if success:
        logging.info(f"Successfully prepared TCGA data for {project_id}")
        
        modalities = ['clinical', 'mrna', 'image']
        dataset = integration_manager.create_multimodal_dataset_from_tcga(project_id, modalities)
        
        if dataset:
            logging.info(f"Created multimodal dataset with {len(dataset)} samples")
        else:
            logging.info("Failed to create multimodal dataset")
    else:
        logging.info(f"Failed to prepare TCGA data for {project_id}")


def example_complete_workflow():
    """Example of complete TCGA workflow."""
    logging.info("\n=== Complete TCGA Workflow Example ===")
    
    loader = TCGACohortLoader(data_dir="./tcga_data", cache_dir="./tcga_cache")
    integration_manager = TCGAIntegrationManager(
        tcga_data_dir="./tcga_data",
        target_data_dir="./data"
    )
    preprocessor = TCGADataPreprocessor()
    
    sample_data = create_sample_tcga_data()
    save_sample_data(sample_data, 'lihc')
    
    logging.info("Processing TCGA data...")
    processed_data = {}
    
    for data_type, df in sample_data.items():
        if data_type == 'clinical':
            processed_data[data_type] = preprocessor.preprocess_clinical_data(df)
        elif data_type == 'mrna':
            processed_data[data_type] = preprocessor.preprocess_mrna_data(df)
        elif data_type == 'image':
            processed_data[data_type] = preprocessor.preprocess_image_data(df)
        elif data_type == 'stages':
            processed_data[data_type] = preprocessor.preprocess_stage_data(df)
    
    save_sample_data(processed_data, 'lihc', suffix='_processed')
    
    logging.info("Integrating with existing system...")
    success = integration_manager.prepare_tcga_data_for_existing_structure('lihc')
    
    if success:
        logging.info("Integration successful!")
        
        modalities = ['clinical', 'mrna', 'image']
        dataset = integration_manager.create_multimodal_dataset_from_tcga('lihc', modalities)
        
        if dataset:
            logging.info(f"Multimodal dataset created with {len(dataset)} samples")
            logging.info(f"Modalities: {dataset.modalities}")
            logging.info(f"Column map keys: {list(dataset.column_map.keys())}")
        else:
            logging.info("Failed to create multimodal dataset")
    else:
        logging.info("Integration failed")


def create_sample_tcga_data():
    """Create sample TCGA data for demonstration."""
    np.random.seed(42)
    
    n_patients = 100
    patient_ids = [f"TCGA-{i:02d}-{j:04d}" for i in range(1, 11) for j in range(1, 11)]
    patient_ids = patient_ids[:n_patients]
    
    clinical_data = {
        'pid': patient_ids,
        'age_at_diagnosis': np.random.randint(30, 80, n_patients),
        'gender': np.random.choice(['male', 'female'], n_patients),
        'race': np.random.choice(['white', 'black', 'asian', 'other'], n_patients),
        'ethnicity': np.random.choice(['hispanic', 'non-hispanic'], n_patients),
        'vital_status': np.random.choice(['alive', 'dead'], n_patients),
        'tumor_stage': np.random.choice(['stage i', 'stage ii', 'stage iii', 'stage iv'], n_patients),
        'tumor_grade': np.random.choice(['g1', 'g2', 'g3', 'g4'], n_patients)
    }
    
    n_genes = 50
    mrna_data = {
        'pid': patient_ids,
        **{f'gene_{i:03d}': np.random.exponential(1, n_patients) for i in range(n_genes)}
    }
    
    n_image_features = 20
    image_data = {
        'pid': patient_ids,
        **{f'img_feature_{i:02d}': np.random.normal(0, 1, n_patients) for i in range(n_image_features)}
    }
    
    stages_data = {
        'pid': patient_ids,
        'stage': clinical_data['tumor_stage'],
        'grade': clinical_data['tumor_grade']
    }
    
    return {
        'clinical': pd.DataFrame(clinical_data),
        'mrna': pd.DataFrame(mrna_data),
        'image': pd.DataFrame(image_data),
        'stages': pd.DataFrame(stages_data)
    }


def save_sample_data(data_dict, project_id, suffix=''):
    """Save sample data to files."""
    data_dir = Path("./tcga_data") / project_id
    data_dir.mkdir(parents=True, exist_ok=True)
    
    for data_type, df in data_dict.items():
        file_path = data_dir / f"{project_id}_{data_type}{suffix}.csv"
        df.to_csv(file_path, index=False)
        logging.info(f"Saved {data_type} data to {file_path}")


def main():
    """Run all examples."""
    logging.info("TCGA Cohort Loader - Examples and Documentation")
    logging.info("=" * 50)
    
    # Run examples
    example_basic_usage()
    example_data_download()
    example_data_processing()
    example_integration()
    example_complete_workflow()
    
    logging.info("\n" + "=" * 50)
    logging.info("Examples completed successfully!")
    logging.info("\nTo use TCGA Cohort Loader in your project:")
    logging.info("1. Import the required modules")
    logging.info("2. Initialize TCGACohortLoader for data download")
    logging.info("3. Use TCGAIntegrationManager for integration")
    logging.info("4. Apply TCGADataPreprocessor for data preprocessing")
    logging.info("5. Create multimodal datasets with existing structure")


if __name__ == "__main__":
    main()
