"""
TCGA Integration Module

This module provides integration functions between TCGA Cohort Loader
and the existing multimodal dataset structure.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path
import logging

from .tcga_cohort_loader import TCGACohortLoader
from .dataset_classes import CustomMultiModalDatasetStratified

logger = logging.getLogger(__name__)


class TCGAIntegrationManager:
    """
    Manager class for integrating TCGA data with existing dataset structure.
    """
    
    def __init__(self, tcga_data_dir: str = "./tcga_data", target_data_dir: str = "./data"):
        """
        Initialize TCGA Integration Manager.
        
        Args:
            tcga_data_dir: Directory containing TCGA data
            target_data_dir: Directory where existing datasets are stored
        """
        self.tcga_loader = TCGACohortLoader(tcga_data_dir)
        self.target_data_dir = Path(target_data_dir)
        self.target_data_dir.mkdir(parents=True, exist_ok=True)
        
    def prepare_tcga_data_for_existing_structure(self, project_id: str) -> bool:
        """
        Prepare TCGA data to match existing dataset structure.
        
        Args:
            project_id: TCGA project identifier ('lihc', 'lusc', 'brca')
            
        Returns:
            True if preparation successful
        """
        try:
            # Load TCGA cohort data
            cohort_data = self.tcga_loader.load_cohort_from_files(project_id)
            
            if not cohort_data:
                logger.error(f"No TCGA data found for {project_id}")
                return False
            
            # Process each data type
            processed_data = {}
            
            # Process clinical data
            if 'clinical' in cohort_data:
                processed_data['clinical'] = self._process_clinical_data(
                    cohort_data['clinical'], project_id
                )
            
            # Process mRNA data
            if 'mrna' in cohort_data:
                processed_data['mrna'] = self._process_mrna_data(
                    cohort_data['mrna'], project_id
                )
            
            # Process image data
            if 'image' in cohort_data:
                processed_data['image'] = self._process_image_data(
                    cohort_data['image'], project_id
                )
            
            # Process stage data
            if 'stages' in cohort_data:
                processed_data['stages'] = self._process_stage_data(
                    cohort_data['stages'], project_id
                )
            
            # Save processed data in target directory
            self._save_processed_data(processed_data, project_id)
            
            logger.info(f"Successfully prepared TCGA data for {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to prepare TCGA data for {project_id}: {e}")
            return False
    
    def _process_clinical_data(self, clinical_df: pd.DataFrame, project_id: str) -> pd.DataFrame:
        """Process clinical data to match existing structure."""
        processed_df = clinical_df.copy()
        
        # Ensure pid column exists
        if 'pid' not in processed_df.columns:
            logger.warning(f"No 'pid' column found in clinical data for {project_id}")
            return processed_df
        
        # Convert numeric columns
        numeric_columns = ['age_at_diagnosis', 'days_to_death', 'days_to_last_follow_up']
        for col in numeric_columns:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Create binary features for categorical variables
        categorical_columns = ['gender', 'race', 'ethnicity', 'vital_status']
        for col in categorical_columns:
            if col in processed_df.columns:
                # Create dummy variables
                dummies = pd.get_dummies(processed_df[col], prefix=col)
                processed_df = pd.concat([processed_df, dummies], axis=1)
                processed_df.drop(columns=[col], inplace=True)
        
        # Handle missing values
        processed_df.fillna(0, inplace=True)
        
        return processed_df
    
    def _process_mrna_data(self, mrna_df: pd.DataFrame, project_id: str) -> pd.DataFrame:
        """Process mRNA data to match existing structure."""
        processed_df = mrna_df.copy()
        
        # For now, create placeholder features
        # In a real implementation, you would download and process actual expression data
        if 'pid' in processed_df.columns:
            # Create dummy gene expression features
            n_genes = 1000  # Placeholder number of genes
            gene_features = np.random.randn(len(processed_df), n_genes)
            
            gene_columns = [f'gene_{i:04d}' for i in range(n_genes)]
            gene_df = pd.DataFrame(gene_features, columns=gene_columns)
            
            processed_df = pd.concat([processed_df[['pid']], gene_df], axis=1)
        
        return processed_df
    
    def _process_image_data(self, image_df: pd.DataFrame, project_id: str) -> pd.DataFrame:
        """Process image data to match existing structure."""
        processed_df = image_df.copy()
        
        # For now, create placeholder image features
        # In a real implementation, you would extract features from histopathology images
        if 'pid' in processed_df.columns:
            # Create dummy image features
            n_features = 100  # Placeholder number of image features
            image_features = np.random.randn(len(processed_df), n_features)
            
            feature_columns = [f'img_feature_{i:03d}' for i in range(n_features)]
            feature_df = pd.DataFrame(image_features, columns=feature_columns)
            
            processed_df = pd.concat([processed_df[['pid']], feature_df], axis=1)
        
        return processed_df
    
    def _process_stage_data(self, stages_df: pd.DataFrame, project_id: str) -> pd.DataFrame:
        """Process stage data to match existing structure."""
        processed_df = stages_df.copy()
        
        # Ensure pid column exists
        if 'pid' not in processed_df.columns:
            logger.warning(f"No 'pid' column found in stage data for {project_id}")
            return processed_df
        
        # Clean stage values
        if 'stage' in processed_df.columns:
            # Map TCGA stage values to standardized format
            stage_mapping = {
                'stage i': 'stage i',
                'stage ii': 'stage ii', 
                'stage iii': 'stage iii',
                'stage iv': 'stage iv',
                'stage ia': 'stage i',
                'stage ib': 'stage i',
                'stage iia': 'stage ii',
                'stage iib': 'stage ii',
                'stage iiia': 'stage iii',
                'stage iiib': 'stage iii',
                'stage iiic': 'stage iii',
                'stage iva': 'stage iv',
                'stage ivb': 'stage iv'
            }
            
            processed_df['stage'] = processed_df['stage'].str.lower().map(stage_mapping)
            processed_df['stage'] = processed_df['stage'].fillna('unknown')
        
        return processed_df
    
    def _save_processed_data(self, processed_data: Dict[str, pd.DataFrame], project_id: str):
        """Save processed data to target directory."""
        for data_type, df in processed_data.items():
            if df is not None and not df.empty:
                file_path = self.target_data_dir / f"{project_id}_{data_type}.csv"
                df.to_csv(file_path, index=False)
                logger.info(f"Saved {data_type} data for {project_id}: {len(df)} records")
    
    def create_multimodal_dataset_from_tcga(self, project_id: str, modalities: List[str]) -> Optional[CustomMultiModalDatasetStratified]:
        """
        Create a multimodal dataset from TCGA data.
        
        Args:
            project_id: TCGA project identifier
            modalities: List of modalities to include
            
        Returns:
            CustomMultiModalDatasetStratified object or None if failed
        """
        try:
            # Load processed data
            features_data = {}
            labels_data = None
            
            # Load features for each modality
            for modality in modalities:
                file_path = self.target_data_dir / f"{project_id}_{modality}.csv"
                if file_path.exists():
                    features_data[modality] = pd.read_csv(file_path)
                else:
                    logger.warning(f"No {modality} data found for {project_id}")
            
            # Load labels
            labels_path = self.target_data_dir / f"{project_id}_stages.csv"
            if labels_path.exists():
                labels_data = pd.read_csv(labels_path)
            else:
                logger.error(f"No stage data found for {project_id}")
                return None
            
            if not features_data or labels_data is None:
                logger.error(f"Insufficient data for {project_id}")
                return None
            
            # Merge all features on pid
            merged_features = None
            column_map = {}
            
            for modality, df in features_data.items():
                if 'pid' in df.columns:
                    if merged_features is None:
                        merged_features = df.copy()
                    else:
                        merged_features = merged_features.merge(df, on='pid', how='inner')
                    
                    # Create column map
                    feature_columns = [col for col in df.columns if col != 'pid']
                    column_map[modality] = feature_columns
            
            if merged_features is None:
                logger.error(f"Failed to merge features for {project_id}")
                return None
            
            # Merge with labels
            if 'pid' in labels_data.columns:
                merged_features = merged_features.merge(labels_data, on='pid', how='inner')
            else:
                logger.error(f"No 'pid' column in labels for {project_id}")
                return None
            
            # Prepare features and labels
            features = merged_features.drop(columns=['pid']).astype(np.float32)
            labels = merged_features[['stage']].copy()
            
            # Create dataset
            dataset = CustomMultiModalDatasetStratified(
                features, labels, modalities, column_map
            )
            
            logger.info(f"Created multimodal dataset for {project_id} with {len(dataset)} samples")
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to create multimodal dataset for {project_id}: {e}")
            return None
    
    def get_available_projects(self) -> List[str]:
        """Get list of available TCGA projects."""
        return list(self.tcga_loader.TCGA_PROJECTS.keys())
    
    def get_project_summary(self, project_id: str) -> Dict:
        """Get summary of a TCGA project."""
        return self.tcga_loader.get_cohort_summary(project_id)


def integrate_tcga_with_existing_system(data_dir: str = "./data", tcga_data_dir: str = "./tcga_data"):
    """
    Convenience function to integrate TCGA data with existing system.
    
    Args:
        data_dir: Directory where existing datasets are stored
        tcga_data_dir: Directory containing TCGA data
        
    Returns:
        Dictionary with integration results
    """
    integration_manager = TCGAIntegrationManager(tcga_data_dir, data_dir)
    
    results = {}
    available_projects = integration_manager.get_available_projects()
    
    for project_id in available_projects:
        try:
            success = integration_manager.prepare_tcga_data_for_existing_structure(project_id)
            results[project_id] = {
                'success': success,
                'summary': integration_manager.get_project_summary(project_id) if success else None
            }
        except Exception as e:
            results[project_id] = {
                'success': False,
                'error': str(e)
            }
    
    return results


if __name__ == "__main__":
    # Example usage
    results = integrate_tcga_with_existing_system()
    logging.info("Integration results:", results)
