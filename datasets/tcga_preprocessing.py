"""
TCGA Data Preprocessing Module

This module provides advanced preprocessing functions for TCGA cohort data,
including normalization, feature selection, and data quality checks.
"""

import pandas as pd
import numpy as np
from typing import Dict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer, KNNImputer
import logging

logger = logging.getLogger(__name__)


class TCGADataPreprocessor:
    """
    Advanced data preprocessing for TCGA cohort data.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scalers = {}
        self.imputers = {}
        self.feature_selectors = {}
        self.label_encoders = {}
        
    def preprocess_clinical_data(self, 
                                clinical_df: pd.DataFrame,
                                normalize: bool = True,
                                handle_missing: str = 'mean') -> pd.DataFrame:
        """
        Preprocess clinical data.
        
        Args:
            clinical_df: Clinical data DataFrame
            normalize: Whether to normalize numeric features
            handle_missing: Strategy for handling missing values ('mean', 'median', 'knn', 'drop')
            
        Returns:
            Preprocessed clinical DataFrame
        """
        processed_df = clinical_df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df, handle_missing)
        
        # Normalize numeric features
        if normalize:
            processed_df = self._normalize_numeric_features(processed_df)
        
        # Encode categorical features
        processed_df = self._encode_categorical_features(processed_df)
        
        return processed_df
    
    def preprocess_mrna_data(self,
                            mrna_df: pd.DataFrame,
                            normalize: bool = True,
                            feature_selection: bool = True,
                            n_features: int = 1000,
                            log_transform: bool = True) -> pd.DataFrame:
        """
        Preprocess mRNA expression data.
        
        Args:
            mrna_df: mRNA expression DataFrame
            normalize: Whether to normalize features
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            log_transform: Whether to apply log transformation
            
        Returns:
            Preprocessed mRNA DataFrame
        """
        processed_df = mrna_df.copy()
        
        # Log transform if specified
        if log_transform:
            processed_df = self._apply_log_transform(processed_df)
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df, 'knn')
        
        # Feature selection
        if feature_selection:
            processed_df = self._select_features(processed_df, n_features)
        
        # Normalize features
        if normalize:
            processed_df = self._normalize_numeric_features(processed_df)
        
        return processed_df
    
    def preprocess_image_data(self,
                            image_df: pd.DataFrame,
                            normalize: bool = True,
                            feature_selection: bool = True,
                            n_features: int = 100) -> pd.DataFrame:
        """
        Preprocess image feature data.
        
        Args:
            image_df: Image features DataFrame
            normalize: Whether to normalize features
            feature_selection: Whether to perform feature selection
            n_features: Number of features to select
            
        Returns:
            Preprocessed image DataFrame
        """
        processed_df = image_df.copy()
        
        # Handle missing values
        processed_df = self._handle_missing_values(processed_df, 'mean')
        
        # Feature selection
        if feature_selection:
            processed_df = self._select_features(processed_df, n_features)
        
        # Normalize features
        if normalize:
            processed_df = self._normalize_numeric_features(processed_df)
        
        return processed_df
    
    def preprocess_stage_data(self,
                            stages_df: pd.DataFrame,
                            binary_classification: bool = True) -> pd.DataFrame:
        """
        Preprocess stage data.
        
        Args:
            stages_df: Stage data DataFrame
            binary_classification: Whether to convert to binary classification
            
        Returns:
            Preprocessed stage DataFrame
        """
        processed_df = stages_df.copy()
        
        # Clean stage values
        processed_df = self._clean_stage_values(processed_df)
        
        # Convert to binary classification if specified
        if binary_classification:
            processed_df = self._convert_to_binary_classification(processed_df)
        
        return processed_df
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Handle missing values in DataFrame."""
        if strategy == 'drop':
            return df.dropna()
        
        # Identify numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return df
        
        if strategy == 'mean':
            imputer = SimpleImputer(strategy='mean')
        elif strategy == 'median':
            imputer = SimpleImputer(strategy='median')
        elif strategy == 'knn':
            imputer = KNNImputer(n_neighbors=5)
        else:
            logger.warning(f"Unknown imputation strategy: {strategy}")
            return df
        
        # Apply imputation
        df_imputed = df.copy()
        df_imputed[numeric_columns] = imputer.fit_transform(df[numeric_columns])
        
        return df_imputed
    
    def _normalize_numeric_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize numeric features."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_columns:
            return df
        
        # Use StandardScaler for normalization
        scaler = StandardScaler()
        df_normalized = df.copy()
        df_normalized[numeric_columns] = scaler.fit_transform(df[numeric_columns])
        
        return df_normalized
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features."""
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        
        df_encoded = df.copy()
        
        for col in categorical_columns:
            if col != 'pid':  # Don't encode patient ID
                # Create dummy variables
                dummies = pd.get_dummies(df[col], prefix=col)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded.drop(columns=[col], inplace=True)
        
        return df_encoded
    
    def _apply_log_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transformation to numeric features."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        df_transformed = df.copy()
        
        for col in numeric_columns:
            if col != 'pid':  # Don't transform patient ID
                # Add small constant to avoid log(0)
                df_transformed[col] = np.log1p(df[col])
        
        return df_transformed
    
    def _select_features(self, df: pd.DataFrame, n_features: int) -> pd.DataFrame:
        """Select top features using statistical tests."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) <= n_features:
            return df
        
        # For feature selection, we need labels
        # This is a simplified version - in practice, you'd need actual labels
        X = df[numeric_columns].values
        
        # Create dummy labels for feature selection (in practice, use real labels)
        y = np.random.randint(0, 2, size=len(X))
        
        # Select features using mutual information
        selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        X_selected = selector.fit_transform(X, y)
        
        # Create new DataFrame with selected features
        selected_columns = [numeric_columns[i] for i in selector.get_support(indices=True)]
        df_selected = df[['pid'] + selected_columns].copy() if 'pid' in df.columns else df[selected_columns].copy()
        
        return df_selected
    
    def _clean_stage_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize stage values."""
        if 'stage' not in df.columns:
            return df
        
        df_cleaned = df.copy()
        
        # Standardize stage values
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
            'stage ivb': 'stage iv',
            'stage x': 'unknown',
            'not reported': 'unknown'
        }
        
        df_cleaned['stage'] = df_cleaned['stage'].str.lower().map(stage_mapping)
        df_cleaned['stage'] = df_cleaned['stage'].fillna('unknown')
        
        return df_cleaned
    
    def _convert_to_binary_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert stage data to binary classification (early vs late stage)."""
        if 'stage' not in df.columns:
            return df
        
        df_binary = df.copy()
        
        # Define early and late stages
        early_stages = ['stage i', 'stage ii']
        late_stages = ['stage iii', 'stage iv']
        
        def stage_to_binary(stage):
            if stage in early_stages:
                return 0  # Early stage
            elif stage in late_stages:
                return 1  # Late stage
            else:
                return -1  # Unknown
        
        df_binary['stage'] = df_binary['stage'].apply(stage_to_binary)
        
        # Remove unknown stages
        df_binary = df_binary[df_binary['stage'] != -1]
        
        return df_binary
    
    def quality_check(self, df: pd.DataFrame, data_type: str) -> Dict:
        """
        Perform quality checks on the data.
        
        Args:
            df: DataFrame to check
            data_type: Type of data ('clinical', 'mrna', 'image', 'stages')
            
        Returns:
            Dictionary with quality metrics
        """
        quality_metrics = {
            'data_type': data_type,
            'total_samples': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'duplicate_rows': df.duplicated().sum(),
            'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object']).columns)
        }
        
        # Data type specific checks
        if data_type == 'clinical':
            quality_metrics['age_range'] = {
                'min': df['age_at_diagnosis'].min() if 'age_at_diagnosis' in df.columns else None,
                'max': df['age_at_diagnosis'].max() if 'age_at_diagnosis' in df.columns else None,
                'mean': df['age_at_diagnosis'].mean() if 'age_at_diagnosis' in df.columns else None
            }
        
        elif data_type == 'stages':
            if 'stage' in df.columns:
                quality_metrics['stage_distribution'] = df['stage'].value_counts().to_dict()
        
        return quality_metrics
    
    def get_preprocessing_summary(self, 
                                 clinical_df: pd.DataFrame = None,
                                 mrna_df: pd.DataFrame = None,
                                 image_df: pd.DataFrame = None,
                                 stages_df: pd.DataFrame = None) -> Dict:
        """
        Get summary of preprocessing results.
        
        Args:
            clinical_df: Clinical data DataFrame
            mrna_df: mRNA data DataFrame  
            image_df: Image data DataFrame
            stages_df: Stage data DataFrame
            
        Returns:
            Dictionary with preprocessing summary
        """
        summary = {
            'preprocessing_timestamp': pd.Timestamp.now().isoformat(),
            'data_types_processed': [],
            'quality_metrics': {}
        }
        
        # Process each data type
        data_types = {
            'clinical': clinical_df,
            'mrna': mrna_df,
            'image': image_df,
            'stages': stages_df
        }
        
        for data_type, df in data_types.items():
            if df is not None:
                summary['data_types_processed'].append(data_type)
                summary['quality_metrics'][data_type] = self.quality_check(df, data_type)
        
        return summary


def preprocess_tcga_cohort(project_id: str, 
                          data_dir: str = "./tcga_data",
                          output_dir: str = "./processed_tcga_data") -> Dict:
    """
    Convenience function to preprocess TCGA cohort data.
    
    Args:
        project_id: TCGA project identifier
        data_dir: Directory containing raw TCGA data
        output_dir: Directory to save processed data
        
    Returns:
        Dictionary with preprocessing results
    """
    from pathlib import Path
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize preprocessor
    preprocessor = TCGADataPreprocessor()
    
    # Load data
    data_path = Path(data_dir) / project_id
    
    results = {
        'project_id': project_id,
        'preprocessing_success': True,
        'processed_files': [],
        'errors': []
    }
    
    # Process each data type
    data_types = ['clinical', 'mrna', 'image', 'stages']
    
    for data_type in data_types:
        file_path = data_path / f"{project_id}_{data_type}.csv"
        
        if file_path.exists():
            try:
                # Load data
                df = pd.read_csv(file_path)
                
                # Preprocess based on data type
                if data_type == 'clinical':
                    processed_df = preprocessor.preprocess_clinical_data(df)
                elif data_type == 'mrna':
                    processed_df = preprocessor.preprocess_mrna_data(df)
                elif data_type == 'image':
                    processed_df = preprocessor.preprocess_image_data(df)
                elif data_type == 'stages':
                    processed_df = preprocessor.preprocess_stage_data(df)
                
                # Save processed data
                output_file = output_path / f"{project_id}_{data_type}_processed.csv"
                processed_df.to_csv(output_file, index=False)
                
                results['processed_files'].append(str(output_file))
                logger.info(f"Processed {data_type} data for {project_id}")
                
            except Exception as e:
                error_msg = f"Failed to process {data_type} data for {project_id}: {e}"
                results['errors'].append(error_msg)
                logger.error(error_msg)
        else:
            logger.warning(f"No {data_type} data found for {project_id}")
    
    # Generate preprocessing summary
    if results['processed_files']:
        results['preprocessing_summary'] = preprocessor.get_preprocessing_summary()
    
    return results


if __name__ == "__main__":
    # Example usage
    results = preprocess_tcga_cohort('lihc')
    logging.info("Preprocessing results:", results)
