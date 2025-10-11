"""
TCGA Cohort Loader Module

This module provides functionality to load and process TCGA cohort data from various sources.
Supports TCGA-LIHC, TCGA-LUSC, and TCGA-BRCA projects.
"""

import pandas as pd
import requests
import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TCGACohortLoader:
    """
    A class to load and process TCGA cohort data from various sources.
    
    This class provides functionality to:
    - Download data from TCGA GDC API
    - Process and clean TCGA data
    - Integrate with existing multimodal dataset structure
    - Support multiple TCGA projects (LIHC, LUSC, BRCA)
    """
    
    # TCGA project mappings
    TCGA_PROJECTS = {
        'lihc': 'TCGA-LIHC',
        'lusc': 'TCGA-LUSC', 
        'brca': 'TCGA-BRCA'
    }
    
    # GDC API endpoints
    GDC_API_BASE = "https://api.gdc.cancer.gov"
    GDC_FILES_ENDPOINT = f"{GDC_API_BASE}/files"
    GDC_CASES_ENDPOINT = f"{GDC_API_BASE}/cases"
    GDC_DATA_ENDPOINT = f"{GDC_API_BASE}/data"
    
    def __init__(self, data_dir: str = "./tcga_data", cache_dir: str = "./tcga_cache"):
        """
        Initialize TCGA Cohort Loader.
        
        Args:
            data_dir: Directory to store processed TCGA data
            cache_dir: Directory to store cached API responses
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        
        # Create directories if they don't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize data storage
        self.cohort_data = {}
        self.metadata = {}
        
    def get_project_info(self, project_id: str) -> Dict:
        """
        Get information about a TCGA project.
        
        Args:
            project_id: TCGA project identifier (e.g., 'lihc', 'lusc', 'brca')
            
        Returns:
            Dictionary containing project information
        """
        if project_id not in self.TCGA_PROJECTS:
            raise ValueError(f"Unknown project: {project_id}. Available: {list(self.TCGA_PROJECTS.keys())}")
            
        tcga_project = self.TCGA_PROJECTS[project_id]
        
        # Check cache first
        cache_file = self.cache_dir / f"{project_id}_project_info.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        # Query GDC API
        try:
            params = {
                'filters': json.dumps({
                    "op": "in",
                    "content": {
                        "field": "cases.submitter_id",
                        "value": [tcga_project]
                    }
                }),
                'expand': 'diagnoses,demographic,exposures',
                'size': 10000
            }
            
            response = requests.get(self.GDC_CASES_ENDPOINT, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Cache the response
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
                
            logger.info(f"Retrieved project info for {tcga_project}")
            return data
            
        except requests.RequestException as e:
            logger.error(f"Failed to retrieve project info for {tcga_project}: {e}")
            raise
    
    def get_available_data_types(self, project_id: str) -> List[str]:
        """
        Get available data types for a TCGA project.
        
        Args:
            project_id: TCGA project identifier
            
        Returns:
            List of available data types
        """
        project_info = self.get_project_info(project_id)
        
        data_types = set()
        for case in project_info.get('data', []):
            for file in case.get('files', []):
                data_types.add(file.get('data_type', ''))
        
        return list(filter(None, data_types))
    
    def download_cohort_data(self, project_id: str, data_types: List[str] = None) -> Dict:
        """
        Download cohort data for a specific TCGA project.
        
        Args:
            project_id: TCGA project identifier
            data_types: List of data types to download (e.g., ['Clinical', 'Gene Expression'])
            
        Returns:
            Dictionary containing downloaded data
        """
        if project_id not in self.TCGA_PROJECTS:
            raise ValueError(f"Unknown project: {project_id}")
            
        tcga_project = self.TCGA_PROJECTS[project_id]
        
        # Default data types if not specified
        if data_types is None:
            data_types = ['Clinical', 'Gene Expression Quantification', 'Slide Image']
        
        logger.info(f"Downloading data for {tcga_project} with types: {data_types}")
        
        # Query for files
        filters = {
            "op": "and",
            "content": [
                {
                    "op": "in",
                    "content": {
                        "field": "cases.project.project_id",
                        "value": [tcga_project]
                    }
                },
                {
                    "op": "in", 
                    "content": {
                        "field": "data_type",
                        "value": data_types
                    }
                }
            ]
        }
        
        params = {
            'filters': json.dumps(filters),
            'expand': 'cases',
            'size': 10000
        }
        
        try:
            response = requests.get(self.GDC_FILES_ENDPOINT, params=params)
            response.raise_for_status()
            
            files_data = response.json()
            
            # Process and organize the data
            cohort_data = self._process_downloaded_data(files_data, project_id)
            
            # Save processed data
            self._save_cohort_data(cohort_data, project_id)
            
            return cohort_data
            
        except requests.RequestException as e:
            logger.error(f"Failed to download data for {tcga_project}: {e}")
            raise
    
    def _process_downloaded_data(self, files_data: Dict, project_id: str) -> Dict:
        """
        Process downloaded TCGA data into structured format.
        
        Args:
            files_data: Raw data from GDC API
            project_id: TCGA project identifier
            
        Returns:
            Processed cohort data
        """
        processed_data = {
            'clinical': [],
            'mrna': [],
            'image': [],
            'stages': []
        }
        
        for file_info in files_data.get('data', []):
            case_id = file_info.get('cases', [{}])[0].get('submitter_id', '')
            data_type = file_info.get('data_type', '')
            file_id = file_info.get('file_id', '')
            
            # Process different data types
            if 'Clinical' in data_type:
                clinical_data = self._extract_clinical_data(file_info)
                if clinical_data:
                    processed_data['clinical'].append(clinical_data)
                    
            elif 'Gene Expression' in data_type:
                mrna_data = self._extract_mrna_data(file_info)
                if mrna_data:
                    processed_data['mrna'].append(mrna_data)
                    
            elif 'Slide Image' in data_type:
                image_data = self._extract_image_data(file_info)
                if image_data:
                    processed_data['image'].append(image_data)
        
        # Extract stage information from clinical data
        processed_data['stages'] = self._extract_stage_data(processed_data['clinical'])
        
        return processed_data
    
    def _extract_clinical_data(self, file_info: Dict) -> Optional[Dict]:
        """Extract clinical data from file information."""
        try:
            case_info = file_info.get('cases', [{}])[0]
            
            clinical_data = {
                'pid': case_info.get('submitter_id', ''),
                'age_at_diagnosis': case_info.get('diagnoses', [{}])[0].get('age_at_diagnosis', None),
                'gender': case_info.get('demographic', {}).get('gender', ''),
                'race': case_info.get('demographic', {}).get('race', ''),
                'ethnicity': case_info.get('demographic', {}).get('ethnicity', ''),
                'vital_status': case_info.get('demographic', {}).get('vital_status', ''),
                'days_to_death': case_info.get('demographic', {}).get('days_to_death', None),
                'days_to_last_follow_up': case_info.get('demographic', {}).get('days_to_last_follow_up', None),
                'tumor_stage': case_info.get('diagnoses', [{}])[0].get('tumor_stage', ''),
                'tumor_grade': case_info.get('diagnoses', [{}])[0].get('tumor_grade', ''),
                'primary_diagnosis': case_info.get('diagnoses', [{}])[0].get('primary_diagnosis', '')
            }
            
            return clinical_data
            
        except Exception as e:
            logger.warning(f"Failed to extract clinical data: {e}")
            return None
    
    def _extract_mrna_data(self, file_info: Dict) -> Optional[Dict]:
        """Extract mRNA expression data from file information."""
        try:
            case_info = file_info.get('cases', [{}])[0]
            
            mrna_data = {
                'pid': case_info.get('submitter_id', ''),
                'file_id': file_info.get('file_id', ''),
                'data_type': file_info.get('data_type', ''),
                'file_size': file_info.get('file_size', 0),
                'experimental_strategy': file_info.get('experimental_strategy', '')
            }
            
            return mrna_data
            
        except Exception as e:
            logger.warning(f"Failed to extract mRNA data: {e}")
            return None
    
    def _extract_image_data(self, file_info: Dict) -> Optional[Dict]:
        """Extract image data from file information."""
        try:
            case_info = file_info.get('cases', [{}])[0]
            
            image_data = {
                'pid': case_info.get('submitter_id', ''),
                'file_id': file_info.get('file_id', ''),
                'data_type': file_info.get('data_type', ''),
                'file_size': file_info.get('file_size', 0),
                'data_format': file_info.get('data_format', '')
            }
            
            return image_data
            
        except Exception as e:
            logger.warning(f"Failed to extract image data: {e}")
            return None
    
    def _extract_stage_data(self, clinical_data: List[Dict]) -> List[Dict]:
        """Extract stage information from clinical data."""
        stages = []
        
        for clinical in clinical_data:
            stage_info = {
                'pid': clinical.get('pid', ''),
                'stage': clinical.get('tumor_stage', ''),
                'grade': clinical.get('tumor_grade', '')
            }
            stages.append(stage_info)
        
        return stages
    
    def _save_cohort_data(self, cohort_data: Dict, project_id: str):
        """Save processed cohort data to files."""
        project_dir = self.data_dir / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Save clinical data
        if cohort_data['clinical']:
            clinical_df = pd.DataFrame(cohort_data['clinical'])
            clinical_df.to_csv(project_dir / f"{project_id}_clinical.csv", index=False)
            logger.info(f"Saved clinical data for {project_id}: {len(clinical_df)} records")
        
        # Save mRNA data
        if cohort_data['mrna']:
            mrna_df = pd.DataFrame(cohort_data['mrna'])
            mrna_df.to_csv(project_dir / f"{project_id}_mrna.csv", index=False)
            logger.info(f"Saved mRNA data for {project_id}: {len(mrna_df)} records")
        
        # Save image data
        if cohort_data['image']:
            image_df = pd.DataFrame(cohort_data['image'])
            image_df.to_csv(project_dir / f"{project_id}_image.csv", index=False)
            logger.info(f"Saved image data for {project_id}: {len(image_df)} records")
        
        # Save stage data
        if cohort_data['stages']:
            stages_df = pd.DataFrame(cohort_data['stages'])
            stages_df.to_csv(project_dir / f"{project_id}_stages.csv", index=False)
            logger.info(f"Saved stage data for {project_id}: {len(stages_df)} records")
    
    def load_cohort_from_files(self, project_id: str) -> Dict[str, pd.DataFrame]:
        """
        Load cohort data from previously saved files.
        
        Args:
            project_id: TCGA project identifier
            
        Returns:
            Dictionary containing loaded DataFrames
        """
        project_dir = self.data_dir / project_id
        
        if not project_dir.exists():
            raise FileNotFoundError(f"No data directory found for {project_id}")
        
        cohort_data = {}
        
        # Load different data types
        data_types = ['clinical', 'mrna', 'image', 'stages']
        
        for data_type in data_types:
            file_path = project_dir / f"{project_id}_{data_type}.csv"
            if file_path.exists():
                cohort_data[data_type] = pd.read_csv(file_path)
                logger.info(f"Loaded {data_type} data for {project_id}: {len(cohort_data[data_type])} records")
            else:
                logger.warning(f"No {data_type} data found for {project_id}")
        
        return cohort_data
    
    def get_cohort_summary(self, project_id: str) -> Dict:
        """
        Get summary statistics for a cohort.
        
        Args:
            project_id: TCGA project identifier
            
        Returns:
            Dictionary containing summary statistics
        """
        try:
            cohort_data = self.load_cohort_from_files(project_id)
            
            summary = {
                'project_id': project_id,
                'tcga_project': self.TCGA_PROJECTS[project_id],
                'total_cases': 0,
                'data_types_available': list(cohort_data.keys()),
                'statistics': {}
            }
            
            # Calculate statistics for each data type
            for data_type, df in cohort_data.items():
                summary['statistics'][data_type] = {
                    'count': len(df),
                    'columns': list(df.columns),
                    'missing_values': df.isnull().sum().to_dict()
                }
                
                if data_type == 'clinical' and 'age_at_diagnosis' in df.columns:
                    summary['statistics'][data_type]['age_stats'] = {
                        'mean': df['age_at_diagnosis'].mean(),
                        'std': df['age_at_diagnosis'].std(),
                        'min': df['age_at_diagnosis'].min(),
                        'max': df['age_at_diagnosis'].max()
                    }
                
                if data_type == 'stages' and 'stage' in df.columns:
                    summary['statistics'][data_type]['stage_distribution'] = df['stage'].value_counts().to_dict()
            
            # Calculate total unique cases
            all_pids = set()
            for df in cohort_data.values():
                if 'pid' in df.columns:
                    all_pids.update(df['pid'].unique())
            
            summary['total_cases'] = len(all_pids)
            
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate summary for {project_id}: {e}")
            raise
    
    def integrate_with_existing_structure(self, project_id: str, target_dir: str) -> bool:
        """
        Integrate TCGA cohort data with existing dataset structure.
        
        Args:
            project_id: TCGA project identifier
            target_dir: Directory where existing datasets are stored
            
        Returns:
            True if integration successful
        """
        try:
            cohort_data = self.load_cohort_from_files(project_id)
            target_path = Path(target_dir)
            
            # Create project-specific files in target directory
            for data_type, df in cohort_data.items():
                if data_type == 'stages':
                    # Save stages data
                    df.to_csv(target_path / f"{project_id}_stages.csv", index=False)
                else:
                    # Save other data types
                    df.to_csv(target_path / f"{project_id}_{data_type}.csv", index=False)
            
            logger.info(f"Successfully integrated {project_id} data into {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to integrate {project_id} data: {e}")
            return False


def main():
    """Example usage of TCGA Cohort Loader."""
    
    # Initialize loader
    loader = TCGACohortLoader()
    
    # Example: Load data for LIHC project
    project_id = 'lihc'
    
    try:
        # Get project information
        project_info = loader.get_project_info(project_id)
        logging.info(f"Project info retrieved for {project_id}")
        
        # Get available data types
        data_types = loader.get_available_data_types(project_id)
        logging.info(f"Available data types: {data_types}")
        
        # Download cohort data (this would take a while for real data)
        # cohort_data = loader.download_cohort_data(project_id, data_types[:3])
        
        # Load from files if they exist
        try:
            cohort_data = loader.load_cohort_from_files(project_id)
            logging.info(f"Loaded cohort data for {project_id}")
            
            # Get summary
            summary = loader.get_cohort_summary(project_id)
            logging.info(f"Cohort summary: {summary}")
            
        except FileNotFoundError:
            logging.info(f"No existing data found for {project_id}")
        
    except Exception as e:
        logging.info(f"Error: {e}")


if __name__ == "__main__":
    main()
