#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import models directly
from models.multimodal import (
    CustomRNAModel, CustomImgModel, CustomClinicalModel, 
    CustomMultiModalModel, CustomRNAImgModel, CustomRNAClinModel, 
    CustomImgClinModel, CustomFederatedModel
)

def modality_to_classifier_mapper(modalities):
    if("mrna" in modalities):
        if("image" in modalities):
            if("clinical" in modalities):
                return "mrna_image_clinical"
            else:
                return "mrna_image"
        elif("clinical" in modalities):
            return "mrna_clinical"
        else:
            return "mrna"
    elif("image" in modalities):
        if("clinical" in modalities):
            return "image_clinical"
        else:
            return "image"
    elif("clinical" in modalities):
        return "clinical"
    else:
        raise ValueError

def model_assigner(modalities):
    if modality_to_classifier_mapper(modalities) == 'mrna_image_clinical':
        return CustomMultiModalModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna_image':
        return CustomRNAImgModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna_clinical':
        return CustomRNAClinModel()
    elif modality_to_classifier_mapper(modalities) == 'image_clinical':
        return CustomImgClinModel()
    elif modality_to_classifier_mapper(modalities) == 'mrna':
        return CustomRNAModel()
    elif modality_to_classifier_mapper(modalities) == 'image':
        return CustomImgModel()
    elif modality_to_classifier_mapper(modalities) == 'clinical':
        return CustomClinicalModel()
    else:
        logging.info("Wrong modalities. Model cannot be created.")
        raise KeyError

def main():
    # Create saved_models directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)
    
    # List of modality combinations
    modality_list = [
        ["mrna"], 
        ["image"], 
        ["clinical"], 
        ["mrna", "image"], 
        ["mrna", "clinical"], 
        ["image", "clinical"], 
        ["mrna", "image", "clinical"]
    ]
    
    MODE = 'no_attention'
    
    if MODE == 'no_attention':
        for modalities in modality_list:
            try:
                model = model_assigner(modalities)
                classifier_name = modality_to_classifier_mapper(modalities)
                model_path = f"saved_models/federated_{classifier_name}_start_model.pt"
                torch.save(model.state_dict(), model_path)
                logging.info(f"Created model: {classifier_name} -> {model_path}")
            except Exception as e:
                logging.error(f"Failed to create model for {modalities}: {e}")
    
    logging.info("Model initialization completed!")

if __name__ == "__main__":
    main()
