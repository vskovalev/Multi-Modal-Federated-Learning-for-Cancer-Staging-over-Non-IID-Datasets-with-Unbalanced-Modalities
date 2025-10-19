#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import numpy as np
import logging
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import models
from models.multimodal import CustomFederatedModel

def create_dummy_data(num_samples=100, num_features=20681):
    """Create dummy data for demonstration"""
    # Create random features (mrna + image features)
    features = torch.randn(num_samples, num_features)
    
    # Create random binary labels
    labels = torch.randint(0, 2, (num_samples,))
    
    # Convert to one-hot encoding
    labels_one_hot = torch.zeros(num_samples, 2)
    labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)
    
    return features, labels_one_hot

def create_column_map():
    """Create column mapping for modalities"""
    return {
        'mrna': list(range(20531)),  # mRNA features
        'image': list(range(20531, 20681))  # Image features
    }

def simulate_federated_training():
    """Simulate federated learning training process"""
    
    logging.info("Starting Multi-Modal Federated Learning Demo")
    
    # Create dummy data for 3 clients (simulating different cohorts)
    clients_data = []
    for i in range(3):
        features, labels = create_dummy_data(num_samples=50)
        dataset = TensorDataset(features, labels)
        clients_data.append(dataset)
        logging.info(f"Created client {i+1} with {len(dataset)} samples")
    
    # Create validation data
    val_features, val_labels = create_dummy_data(num_samples=30)
    val_dataset = TensorDataset(val_features, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=10, shuffle=False)
    
    # Create column map
    column_map = create_column_map()
    
    # Initialize global model
    modalities = ['mrna', 'image']
    global_model = CustomFederatedModel(modalities=modalities, column_map=column_map)
    
    # Initialize client models
    client_models = []
    client_optimizers = []
    client_loaders = []
    
    for i, dataset in enumerate(clients_data):
        model = CustomFederatedModel(modalities=modalities, column_map=column_map)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loader = DataLoader(dataset, batch_size=5, shuffle=True)
        
        client_models.append(model)
        client_optimizers.append(optimizer)
        client_loaders.append(loader)
    
    # Training parameters
    num_rounds = 5
    epochs_per_round = 2
    criterion = nn.CrossEntropyLoss()
    
    # Training history
    global_acc_history = []
    global_loss_history = []
    
    logging.info(f"Starting federated training for {num_rounds} rounds")
    
    for round_num in range(num_rounds):
        logging.info(f"\n=== Federated Round {round_num + 1} ===")
        
        # Local training on each client
        client_losses = []
        client_accs = []
        
        for client_id, (model, optimizer, loader) in enumerate(zip(client_models, client_optimizers, client_loaders)):
            logging.info(f"Training client {client_id + 1}")
            
            # Sync with global model
            model.load_state_dict(global_model.state_dict())
            
            # Local training
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for epoch in range(epochs_per_round):
                for batch_features, batch_labels in loader:
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
                    
                    # Backward pass
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    _, true_labels = torch.max(batch_labels, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == true_labels).sum().item()
            
            avg_loss = total_loss / len(loader)
            accuracy = 100 * correct / total
            client_losses.append(avg_loss)
            client_accs.append(accuracy)
            
            logging.info(f"Client {client_id + 1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        # Model aggregation (simple averaging)
        logging.info("Aggregating models...")
        with torch.no_grad():
            # Get state dicts from all clients
            client_state_dicts = [model.state_dict() for model in client_models]
            
            # Average the parameters
            averaged_state_dict = {}
            for key in client_state_dicts[0].keys():
                averaged_state_dict[key] = torch.stack([state_dict[key] for state_dict in client_state_dicts]).mean(0)
            
            # Update global model
            global_model.load_state_dict(averaged_state_dict)
        
        # Evaluate global model
        global_model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                outputs = global_model(val_features)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                _, true_labels = torch.max(val_labels, 1)
                val_total += val_labels.size(0)
                val_correct += (predicted == true_labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total
        
        global_acc_history.append(val_accuracy)
        global_loss_history.append(avg_val_loss)
        
        logging.info(f"Global Model - Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
        logging.info(f"Average Client Loss: {np.mean(client_losses):.4f}, Average Client Accuracy: {np.mean(client_accs):.2f}%")
    
    # Plot results
    logging.info("Generating training plots...")
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, num_rounds + 1), global_acc_history, 'b-o', label='Global Model')
    plt.xlabel('Federated Round')
    plt.ylabel('Validation Accuracy (%)')
    plt.title('Global Model Validation Accuracy')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, num_rounds + 1), global_loss_history, 'r-o', label='Global Model')
    plt.xlabel('Federated Round')
    plt.ylabel('Validation Loss')
    plt.title('Global Model Validation Loss')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/federated_training_demo.png', dpi=300, bbox_inches='tight')
    logging.info("Training plots saved to results/federated_training_demo.png")
    
    # Save final model
    torch.save(global_model.state_dict(), 'saved_models/demo_federated_model.pt')
    logging.info("Final model saved to saved_models/demo_federated_model.pt")
    
    logging.info("\n=== Training Summary ===")
    logging.info(f"Final Global Model Accuracy: {global_acc_history[-1]:.2f}%")
    logging.info(f"Final Global Model Loss: {global_loss_history[-1]:.4f}")
    logging.info(f"Accuracy Improvement: {global_acc_history[-1] - global_acc_history[0]:.2f}%")
    
    return global_model, global_acc_history, global_loss_history

def main():
    """Main function to run the federated learning demo"""
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('saved_models', exist_ok=True)
    
    try:
        # Run federated learning simulation
        global_model, acc_history, loss_history = simulate_federated_training()
        
        logging.info("\nMulti-Modal Federated Learning Demo Completed Successfully!")
        logging.info("This demonstration shows:")
        logging.info("1. Model initialization for different modality combinations")
        logging.info("2. Federated learning training process")
        logging.info("3. Model aggregation across clients")
        logging.info("4. Performance evaluation and visualization")
        logging.info("5. Model saving and persistence")
        
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise

if __name__ == "__main__":
    main()
