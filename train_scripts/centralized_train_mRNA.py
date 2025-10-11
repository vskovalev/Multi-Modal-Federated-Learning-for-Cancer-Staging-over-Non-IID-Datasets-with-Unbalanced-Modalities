import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datetime import datetime

from datasets.mrna import create_mrna_train_test_strat
from models.mrna import CustomRNAModelMedium, CustomRNAModelBIG, CustomRNAModelSmall
from mrna_arg_parser import parse_arguments



##### Centralized Training Procedure #####

def main():

    args = parse_arguments()

    path_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = os.path.join("..","results","mrna", path_time+"_"+args.model_type)
    os.mkdir(SAVE_DIR)

    # BATCH_SIZE = 64
    # NUM_EPOCHS = 10
    # NUM_FEATURES = 20531
    # NUM_CLASSES = 2
    # VALIDATION_SPLIT = 0.2
    # SHUFFLE_DATASET = True
    # RANDOM_SEED = 42
    # MPS_USED = True
    # LEARNING_RATE = 5e-5
    # DECAY_RATE = 0.9
    # DECAY_STEPS = 7

    # Setting the device to Apple Silicon
    if args.acc_used:
        device = torch.device("mps")
    
    # Create model
    if args.model_type == 'large':
        model = CustomRNAModelBIG(args.num_features, args.num_classes)
        model.load_state_dict(torch.load(f'../saved_models/mrna_start_model_{args.model_type}.pt')) 
    elif args.model_type == 'medium':
        model = CustomRNAModelMedium(args.num_features, args.num_classes)
        model.load_state_dict(torch.load(f'../saved_models/mrna_start_model_{args.model_type}.pt'))
    elif args.model_type == 'small':
        model = CustomRNAModelSmall(args.num_features, args.num_classes)
        model.load_state_dict(torch.load(f'../saved_models/mrna_start_model_{args.model_type}.pt'))
    else:
        logging.info("## model type not supported ##")
        raise ValueError
    
    if args.acc_used:
        model.to(device)
    logging.info("Model created")

    # Create dataset
    train_dataset, valid_dataset = create_mrna_train_test_strat(args.validation_split, args.data_path, args.random_seed)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True)
    logging.info("Dataloaders created")

    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader)
    optimizer = optim.SGD(model.parameters(), lr=args.init_lr)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    valid_loss_min = np.Inf
    val_loss = []
    val_acc = []
    train_loss = []
    train_acc = []
    total_steps_taken = 0

    logging.info("Train loop variables created")

    for epoch in range(args.num_epochs):

        running_loss = 0.0
        

        logging.info(f'Epoch {epoch+1}\n')
        
        for batch_idx, (data_, target_) in enumerate(train_loader):

            model.train()

            correct = 0
            total = 0

            # data_ /= 1e3
            # logging.info(data_[:,2:8])
            # logging.info(target_)
            # data_ = data_.unsqueeze(1)

            if args.acc_used:
                data_ = data_.to(device)
                target_ = target_.to(device)

            ### Fwd pass
            outputs = model(data_)

            # logging.info(outputs)
            # logging.info(target_)
            # raise "kirekhar"
            
            ### Gradient calc
            loss = criterion(outputs, target_)
            loss.backward()
            optimizer.step()

            ### logging.info stats
            running_loss += loss.item()
            _, pred = torch.max(outputs, dim=1)
            _, target_label = torch.max(target_, dim=1)
            correct += torch.sum(pred==target_label).item()
            total += target_.size(0)
            
            
            if total_steps_taken%args.steps_per_decay==0:
                scheduler.step()
                    
            train_acc.append(100*correct/total)
            train_loss.append(loss.item())


            ## Validate Model Accuracy

            batch_loss = 0
            total_t = 0
            correct_t = 0


            with torch.no_grad():
                model.eval()
                for data_t, target_t in (valid_loader):

                    # data_t = data_t.unsqueeze(1)

                    ### To device
                    if args.acc_used:
                        data_t = data_t.to(device)
                        target_t = target_t.to(device)

                    ### Fwd pass
                    outputs_t = model(data_t)

                    ### logging.info Stats
                    loss_t = criterion(outputs_t, target_t)
                    # logging.info(loss_t.item())
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    _, target_t_label = torch.max(target_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t_label).item()
                    total_t += target_t.size(0)
                
                val_acc.append(100*correct_t/total_t)
                val_loss.append(batch_loss/len(valid_loader))
                network_learned = batch_loss < valid_loss_min

                if (batch_idx % 2 == 0):
                    logging.info(f'Epoch [{epoch+1}/{args.num_epochs}], Step [{batch_idx}/{total_steps}],\ntrain loss: {train_loss[-1]:.4f}, train acc: {(100 * correct / total):.4f} \
                        \nvalidation loss: {np.mean(val_loss):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n')

                    

                if network_learned:
                    valid_loss_min = batch_loss
                    torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'classification_model_mrna_centralized.pt'))
                    logging.info('Saving current model due to improvement')
            
            # model.train()

            ### Zero gradients
            optimizer.zero_grad()
            total_steps_taken += 1
    
    
    plt.figure()
    plt.plot(np.array([x for x in range(len(train_loss))]), train_loss, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_loss))]), val_loss, color='g', label='Validation')
    plt.title("mRNA Loss for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Crossentropy Loss')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"loss_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    # plt.savefig(f"results/centralized/loss_type_{args.model_type}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    plt.figure()
    plt.plot(np.array([x for x in range(len(train_acc))]), train_acc, color='r', label='Train')
    plt.plot(np.array([x for x in range(len(val_acc))]), val_acc, color='g', label='Validation')
    plt.title("mRNA Accuracy for Centralized")
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, f"acc_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
    # plt.savefig(f"results/centralized/acc_type_{args.model_type}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    with torch.no_grad():
        cm_pred = []
        cm_target = []
        for data_cm, target_cm in (valid_loader):

            # data_t = data_t.unsqueeze(1)

            ### To device
            if args.acc_used:
                data_cm = data_cm.to(device)
                target_cm = target_cm.to(device)

            ### Fwd pass
            outputs_cm = model(data_cm)

            _, pred_cm = torch.max(outputs_cm, dim=1)
            _, target_cm_label = torch.max(target_cm, dim=1)
            
            cm_pred.append(pred_cm.numpy(force=True)[:])
            cm_target.append(target_cm_label.numpy(force=True)[:])
        
        logging.info(len(cm_target))
        cm = confusion_matrix(cm_target, cm_pred, labels=[0,1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
        disp.plot()
        plt.savefig(os.path.join(SAVE_DIR, f"cm_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps.png"))
        # plt.savefig(f"results/centralized/cm_type_{args.model_type}_lr_{args.init_lr}_gamma_{args.lr_decay_rate}_every{args.steps_per_decay}steps_mRNA_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")

    


        
main()