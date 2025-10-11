import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from copy import deepcopy
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score
from datetime import datetime

from datasets.multimodal_federated import create_datasets_fed_stratified
from models import CustomFederatedDistributedAttentionModel
from fed_utils import average_encoder_weights, modality_to_classifier_mapper, classifier_to_modality_mapper, random_average_attention_weights


BATCH_SIZE = 5
NUM_WORKERS = 2
# NUM_EPOCHS = {('brca', 1):1, ('brca', 2): 3, ('brca', 3): 3, ('brca', 4): 3, ('brca', 5): 3, ('brca', 6): 6, ('brca', 7): 2,
#               ('lusc', 1): 2, ('lusc', 2): 2, ('lusc', 3): 2, ('lihc', 1): 2, ('lihc', 2): 2, ('lihc', 3): 2}

NUM_EPOCHS = {('brca', 1): 2, ('brca', 2): 2, ('brca', 3): 2, ('lusc', 1): 2, ('lusc', 2): 2, ('lusc', 3): 2, ('lihc', 1): 2, ('lihc', 2): 2, ('lihc', 3): 2}
# NUM_EPOCHS = {('brca', 1): 1, ('brca', 2): 1, ('brca', 3): 1, ('lusc', 1): 1, ('lusc', 2): 1, ('lusc', 3): 1, ('lihc', 1): 1, ('lihc', 2): 1, ('lihc', 3): 1}
SHUFFLE_DATASET = True
RANDOM_SEED = 42
ACC_USED = False
NUM_FED_LOOPS = 100
# COHORTS = ["brca"]
COHORTS = ["brca", "lusc", "lihc"]
DATA_PATH = os.path.join("..", "data", "multi_modal_features", 'may_19_2023')
INIT_LR = 1e-6
LR_DECAY_RATE = 0.9
STEPS_PER_DECAY = 7
MODE = 'bi_modal_rand'
INIT_REG = 1e-2
STOP_CRITERIA = 15

def main():

    if not os.path.exists(os.path.join("..","results","mm_dist_attention",MODE)):
        os.mkdir(os.path.join("..","results","mm_dist_attention",MODE))

    path_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    SAVE_DIR = os.path.join("..","results","mm_dist_attention",MODE,path_time)
    os.mkdir(SAVE_DIR)

    if ACC_USED:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Building Network
    network = {}
    network['clients'] = []
    network['validation_datasets'] = []
    network['global_valid_loss_memory'] = []
    network['global_valid_acc_memory'] = []
    network['weighted_sum_of_losses'] = []
    
    # Building Global Model for Network
    

    ## Generate Datasets
    for cohort in COHORTS:
        datasets = create_datasets_fed_stratified(cohort, DATA_PATH, random_state=RANDOM_SEED)
        for clientbuildnum in range(len(datasets)):
            if clientbuildnum==0:
                logging.info(f"Test dataset for cohort {cohort} added.")
                network['validation_datasets'].append(datasets[clientbuildnum])
            else:
                logging.info(f"Creating client {clientbuildnum} for cohort {cohort}")
                client = {}
                client['cohort_id'] = (cohort, clientbuildnum)
                client['dataset'] = datasets[clientbuildnum]
                logging.info(f"len dataset = {len(client['dataset'])}")
                client['modalities'] = client['dataset'].modalities
                client['dataloader'] = DataLoader(client['dataset'], batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET)
                client['train_loss_memory'] = []
                client['valid_loss_memory'] = []
                client['train_acc_memory'] = []
                client['valid_acc_memory'] = []
                client['valid_loss_min'] = np.inf
                client['total_steps_taken'] = 0
                client['dataset_size'] = len(client['dataset'])
                client['model'] = CustomFederatedDistributedAttentionModel(client['dataset'].modalities, client['dataset'].column_map)
                if ACC_USED:
                    client['model'].to(device)
                if LR_DECAY_RATE!=None:
                    client['optimizer'] = optim.SGD(client['model'].parameters(), lr=INIT_LR)
                    client['scheduler'] = optim.lr_scheduler.ExponentialLR(client['optimizer'], gamma=LR_DECAY_RATE)
                else:
                    client['optimizer'] = optim.Adam(client['model'].parameters(), lr=INIT_LR)
                network['clients'].append(client)
    
    network['validation_dataloaders'] = []
    network['validation_len'] = 0

    # model = CustomFederatedDistributedAttentionModel(modalities=["mrna", "image", "clinical"], column_map=network['validation_datasets'][0].column_map)
    network['global_model'] = CustomFederatedDistributedAttentionModel(modalities=["mrna", "image"], column_map=network['validation_datasets'][0].column_map)
    # model = CustomFederatedDistributedAttentionModel(modalities=["mrna"], column_map=network['validation_datasets'][0].column_map)

    if ACC_USED:
        network['global_model'].to(device)
    
    # model.load_state_dict(torch.load('../saved_models/federated_attention_mrna_image_clinical_start_model.pt'))
    network['global_model'].load_state_dict(torch.load('../saved_models/federated_attention_mrna_image_start_model.pt'))

    for val_set in network['validation_datasets']:
        network['validation_dataloaders'].append(DataLoader(val_set, batch_size=1))
        network['validation_len'] += len(network['validation_dataloaders'][-1])
    
    network['dataset_size'] = 0
    network['client_dataset_sizes'] = {}
    for client in network['clients']:
        network['dataset_size'] += client['dataset_size'] * NUM_EPOCHS[client['cohort_id']]
        network['client_dataset_sizes'][client['cohort_id']] = client['dataset_size'] * NUM_EPOCHS[client['cohort_id']]
    

    ## For early stopping ##
    network['steps_without_improvements'] = 0
    network['gamma'] = INIT_REG
    

    ### Initializing network's modality-based classifiers ###
    # network['global_classifiers'] = {"mrna":{}, "image":{}, "clinical":{}, "image_clinical":{}, "mrna_image":{},
    #                                  "mrna_clinical":{}, "mrna_image_clinical":{}}
    network['global_classifiers'] = {"mrna":{}, "image":{}, "mrna_image":{}}
    # network['global_classifiers'] = {"mrna_image":{}}

    for classifier_pair in network['global_classifiers'].keys():
        classifier_model_path = '../saved_models/federated_attention_'+classifier_pair+'_start_model.pt'
        dummy_model = CustomFederatedDistributedAttentionModel(modalities=classifier_to_modality_mapper(classifier_pair), column_map=network['validation_datasets'][0].column_map)
        dummy_model.load_state_dict(torch.load(classifier_model_path))
        network['global_classifiers'][classifier_pair] = deepcopy(dummy_model.classifier)
    

    
    # Building Training Loop

    network['global_model_min_loss'] = np.inf

    for fed_round in range(NUM_FED_LOOPS):
        network['round_wsl'] = 0
        logging.info(f"\n \n### Start of federated round {fed_round+1} ###")

        ### Trained Encoder Dict ###
        # network['trained_encoders'] = {'mrna':{}, 'image':{}, "clinical":{}}
        network['trained_encoders'] = {'mrna':{}, 'image':{}}
        # network['trained_encoders'] = {'mrna':{}}


        ### Trained Classifier Dict ###
        # network['trained_classifiers'] = {"mrna":{}, "image":{}, "clinical":{}, "image_clinical":{}, "mrna_image":{},
        #                                   "mrna_clinical":{}, "mrna_image_clinical":{}}
        network['trained_classifiers'] = {"mrna":{}, "image":{}, "mrna_image":{}}
        # network['trained_classifiers'] = {"mrna_image":{}}

        ### Trained Attention Dict ###
        # network['trained_attentions'] = {"mrna_image":{}, "mrna_clinical":{}, "image_clinical":{}}
        network['trained_attentions'] = {"mrna_image":{}}

        

        for client in network['clients']:

            logging.info(f"Training client {client['cohort_id']}")
            
            ## Loading Global Model Weights in the Beginning of Federated Loop
            # model = client['model']
            with torch.no_grad():

                ## Syncing encoders w/ global model
                for modality in client['model'].modalities:
                    getattr(client['model'], modality+"_encoder").load_state_dict(getattr(network['global_model'], modality+"_encoder").state_dict())
                
                ## Syncing attention blocks w/ global model
                if len(client['modalities']) == 3:
                    client['model'].mrna_image_attention.load_state_dict(network['global_model'].mrna_image_attention.state_dict())
                    client['model'].mrna_clinical_attention.load_state_dict(network['global_model'].mrna_clinical_attention.state_dict())
                    client['model'].image_clinical_attention.load_state_dict(network['global_model'].image_clinical_attention.state_dict())
                elif len(client['modalities']) == 2:
                    getattr(client['model'], client['modalities'][0]+"_"+client['modalities'][1]+"_attention").load_state_dict(getattr(network['global_model'], client['modalities'][0]+"_"+client['modalities'][1]+"_attention").state_dict())
                
                ## Syncing classifier w/ corresponding version in global dict
                client['model'].classifier.load_state_dict(network['global_classifiers'][modality_to_classifier_mapper(client['modalities'])].state_dict())
                # if((fed_round==1) and (client['cohort_id'] == ('brca', 1))):
                    
                    # for (key, value) in model.encoders[modality].state_dict().items():
                    #     if modality == 'mrna':
                    #         if key == 'fc1.weight':
                    #             logging.info(value)
            # model.load_state_dict(network['global_model'].state_dict())
            criterion = nn.CrossEntropyLoss()
            # optimizer = client['optimizer']
            # if LR_DECAY_RATE != None:
            #     scheduler = client['scheduler']
            # valid_loss_min = np.Inf
            # temp_val_loss = []
            # temp_val_acc = []
            total_steps = len(client['dataloader'])

            running_loss = 0.0
            correct = 0
            total = 0

            

            for epoch in range(NUM_EPOCHS[client['cohort_id']]):
                # running_loss = 0.0
                
                logging.info(f'Epoch {epoch}\n')

                # correct = 0
                # total = 0

                ## Start training

                for batch_idx, (data_, target_) in enumerate(client['dataloader']):

                    client['model'].train()

                    if batch_idx < 10:    
                        if ACC_USED:
                            data_ = data_.to(device)
                            target_ = target_.to(device)
                        
                        ### Zero Our Parameter Gradients
                        client['optimizer'].zero_grad()

                        ### FWD + BKWD + OPTIM
                        outputs = client['model'](data_)
                        # logging.info(outputs)
                        # raise ValueError
                        # logging.info(outputs)
                        loss = criterion(outputs, target_)
                        loss.backward()
                        client['optimizer'].step()

                        ### logging.info Stats
                        running_loss += loss.item()
                        _, pred = torch.max(outputs, dim=1)
                        _, target_label = torch.max(target_, dim=1)
                        correct += torch.sum(pred==target_label).item()
                        total += target_.size(0)
                        # if(batch_idx) == 3:
                        #     logging.info('Epoch [{}/{}], Step [{}/{}], loss: {:.4f}'.format(epoch, NUM_EPOCHS, batch_idx, total_steps, loss.item()))
                        
                        client['total_steps_taken'] += 1

                        ## Scheduler step
                        if((LR_DECAY_RATE is not None) and(client['total_steps_taken'] % STEPS_PER_DECAY == 0)):
                            client['scheduler'].step()
                    
                    client['train_acc_memory'].append(100*correct/total)
                    client['train_loss_memory'].append(running_loss/len(client['dataloader']))
                            
                            
                    # if batch_idx%2==1:
                    #     logging.info(f'Epoch [{epoch}/{NUM_EPOCHS}], Step [{batch_idx}/{total_steps}]: Accuracy = {step_accuracy}, Loss = {step_loss}')
                    
                    # temp_train_acc.append(100*correct/total)
                    # temp_train_loss.append(running_loss/total_steps)
                    # logging.info(f'\ntrain loss: {np.mean(train_loss):.4f}, train acc: {(100 * correct / total):.4f}')


                    ## Validate Model ##

                    with torch.no_grad():
                        batch_loss = 0
                        total_t = 0
                        correct_t = 0
                        cm_pred = np.array([])
                        cm_target = np.array([])
                        client['model'].eval()
                        for val_loader in network['validation_dataloaders']:
                            for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                                if ACC_USED:
                                    data_t = data_t.to(device)  
                                    target_t = target_t.to(device)
                                outputs_t = client['model'](data_t)
                                loss_t = criterion(outputs_t, target_t)
                                batch_loss += loss_t.item()
                                _, pred_t = torch.max(outputs_t, dim=1)
                                _, target_t_label = torch.max(target_t, dim=1)
                                correct_t += torch.sum(pred_t==target_t_label).item()
                                total_t += target_t.size(0)
                                cm_pred = np.append(cm_pred, pred_t.numpy(force=True))
                                cm_target = np.append(cm_target, target_t_label.numpy(force=True))
                            
                        client['valid_acc_memory'].append(100*correct_t/total_t)
                        client['valid_loss_memory'].append(batch_loss/network['validation_len'])
                        network_learned = client['valid_loss_memory'][-1] < client['valid_loss_min']
                        # logging.info(f"validation loss: {np.mean(batch_loss/total_t):.4f}, validation acc: {(100 * correct_t / total_t):.4f}\n")

                        

                        if network_learned:
                            client['valid_loss_min'] = client['valid_loss_memory'][-1]
                            model_save_dir = os.path.join(SAVE_DIR, f"{client['cohort_id']}.pt")
                            torch.save(client['model'].state_dict(), model_save_dir)
                            logging.info('Saving current model due to improvement')
            
            # logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS[client['cohort_id']]}], Step [{batch_idx}/{total_steps}],\ntrain loss: {client['train_loss_memory'][-1]:.4f}, train acc: {client['train_acc_memory'][-1]:.4f} \n \
            #         validation loss: {client['valid_loss_memory'][-1]:.4f}, validation acc: {client['valid_acc_memory'][-1]:.4f}\n \
            #         f-1 score: {f1_score(cm_target, cm_pred, average='macro')}")
            logging.info(f"Fed round [{fed_round}],\nTrain loss: {client['train_loss_memory'][-1]:.4f}, train acc: {client['train_acc_memory'][-1]:.4f} \n \
                    validation loss: {client['valid_loss_memory'][-1]:.4f}, validation acc: {client['valid_acc_memory'][-1]:.4f}\n \
                    f-1 score: {f1_score(cm_target, cm_pred, average='macro')}")
            
            
            # Updating the WSL using client
            network['round_wsl'] += network['client_dataset_sizes'][client['cohort_id']] * client['valid_loss_memory'][-1] / network['dataset_size']
            
            # client['model'] = model
            
            for modality in client['modalities']:
                network['trained_encoders'][modality][client['cohort_id']] = getattr(client['model'], modality+"_encoder")
            # network[modality_to_classifier_mapper(client['modalities'])].append({'classifier':client['model'].classifier, 'dataset_size':client['dataset_size']})
            
            network['trained_classifiers'][modality_to_classifier_mapper(client['modalities'])][client['cohort_id']] = client['model'].classifier

            if len(client['modalities']) == 3:
                network['trained_attentions']['mrna_image'][client['cohort_id']] = client['model'].mrna_image_attention
                network['trained_attentions']['mrna_clinical'][client['cohort_id']] = client['model'].mrna_clinical_attention
                network['trained_attentions']['image_clinical'][client['cohort_id']] = client['model'].image_clinical_attention
            elif len(client['modalities']) == 2:
                network['trained_attentions'][client['modalities'][0]+"_"+client['modalities'][1]][client['cohort_id']] = getattr(client['model'], client['modalities'][0]+"_"+client['modalities'][1]+"_attention")
        
        network['weighted_sum_of_losses'].append(network['round_wsl'])
        ## Aggregate models
        
        logging.info("keys for each encoder")
        for key in network['trained_encoders'].keys():
            logging.info(key)
            logging.info(network['trained_encoders'][key].keys())

        with torch.no_grad():
            for modality in network['global_model'].modalities:
                logging.info(f"aggregating {modality} encoders")
                getattr(network['global_model'], modality+"_encoder").load_state_dict(average_encoder_weights(getattr(network['global_model'], modality+"_encoder"), network['trained_encoders'][modality], network['client_dataset_sizes'], modality, device).state_dict())
            
            logging.info("aggregating classifiers")
            for classifier_pair in network['trained_classifiers'].keys():
                network['global_classifiers'][classifier_pair].load_state_dict(average_encoder_weights(network['global_classifiers'][classifier_pair], network['trained_classifiers'][classifier_pair], network['client_dataset_sizes'], classifier_pair, device).state_dict())
            
            for attention_pair in network['trained_attentions'].keys():
                logging.info(f"aggregating {attention_pair} attention modules")
                getattr(network['global_model'], attention_pair+"_attention").load_state_dict(random_average_attention_weights(getattr(network['global_model'], attention_pair+"_attention"), network['trained_attentions'][attention_pair], network['client_dataset_sizes'], attention_pair, device).state_dict())

            # network['global_model'].classifier.load_state_dict(network['global_classifiers']['mrna_image_clinical'].state_dict())
            network['global_model'].classifier.load_state_dict(network['global_classifiers']['mrna_image'].state_dict())


        # eval_model = network['global_model']
        eval_batch_loss = 0
        eval_total_t = 0
        eval_correct_t = 0
        with torch.no_grad():
            network['global_model'].eval()
            for val_loader in network['validation_dataloaders']:
                for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                    if ACC_USED:
                        data_t = data_t.to(device)
                        target_t.to(device)
                    outputs_t = network['global_model'](data_t)
                    loss_t = criterion(outputs_t, target_t)
                    eval_batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    _, target_t_label = torch.max(target_t, dim=1)
                    eval_correct_t += torch.sum(pred_t==target_t_label).item()
                    eval_total_t += target_t.size(0)
                
            network['global_valid_acc_memory'].append(100*eval_correct_t/eval_total_t)
            network['global_valid_loss_memory'].append(eval_batch_loss/network['validation_len'])
            logging.info(f"current round global model loss: {eval_batch_loss/network['validation_len']} \t min global model loss {network['global_model_min_loss']}")

            
            if eval_batch_loss/network['validation_len'] < network['global_model_min_loss']:
                network['global_model_min_loss'] = eval_batch_loss/network['validation_len']
                network['steps_without_improvements'] = 0
                logging.info("improvement, saving model")
                eval_model_save_dir = os.path.join(SAVE_DIR, f"best_global.pt")
                torch.save(network['global_model'].state_dict(), eval_model_save_dir)
            else:
                network['steps_without_improvements'] += 1
                logging.info(f"{network['steps_without_improvements']} steps with no improvement.")
                
            if network['steps_without_improvements'] == STOP_CRITERIA:
                logging.info("no improvement in 5 rounds. halting training.")
                break
    
    ### Saving Results ###

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['global_valid_acc_memory']))]), network['global_valid_acc_memory'])
    plt.title("Global Model Accuracy")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_global_acc.png")
    plt.savefig(os.path.join(SAVE_DIR, f"acc_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.png"))
    val_acc_save = np.asarray(network['global_valid_acc_memory'])
    np.savetxt(os.path.join(SAVE_DIR, f"acc_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.csv"), val_acc_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['global_valid_loss_memory']))]), network['global_valid_loss_memory'])
    plt.title("Global Model Loss")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_global_loss.png")
    plt.savefig(os.path.join(SAVE_DIR, f"loss_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.png"))
    val_loss_save = np.asarray(network['global_valid_loss_memory'])
    np.savetxt(os.path.join(SAVE_DIR, f"loss_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.csv"), val_loss_save, delimiter=",")

    plt.figure()
    plt.plot(np.array([x for x in range(len(network['weighted_sum_of_losses']))]), network['weighted_sum_of_losses'])
    plt.title("Weighted Sum of Losses")
    # plt.savefig("../results/mm_no_attention/bi_modal_4_network_wsl.png")
    plt.savefig(os.path.join(SAVE_DIR, f"wsl_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.png"))
    val_wsl_save = np.asarray(network['weighted_sum_of_losses'])
    np.savetxt(os.path.join(SAVE_DIR, f"wsl_lr_{INIT_LR}_gamma_{LR_DECAY_RATE}_every{STEPS_PER_DECAY}steps.csv"), val_wsl_save, delimiter=",")

    client_plot_dir = os.path.join(SAVE_DIR, "client_plots")
    os.mkdir(client_plot_dir)

    kirekhar = []
    for client in network['clients']:

        client['model'].eval()
        # model = client['model']
        with torch.no_grad():

            
            for modality in client['model'].modalities:
                    # logging.info(torch.all(client['model'].encoders[modality].state_dict()['fc1.weight'].eq(network['global_model'].encoders[modality].state_dict()['fc1.weight'])))
                getattr(client['model'], modality+"_encoder").load_state_dict(getattr(network['global_model'], modality+"_encoder").state_dict())
                
                
            client['model'].classifier.load_state_dict(network['global_classifiers'][modality_to_classifier_mapper(client['modalities'])].state_dict())

            if len(client['modalities']) == 3:
                client['model'].mrna_image_attention.load_state_dict(network['global_model'].mrna_image_attention.state_dict())
                client['model'].mrna_clinical_attention.load_state_dict(network['global_model'].mrna_clinical_attention.state_dict())
                client['model'].image_clinical_attention.load_state_dict(network['global_model'].image_clinical_attention.state_dict())
            elif len(client['modalities']) == 2:
                getattr(client['model'], client['modalities'][0]+"_"+client['modalities'][1]+"_attention").load_state_dict(getattr(network['global_model'], client['modalities'][0]+"_"+client['modalities'][1]+"_attention").state_dict())
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #     kirekhar.append(client['model'].classifier.state_dict()['4.weight'])
            #     if len(kirekhar) > 1:
            #         for kirekhar_counter in range(len(kirekhar)-1):
            #             logging.info(torch.all(client['model'].classifier.state_dict()['4.weight'].eq(kirekhar[kirekhar_counter])))
        
            correct_t = 0
            total_t = 0
            batch_loss = 0  
            cm_pred = np.array([])
            cm_target = np.array([])
            # logging.info(client['model'].encoders['mrna'].dropout.training)
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #         logging.info(torch.all(client['model'].encoders['mrna'].state_dict()['fc1.weight'].eq(network['global_model'].encoders['mrna'].state_dict()['fc1.weight'])))
            for val_loader in network['validation_dataloaders']:
                for val_batch_idx, (data_t, target_t) in enumerate(val_loader):

                    if ACC_USED:
                        data_t = data_t.to(device)  
                        target_t = target_t.to(device)
                    
                    # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
                    #     logging.info(torch.all(client['model'].classifier.state_dict()['4.weight'].eq(network['global_classifiers'][modality_to_classifier_mapper(client['modalities'])].state_dict()['4.weight'])))
                        
                    outputs_t = client['model'](data_t)
                    loss_t = criterion(outputs_t, target_t)
                    batch_loss += loss_t.item()
                    _, pred_t = torch.max(outputs_t, dim=1)
                    _, target_t_label = torch.max(target_t, dim=1)
                    correct_t += torch.sum(pred_t==target_t_label).item()
                    total_t += target_t.size(0)
                    cm_pred = np.append(cm_pred, pred_t.numpy(force=True))
                    cm_target = np.append(cm_target, target_t_label.numpy(force=True))
            
            # if (modality_to_classifier_mapper(client['model'].modalities) == 'mrna_image'):
            #     logging.info(cm_pred)
            #     logging.info(cm_target)
            final_f1_score = f1_score(cm_target, cm_pred, average='macro')
            logging.info(f"Final model f1-score: {final_f1_score}")
            cm = confusion_matrix(cm_target, cm_pred, labels=[0,1])
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
            disp.plot()
            plt.savefig(os.path.join(client_plot_dir, f"cm_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))

        plt.figure()
        plt.plot(np.array([x for x in range(len(client['valid_acc_memory']))]), client['valid_acc_memory'])
        plt.plot(np.array([x for x in range(len(client['train_acc_memory']))]), client['train_acc_memory'])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title(f"Train vs. Validation Accuracy for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
        plt.legend(['Validaiton', 'Train'])
        plt.savefig(os.path.join(client_plot_dir, f"acc_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))

        plt.figure()
        plt.plot(np.array([x for x in range(len(client['valid_loss_memory']))]), client['valid_loss_memory'])
        plt.plot(np.array([x for x in range(len(client['train_loss_memory']))]), client['train_loss_memory'])
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('CrossEntropy Loss')
        plt.title(f"Train vs. Validation Loss for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
        plt.legend(['Validaiton', 'Train'])
        plt.savefig(os.path.join(client_plot_dir, f"loss_{client['cohort_id'][0]}_{client['cohort_id'][1]}_{modality_to_classifier_mapper(client['modalities'])}.png"))
    
    # plt.figure()
    # for client in network['clients']:
    #     plt.plot(np.array([x for x in range(len(client['valid_acc_memory']))]), client['valid_acc_memory'], label=f"{client['cohort_id'][0]}_{client['cohort_id'][1]}")
    # plt.grid(True)
    # plt.xlabel('Epochs')
    # plt.ylabel('CrossEntropy Loss')
    # plt.title(f"Train vs. Validation Accuracy for client {client['cohort_id'][0]}, {client['cohort_id'][1]}")
    # plt.legend(['Validaiton', 'Train'])
    # plt.savefig(os.path.join(client_plot_dir, f"{client['cohort_id'][0]}_{client['cohort_id'][1]}.png"))
        



    

        
        
main()