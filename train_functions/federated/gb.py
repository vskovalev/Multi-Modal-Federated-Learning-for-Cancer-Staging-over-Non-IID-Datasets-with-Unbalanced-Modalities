import numpy as np
from .utils import modality_to_classifier_mapper
from itertools import chain, combinations


## Function for loss dict initialization
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

def init_loss_dict(modalities, mode="average"):

    loss_dict = {}
    index_list = [x for x in range(len(modalities))]
    for comb in list(powerset(index_list)):
        if(len(comb) > 0):
            modality_subset = [modalities[x] for x in list(comb)]
            if mode=="per_client":
                loss_dict[modality_to_classifier_mapper(modality_subset)] = {}
            elif mode=="average":
                loss_dict[modality_to_classifier_mapper(modality_subset)] = []
    
    return loss_dict


## Average Losses
def avg_gb_losses(per_client_loss, averaged_loss):
    for modality in per_client_loss.keys():
        averaged_loss[modality].append(np.mean(list(per_client_loss[modality].values())))


## Function for overfitting calc
def calculate_overfitting(train_loss:list[float],
                          val_loss:list[float]):

    return (val_loss[-1] - train_loss[-1]) - (val_loss[-2] - train_loss[-2])


## Add function for generalization calc
def calculate_generalization(val_loss:list[float]):

    return val_loss[-1] - val_loss[-2]


## Add function for OGR calc
def calculate_ogr2(train_loss:list[float],
                   val_loss:list[float]):

    overfitting = calculate_overfitting(train_loss, val_loss)
    generalization = calculate_generalization(val_loss)

    return abs(generalization)/(abs(overfitting) ** 2)


## Calculate OGRs for all modality combinations
def calc_ogr2_systemwide(train_loss_dict, val_loss_dict):
    modality_ogrs = {}

    for modality in train_loss_dict.keys():
        modality_ogrs[modality] = calculate_ogr2(train_loss_dict[modality],
                                                 val_loss_dict[modality])
    
    return modality_ogrs

## Calculate weights for client based on its modalities (takes place on client)
def calculate_weights(modality_ogrs, modalities, modality_weights):

    ## Filter ogr dict based on available modalities on device
    if len(modalities)>1:
        filtered_modality_ogrs = {key:modality_ogrs[key] for key in modalities+[modality_to_classifier_mapper(modalities)]}
    else:
        filtered_modality_ogrs = {key:modality_ogrs[key] for key in modalities}
    
    ## Calculate weights based on filtered ogr dict values
    sum_ogrs = np.sum(np.asarray(list(filtered_modality_ogrs.values())))
    if len(modalities) > 1:
        for modality in filtered_modality_ogrs.keys():
            modality_weights[modality].append(filtered_modality_ogrs[modality] / (sum_ogrs))
    else:
        modality_weights[modalities[0]].append(filtered_modality_ogrs[modalities[0]] / (sum_ogrs))

## Add function for hook registration
def encoder_hooker(encoder_modality, gradient_weight):

    def hook(module, grad_input, grad_output):
        logging.info(f"encoder modality: {encoder_modality}")
        for key in module.state_dict().keys():
            layer_name, layer_attr = key.split(".")
            logging.info("layer name and attribute", layer_name, layer_attr)
            # if(encoder_modality=="mrna"):
            logging.info("grads (direct): ", getattr(getattr(module, layer_name), layer_attr).grad)
            logging.info("grads (indirect): ", grad_input)
            getattr(getattr(module, layer_name), layer_attr).grad *= gradient_weight
    
    return hook

def model_hooker(modality_list, gradient_weights):

    if len(modality_list)==1:

        def single_modal_hook(module, grad_input, grad_output):

            logging.info("hooking single modal client")

            logging.info(f"hooking classifier")
            for key in module.classifier.state_dict().keys():
                layer_name, layer_attr = key.split(".")
                getattr(getattr(module.classifier, layer_name), layer_attr).grad *= gradient_weights[modality_list[0]]
            
            logging.info(f"hooking {modality_list[0]} encoder")
            logging.info(getattr(module, f"{modality_list[0]}_encoder").state_dict().keys())
            for key in getattr(module, f"{modality_list[0]}_encoder").state_dict().keys():
                layer_name, layer_attr = key.split(".")
                logging.info("layer_name: ", layer_name, " layer_attr: ", layer_attr)
                getattr(getattr(getattr(module, f"{modality_list[0]}_encoder"), layer_name), layer_attr).grad *= gradient_weights[modality_list[0]]
            
        
        return single_modal_hook

    else:

        def multi_modal_hook(module, grad_input, grad_output):

            logging.info("hooking multi_modal client")

            logging.info("hooking classifier")
            for key in module.classifier.state_dict().keys():
                layer_name, layer_attr = key.split(".")
                logging.info(getattr(getattr(module.classifier, layer_name), layer_attr).grad)
                getattr(getattr(module.classifier, layer_name), layer_attr).grad *= gradient_weights[modality_to_classifier_mapper(modality_list)]
            
            for modality in modality_list:
                logging.info(f"hooking {modality} encoder")
                for key in getattr(module, f"{modality}_encoder").state_dict().keys():
                    layer_name, layer_attr = key.split(".")
                    getattr(getattr(getattr(module, f"{modality}_encoder"), layer_name), layer_attr).grad *= gradient_weights[modality]

        return multi_modal_hook




def debug_hook_maker(modality_list):

    def debug_hook(module, grad_input, grad_output):

        logging.info("mrna grads")
        for key in module.mrna_encoder.state_dict().keys():
            layer_name, layer_attr = key.split(".")
            logging.info(getattr(getattr(module.mrna_encoder, layer_name), layer_attr).grad)
        
        logging.info("classifier grads")
        for key in module.classifier.state_dict().keys():
            layer_name, layer_attr = key.split(".")
            logging.info(f"layer_name: {layer_name}, layer_attr: {layer_attr}")
            logging.info(getattr(getattr(module.classifier, layer_name), layer_attr).grad)
    
    return debug_hook


def debug_hook_applier(client):

    logging.info("hooking up client: \t", list(client['weight_dict'].keys()))
    debug_hook = debug_hook_maker(client['modalities'])
    client_handle = client['model'].register_full_backward_hook(debug_hook)
    client['handle'] = client_handle




def one_shot_hooker(client):

    
    logging.info("hooking up client: \t", list(client['weight_dict'].keys()))
    client_hook = model_hooker(client['modalities'], client['weight_dict'])
    client_handle = client['model'].register_full_backward_hook(client_hook)
    client['handle'] = client_handle



def client_hooker(client):

    client['handle_dict'] = {}
    logging.info("hooking up client: \t", list(client['weight_dict'].keys()))

    if len(client['modalities'])==1:
        modality = client['modalities'][0]
        logging.info("hooking classifier")
        classifier_hook = encoder_hooker(modality+"_classifier", client['weight_dict'][modality][-1])
        classifier_handle = client['model'].classifier.register_full_backward_hook(classifier_hook)
        client['handle_dict']['classifier'] = classifier_handle
        logging.info("hooking encoder")
        modality_hook = encoder_hooker(modality, client['weight_dict'][modality][-1])
        if modality=="mrna":
            modality_handle = client['model'].mrna_encoder.register_full_backward_hook(modality_hook)
        else:
            modality_handle = getattr(client['model'], modality+"_encoder").register_full_backward_hook(modality_hook)
        client['handle_dict'][modality] = modality_handle
    
    else:
        for modality in client['weight_dict'].keys():
            if modality not in client['modalities']:
                logging.info(f"hooking up {modality} classifier")
                classifier_hook = encoder_hooker(modality, client['weight_dict'][modality][-1])
                classifier_handle = client['model'].classifier.register_full_backward_hook(classifier_hook)
                client['handle_dict']['classifier'] = classifier_handle
            else:
                logging.info(f"hooking up {modality} encoder")
                modality_hook = encoder_hooker(modality, client['weight_dict'][modality][-1])
                modality_handle = getattr(client['model'], modality+"_encoder").register_full_backward_hook(modality_hook)
                client['handle_dict'][modality] = modality_handle

def unhook_client(client):

    for modality in client['handle_dict'].keys():
        client['handle_dict'][modality].remove()








