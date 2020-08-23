import torch
import numpy as np
import task_generator as tg
import models
import os
import copy
import torch.nn as nn
import scipy as sp
import scipy.stats
from collections import OrderedDict

FEATURE_DIM = 64  # args.feature_dim
RELATION_DIM = 8  # args.relation_dim
CLASS_NUM = 5  # args.class_num
SAMPLE_NUM_PER_CLASS = 5  # args.sample_num_per_class
BATCH_NUM_PER_CLASS = 15  # args.batch_num_per_class
EPISODE = 1000000  # args.episode
TEST_EPISODE = 600  # args.test_episode
LEARNING_RATE = 0.001  # args.learning_rate
# GPU = # args.gpu
HIDDEN_UNIT = 10  # args.hidden_unit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    # * Step 1: init data folders
    print("init data folders")
    
    # * init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = models.CNNEncoder().to(device)    
    relation_network = models.RelationNetwork(FEATURE_DIM, RELATION_DIM).to(device)

    #feature_encoder.eval()
    #relation_network.eval()

    # * https://discuss.pytorch.org/t/performance-highly-degraded-when-eval-is-activated-in-the-test-phase/3323/16    
    '''
    for child in feature_encoder.children():
        for ii in range(len(child)):
            if type(child[ii]) == nn.BatchNorm2d:
                child[ii].track_running_stats = False
                
    for child in relation_network.children():
        for ii in range(len(child)):
            if type(child[ii]) == nn.BatchNorm2d:
                child[ii].track_running_stats = False
    '''
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")

    if os.path.exists(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")

    total_accuracy = 0.0    
    result_list = []
    cross_entropy = nn.CrossEntropyLoss()
    inner_lr = 0.01
    for test_iter in range(5):
        print(f"Testing {test_iter+1}th...")       
        max_accuracy = 0
        total_accuracy = []
        number_of_query_image = 15
        for i in range(TEST_EPISODE):
            total_reward = 0
            test_feature_encoder = copy.deepcopy(feature_encoder)
            test_relation_network = copy.deepcopy(relation_network)
            relation_fast_weights = OrderedDict(test_relation_network.named_parameters())

            for inner_batch in range(5):
                inner_num_of_query_image = 5
                task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS * 6, inner_num_of_query_image)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=inner_num_of_query_image, split="train", shuffle=True)    
                
                samples, sample_labels = next(iter(sample_dataloader))
                batches, batch_labels = next(iter(batch_dataloader))
                # * len(5 shot 5 way) * 3 == len(5 classes with 15 queries)
                
                samples, sample_labels = samples.to(device), sample_labels.to(device)
                batches, batch_labels = batches.to(device), batch_labels.to(device)
                
                inner_sample_features = test_feature_encoder(samples)            
                inner_sample_features = inner_sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                inner_sample_features = torch.sum(inner_sample_features, 1).squeeze(1)
                
                inner_batch_features = test_feature_encoder(batches)
                inner_sample_feature_ext = inner_sample_features.unsqueeze(0).repeat(inner_num_of_query_image * CLASS_NUM, 1, 1, 1, 1)
                inner_batch_features_ext = inner_batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)      
                inner_batch_features_ext = torch.transpose(inner_batch_features_ext, 0, 1)
                
                inner_relation_pairs = torch.cat((inner_sample_feature_ext, inner_batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                inner_relations = test_relation_network(inner_relation_pairs).view(-1, CLASS_NUM)
                
                # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
                inner_regularizer = inner_relations.pow(2).sum() / (2.0 * inner_relations.size(0))
        
                inner_loss = cross_entropy(inner_relations, batch_labels) + inner_regularizer
                inner_relation_gradients = torch.autograd.grad(inner_loss, relation_fast_weights.values(), create_graph=True, allow_unused=True)
                
                relation_fast_weights = OrderedDict(
                    (name, param - inner_lr * (0 if grad is None else grad))                    
                    for ((name, param), grad) in zip(relation_fast_weights.items(), inner_relation_gradients)                    
                )        
                
            # * init dataset
            # * sample_dataloader is to obtain previous samples for compare
            # * batch_dataloader is to batch samples for training
            task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS * 6, number_of_query_image)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
            batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)
            # * num_per_class : number of query images
            
            # * sample datas
            samples, sample_labels = next(iter(sample_dataloader))
            batches, batch_labels = next(iter(batch_dataloader))
            
            samples, sample_labels = samples.to(device), sample_labels.to(device)
            batches, batch_labels = batches.to(device), batch_labels.to(device)
            
            sample_features = test_feature_encoder(samples)
            sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            batch_features = test_feature_encoder(batches)
            
            # * calculate relations
            # * each batch sample link to every samples to calculate relations
            # * to form a 100 * 128 matrix for relation network
            sample_features_ext = sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            
            test_relation_network.weight = relation_fast_weights
            
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)   
            relations = test_relation_network(relation_pairs).view(-1, CLASS_NUM)
            
            _, predict_labels = torch.max(relations.data, 1)
            rewards = [1 if predict_labels[j] == batch_labels[j] else 0 for j in range(len(batch_labels))] # CLASS_NUM * number_of_query_image
            total_reward += np.sum(rewards)
            
            test_accuracy = total_reward / len(batch_labels) # (1.0 * CLASS_NUM * number_of_query_image)
            #print(test_accuracy)
            total_accuracy.append(test_accuracy)
            if test_accuracy > max_accuracy:
                max_accuracy = test_accuracy
            
        mean_accuracy, conf_int = mean_confidence_interval(total_accuracy)        
        print(f"Total accuracy : {mean_accuracy:.4f}")
        print(f"confidence interval : {conf_int:.4f}")
        #print(f"max accuracy : {max_accuracy:.4f}")                  
        result = [mean_accuracy, conf_int]  
        result_list.append(result)
        
    sorted_list = sorted(result_list, reverse=True)
    print(*sorted_list, sep='\n')
    
    
if __name__ == "__main__":
    main()
