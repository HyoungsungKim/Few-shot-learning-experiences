import torch
import torch.nn as nn
import torch.autograd

from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import numpy as np
import task_generator as tg
import os
import models
from collections import OrderedDict

writer = SummaryWriter(logdir='scalar')

FEATURE_DIM = 64  # args.feature_dim
RELATION_DIM = 8  # args.relation_dim
CLASS_NUM = 5  # args.class_num
SAMPLE_NUM_PER_CLASS = 5  # args.sample_num_per_class
BATCH_NUM_PER_CLASS = 15  # args.batch_num_per_class
EPISODE = 5000000  # args.episode
TEST_EPISODE = 600  # args.test_episode
LEARNING_RATE = 0.001  # args.learning_rate
#LEARNING_RATE = 0.001
# GPU = # args.gpu
HIDDEN_UNIT = 10  # args.hidden_unit
        
        
def main():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = models.CNNEncoder()    
    relation_network = models.RelationNetwork(FEATURE_DIM, RELATION_DIM)
    
    feature_encoder.train()
    relation_network.train()       
    
    feature_encoder.apply(models.weights_init)
    relation_network.apply(models.weights_init)
    
    feature_encoder.to(device)
    relation_network.to(device)

    cross_entropy = nn.CrossEntropyLoss()
        
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=10000, gamma=0.5)
    
    relation_network_optim = torch.optim.Adam(relation_network.parameters(), lr=LEARNING_RATE)
    relation_network_scheduler = StepLR(relation_network_optim, step_size=10000, gamma=0.5)
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")            
        
    if os.path.exists(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        relation_network.load_state_dict(torch.load(str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load relation network success")
        
    # * Step 3: build graph
    print("Training...")
    
    last_accuracy = 0.0    
    loss_list = []
    inner_lr = 0.01
    number_of_query_image = 15
    for episode in range(EPISODE):
        #print(f"EPISODE : {episode}")
        task_feature_gradients = []
        task_relation_gradients = []
        task_losses = []
        task_predictions = []
        for meta_batch in range(5):
            #feature_fast_weights = OrderedDict(feature_encoder.named_parameters())
            relation_fast_weights = OrderedDict(relation_network.named_parameters())
            for inner_batch in range(5):
                task = tg.MiniImagenetTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)    
                
                samples, sample_labels = next(iter(sample_dataloader))
                batches, batch_labels = next(iter(batch_dataloader))
                
                samples, sample_labels = samples.to(device), sample_labels.to(device)
                batches, batch_labels = batches.to(device), batch_labels.to(device)
                
                inner_sample_features = feature_encoder(samples)            
                inner_sample_features = inner_sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                inner_sample_features = torch.sum(inner_sample_features, 1).squeeze(1)
                
                inner_batch_features = feature_encoder(batches)
                inner_sample_feature_ext = inner_sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
                inner_batch_features_ext = inner_batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)      
                inner_batch_features_ext = torch.transpose(inner_batch_features_ext, 0, 1)
                
                inner_relation_pairs = torch.cat((inner_sample_feature_ext, inner_batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                inner_relations = relation_network(inner_relation_pairs).view(-1, CLASS_NUM)
                
                # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
                inner_regularizer = inner_relations.pow(2).sum() / (2.0 * inner_relations.size(0))
        
                inner_loss = cross_entropy(inner_relations, batch_labels) + inner_regularizer
                #inner_feature_gradients = torch.autograd.grad(inner_loss, feature_fast_weights.values(), create_graph=True, allow_unused=True)
                inner_relation_gradients = torch.autograd.grad(inner_loss, relation_fast_weights.values(), create_graph=True, allow_unused=True)
                
                '''
                feature_fast_weights = OrderedDict(
                    (name, param - inner_lr * grad)
                    for ((name, param), grad) in zip(feature_fast_weights.items(), inner_feature_gradients)
                )
                '''        

                '''
                for ((name, param), grad) in zip(relation_fast_weights, inner_relation_gradients):                    
                    if grad is None:
                        grad = 0
                    relation_fast_weights[name] = param - inner_lr * grad              
                
                
                feature_fast_weights = OrderedDict(
                    (name, param - inner_lr * (0 if grad is None else grad))                    
                    for ((name, param), grad) in zip(feature_fast_weights.items(), inner_feature_gradients)                    
                )  
                '''
                relation_fast_weights = OrderedDict(
                    (name, param - inner_lr * (0 if grad is None else grad))                    
                    for ((name, param), grad) in zip(relation_fast_weights.items(), inner_relation_gradients)                    
                )        
                
            # * init dataset
            # * sample_dataloader is to obtain previous samples for compare
            # * batch_dataloader is to batch samples for training
            task = tg.MiniImagenetTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
            sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
            batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)
            # * num_per_class : number of query images
            
            # * sample datas
            samples, sample_labels = next(iter(sample_dataloader))
            batches, batch_labels = next(iter(batch_dataloader))
            
            samples, sample_labels = samples.to(device), sample_labels.to(device)
            batches, batch_labels = batches.to(device), batch_labels.to(device)
                            
            # * calculates features
            #feature_encoder.weight = feature_fast_weights
            
            sample_features = feature_encoder(samples)
            sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
            sample_features = torch.sum(sample_features, 1).squeeze(1)
            batch_features = feature_encoder(batches)
            
            # * calculate relations
            # * each batch sample link to every samples to calculate relations
            # * to form a 100 * 128 matrix for relation network
            sample_features_ext = sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
            batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
            
            relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)   
            
            relation_network.weight = relation_fast_weights
            
            relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
            #one_hot_labels = torch.zeros(BATCH_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).to(device).scatter_(1, batch_labels.view(-1, 1), 1)
            
            # * https://github.com/facebookresearch/low-shot-shrink-hallucinate/blob/master/losses.py
            regularizer = relations.pow(2).sum() / (2.0 * relations.size(0))
                
            loss = cross_entropy(relations, batch_labels) + regularizer                            
            #loss.backward(retain_graph=True)
            #y_pred = relations.softmax(dim=1)
            #feature_gradients = torch.autograd.grad(loss, feature_fast_weights.values(), create_graph=True, allow_unused=True)
            relation_gradients = torch.autograd.grad(loss, relation_fast_weights.values(), create_graph=True, allow_unused=True)
            
            #feature_name_grads = {name: g for ((name, _), g) in zip(feature_fast_weights.items(), feature_gradients)}
            relation_name_grads = {name: g for ((name, _), g) in zip(relation_fast_weights.items(), relation_gradients)}
            
            #task_predictions.append(y_pred)            
            task_losses.append(loss)       
            #task_feature_gradients.append(feature_name_grads)     
            task_relation_gradients.append(relation_name_grads)
            
        feature_encoder_optim.zero_grad()
        relation_network_optim.zero_grad()     
        
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(relation_network.parameters(), 0.5)
        
        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_loss.backward()
                
        feature_encoder_optim.step()
        relation_network_optim.step()
                
        feature_encoder_scheduler.step(episode)
        relation_network_scheduler.step(episode)
        
        if (episode + 1) % 100 == 0:
            print(f"episode : {episode+1}, meta_batch_loss : {meta_batch_loss.cpu().detach().numpy()}")
            loss_list.append(meta_batch_loss.cpu().detach().numpy())
            
        if (episode + 1) % 500 == 0:
            print("Testing...")
            total_reward = 0
            
            for i in range(TEST_EPISODE):
                number_of_query_image = 15
                task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                    
                # * calculate features
                sample_features = feature_encoder(sample_images)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                test_features = feature_encoder(test_images)
                
                # * calculate relations
                # * each batch sample link to every samples to calculate relations
                # * to form a 100x128 matrix for relation network
                
                sample_features_ext = sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                relations = relation_network(relation_pairs).view(-1, CLASS_NUM)
                
                _, predict_labels = torch.max(relations.data, 1)
                
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(CLASS_NUM * SAMPLE_NUM_PER_CLASS)]
                total_reward += np.sum(rewards)
                
            test_accuracy = total_reward / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS * TEST_EPISODE)
            # print("test accuracy : ", test_accuracy)
            mean_loss = np.mean(loss_list)
            
            print(f'mean loss : {mean_loss}')   
            writer.add_scalar('loss', mean_loss, episode + 1)            
            writer.add_scalar('test accuracy', test_accuracy, episode + 1)
            
            loss_list = []            
            print("test accuracy : ", test_accuracy)
            
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(
                    feature_encoder.state_dict(),
                    str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                torch.save(
                    relation_network.state_dict(),
                    str("./models/miniimagenet_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy    
    
            
if __name__ == "__main__":
    main()
