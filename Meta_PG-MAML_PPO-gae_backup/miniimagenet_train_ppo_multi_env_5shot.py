import torch
import torch.nn as nn
import torch.autograd

from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import numpy as np
import task_generator as tg
import os
import models
import ppoAgent_gae_refactoring as ppoAgent
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
HIDDEN_UNIT = 10  # args.hidden_unit
GAMMA = 0.9
ENTROPY_WEIGHT = 1e-2
ENV_LENGTH = 5
META_ENV_LENGTH = 3
INNER_BATCH_RANGE = 3
META_BATCH_RANGE = 3
CLIP_DECREASE = 10000


def main():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = models.CNNEncoder()    
    model = models.ActorCritic(FEATURE_DIM, RELATION_DIM, CLASS_NUM)

    #feature_encoder = torch.nn.DataParallel(feature_encoder)
    #actor = torch.nn.DataParallel(actor)
    #critic = torch.nn.DataParallel(critic)
    
    feature_encoder.train()
    model.train()
    
    feature_encoder.apply(models.weights_init)
    model.apply(models.weights_init)
    
    feature_encoder.to(device)
    model.to(device)

    cross_entropy = nn.CrossEntropyLoss()
        
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=10000, gamma=0.5)
    
    model_optim = torch.optim.Adam(model.parameters(), lr=2.5 * LEARNING_RATE)
    model_scheduler = StepLR(model_optim, step_size=10000, gamma=0.5)
    
    agent = ppoAgent.PPOAgent(model, model_optim, GAMMA, ENTROPY_WEIGHT, CLASS_NUM, device)
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")            
        
    if os.path.exists(str("./models/miniimagenet_actor_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        model.load_state_dict(torch.load(str("./models/miniimagenet_actor_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load model network success")
        
    # * Step 3: build graph
    print("Training...")
    loss_list = []
    last_accuracy = 0.0    
    number_of_query_image = 15
    clip_param = 0.1
    for episode in range(EPISODE):
        if clip_param > 0 and clip_param % CLIP_DECREASE == 0:
            clip_param *= 0.5
            
        #print(f"EPISODE : {episode}")
        losses = []        
        for meta_batch in range(META_BATCH_RANGE):
            meta_env_states_list = []
            meta_env_labels_list = []
            for inner_batch in range(INNER_BATCH_RANGE):
                # * Generate environment
                env_states_list = []
                env_labels_list = []
                for _ in range(ENV_LENGTH):
                    task = tg.MiniImagenetTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                    sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                    batch_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=5, split="test", shuffle=True)    
                    
                    samples, sample_labels = next(iter(sample_dataloader))
                    samples, sample_labels = samples.to(device), sample_labels.to(device)
                    for batches, batch_labels in batch_dataloader:
                        batches, batch_labels = batches.to(device), batch_labels.to(device)
                        
                        inner_sample_features = feature_encoder(samples)            
                        inner_sample_features = inner_sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 19, 19)
                        inner_sample_features = torch.sum(inner_sample_features, 1).squeeze(1)
                        
                        inner_batch_features = feature_encoder(batches)
                        inner_sample_feature_ext = inner_sample_features.unsqueeze(0).repeat(5 * CLASS_NUM, 1, 1, 1, 1)
                        inner_batch_features_ext = inner_batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)      
                        inner_batch_features_ext = torch.transpose(inner_batch_features_ext, 0, 1)
                        
                        inner_relation_pairs = torch.cat((inner_sample_feature_ext, inner_batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 19, 19)
                        env_states_list.append(inner_relation_pairs)
                        env_labels_list.append(batch_labels)
                
                inner_env = ppoAgent.env(env_states_list, env_labels_list)
                agent.train(inner_env, inner_update=True)
            
            # * Generate env for meta update
            for _ in range(META_ENV_LENGTH):
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
                
                meta_env_states_list.append(relation_pairs)
                meta_env_labels_list.append(batch_labels)
            
            meta_env = ppoAgent.env(meta_env_states_list, meta_env_labels_list)
            agent.train(meta_env, loss_list=losses, clip_param=clip_param)
            
        feature_encoder_optim.zero_grad()
        #model_optim.zero_grad()     
        
        torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)

        meta_batch_loss = torch.stack(losses).mean()
        #meta_batch_loss.backward()
                
        feature_encoder_optim.step()
        #model_optim.step()

        feature_encoder_scheduler.step()
        model_scheduler.step()
        
        mean_loss = None
        if (episode + 1) % 100 == 0:
            mean_loss = meta_batch_loss.cpu().detach().numpy()
            print(f"episode : {episode+1}, meta_loss : {mean_loss:.4f}")
            loss_list.append(mean_loss)
            
        if (episode + 1) % 500 == 0:
            print("Testing...")
            total_reward = 0
            
            total_test_samples = 0            
            for i in range(TEST_EPISODE):
                # * Generate env
                env_states_list = []
                env_labels_list = []
                number_of_query_image = 10
                task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                total_test_samples += len(test_labels)

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
                env_states_list.append(relation_pairs)
                env_labels_list.append(test_labels)
                    
                test_env = ppoAgent.env(env_states_list, env_labels_list)
                rewards = agent.test(test_env)
                total_reward += rewards 
                
            test_accuracy = total_reward / (1.0 * total_test_samples)

            print(f'mean loss : {mean_loss}')   
            print("test accuracy : ", test_accuracy)
            
            writer.add_scalar('1.loss', mean_loss, episode + 1)      
            writer.add_scalar('4.test accuracy', test_accuracy, episode + 1)
            
            loss_list = []   
            
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(
                    feature_encoder.state_dict(),
                    str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                torch.save(
                    model.state_dict(),
                    str("./models/miniimagenet_actor_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                
                print("save networks for episode:", episode)
                last_accuracy = test_accuracy    
    
            
if __name__ == "__main__":
    main()
