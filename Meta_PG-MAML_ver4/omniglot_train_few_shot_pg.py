import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tensorboardX import SummaryWriter

import numpy as np
import omniglot_task_generator as tg
import omniglot_models as models
import a2cAgent
import os
import random


writer = SummaryWriter(logdir='scalar')

# * Hyper Parameters
# * Hyper Parameters for 5 way 5 shots train
FEATURE_DIM = 64  # args.feature_dim
RELATION_DIM = 8  # args.relation_dim
CLASS_NUM = 5  # args.class_num
SAMPLE_NUM_PER_CLASS = 5  # args.sample_num_per_class
BATCH_NUM_PER_CLASS = 15  # args.batch_num_per_class
EPISODE = 1000000  # args.episode
TEST_EPISODE = 1000  # args.test_episode
LEARNING_RATE = 0.001  # args.learning_rate
# GPU = # args.gpu
HIDDEN_UNIT = 10  # args.hidden_unit
GAMMA = 0.9
ENTROPY_WEIGHT = 1e-2
        
def main():    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.omniglot_character_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = models.CNNEncoder()
    #actor = models.Actor(FEATURE_DIM, RELATION_DIM, CLASS_NUM)
    #critic = models.Critic(FEATURE_DIM, RELATION_DIM)
    
    agent = a2cAgent.A2CAgent(GAMMA, ENTROPY_WEIGHT, FEATURE_DIM, RELATION_DIM, CLASS_NUM, device)
        
    feature_encoder.apply(models.weights_init)
    #actor.apply(models.weights_init)
    #critic.apply(models.weights_init)
    
    feature_encoder.to(device)
    #actor.to(device)
    #critic.to(device)
    
    cross_entropy = nn.CrossEntropyLoss()
            
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=LEARNING_RATE)
    feature_encoder_scheduler = StepLR(feature_encoder_optim, step_size=100000, gamma=0.5)
    
    if os.path.exists(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")
        
    '''
    if os.path.exists(str("./models/omniglot_actor_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        actor.load_state_dict(torch.load(str("./models/omniglot_actor_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load actor network success")
        
    if os.path.exists(str("./models/omniglot_critic_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        critic.load_state_dict(torch.load(str("./models/omniglot_critic_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load critic network success")
    '''
        
    # * Step 3: build graph
    print("Training...")
    
    last_accuracy = 0.0
    # embedding_loss_list = []
    policy_loss_list = []
    value_loss_list = []

    for episode in range(EPISODE):
        
        # * init dataset
        # * sample_dataloader is to obtain previous samples for compare
        # * batch_dataloader is to batch samples for training
        degrees = random.choice([0, 90, 180, 270])
        number_of_query_image = 10
        task = tg.OmniglotTask(metatrain_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
        sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
        batch_dataloader = tg.get_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True, rotation=degrees)
        
        # * sample datas
        # samples, sample_labels = sample_dataloader.__iter__().next()
        # batches, batch_labels = batch_dataloader.__iter__().next()
        
        samples, sample_labels = next(iter(sample_dataloader))
        batches, batch_labels = next(iter(batch_dataloader))
        
        # RFT_samples, RFT_sample_labels = samples, sample_labels
        
        samples, sample_labels = samples.to(device), sample_labels.to(device)
        batches, batch_labels = batches.to(device), batch_labels.to(device)
        
        # one_hot_sample_labels = torch.zeros(SAMPLE_NUM_PER_CLASS * CLASS_NUM, CLASS_NUM).to(device).scatter_(1, sample_labels.view(-1, 1), 1)
        
        # * calculates features
        sample_features = feature_encoder(samples)
        # RFT_sample_features = sample_features.detach().cpu().reshape(RFT_samples.shape[0], -1)
        
        sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
        sample_features = torch.sum(sample_features, 1).squeeze(1)
        
        batch_features = feature_encoder(batches)
        # RFT_batch_features = batch_features.detach().cpu().reshape(RFT_batches.shape[0], -1)
        
        # embedding_loss = mse(linear, one_hot_sample_labels)
        # embedding_loss = cross_entropy(linear, sample_labels)
        
        # * calculate relations
        # * each batch sample link to every samples to calculate relations
        # * to form a 100 * 128 matrix for relation network
        sample_features_ext = sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = batch_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)
        
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
        policy_loss, value_loss = agent.update_model(relation_pairs, batch_labels)

        if (episode + 1) % 100 == 0:
            print(f"episode : {episode+1}, policy_loss : {policy_loss.cpu().detach().numpy()}")
            print(f"episode : {episode+1}, policy_loss : {value_loss.cpu().detach().numpy()}")
            
            policy_loss_list.append(policy_loss.cpu().detach().numpy())
            value_loss_list.append(value_loss.cpu().detach().numpy())
            
        if (episode + 1) % 1000 == 0:
            print("Testing...")
            total_reward = 0
            
            # feature_encoder.eval()
            # relation_network.eval()
            for i in range(TEST_EPISODE):
                degrees = random.choice([0, 90, 180, 270])
                number_of_query_image = 10
                task = tg.OmniglotTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                sample_dataloader = tg.get_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False, rotation=degrees)
                test_dataloader = tg.get_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True, rotation=degrees)
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                    
                # * calculate features
                sample_features = feature_encoder(sample_images)
                sample_features = sample_features.view(CLASS_NUM, SAMPLE_NUM_PER_CLASS, FEATURE_DIM, 5, 5)
                sample_features = torch.sum(sample_features, 1).squeeze(1)
                test_features = feature_encoder(test_images)
                
                # * calculate relations
                # * each batch sample link to every samples to calculate relations
                # * to form a 100x128 matrix for relation network
                
                sample_features_ext = sample_features.unsqueeze(0).repeat(number_of_query_image * CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = test_features.unsqueeze(0).repeat(CLASS_NUM, 1, 1, 1, 1)
                test_features_ext = torch.transpose(test_features_ext, 0, 1)

                relation_pairs = torch.cat((sample_features_ext, test_features_ext), 2).view(-1, FEATURE_DIM * 2, 5, 5)
                
                rewards = agent.step(relation_pairs, test_labels)
                total_reward += np.sum(rewards)

           # feature_encoder.train()
           # relation_network.train()
            
            test_accuracy = total_reward / (1.0 * CLASS_NUM * SAMPLE_NUM_PER_CLASS * TEST_EPISODE)
            mean_policy_loss = np.mean(policy_loss_list)
            mean_value_loss = np.mean(value_loss_list)
            
            print(f'mean_policy_loss : {mean_policy_loss}')      
            print(f'mean_value_loss : {mean_value_loss}')      
            
            writer.add_scalar('1.mean_policy_loss', mean_policy_loss, episode + 1)
            writer.add_scalar('2.mean_value_loss', mean_value_loss, episode + 1)
            writer.add_scalar('test accuracy', test_accuracy, episode + 1)
            
            policy_loss_list = []
            value_loss_list = []
            
            if test_accuracy > last_accuracy:
                # save networks
                torch.save(
                    feature_encoder.state_dict(),
                    str("./models/omniglot_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )
                torch.save(
                    relation_network.state_dict(),
                    str("./models/omniglot_relation_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")
                )

                print("save networks for episode:", episode)

                last_accuracy = test_accuracy    
    
            
if __name__ == "__main__":
    main()
