import torch
import numpy as np
import task_generator as tg
import miniimagenet_train_pg as ot
import models
import a2cAgent
import os
import scipy as sp
import scipy.stats


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
GAMMA = 0.9
ENTROPY_WEIGHT = 1e-2
ENV_LENGTH = 5
INNER_BATCH_RANGE = 3
META_BATCH_RANGE = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1 + confidence) / 2, n - 1)
    return m, h


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    
    # * Step 1: init data folders
    print("init data folders")
    
    # * Init character folders for dataset construction
    metatrain_character_folders, metatest_character_folders = tg.mini_imagenet_folders()
    
    # * Step 2: init neural networks
    print("init neural networks")
    
    feature_encoder = models.CNNEncoder()    
    actor = models.Actor(FEATURE_DIM, RELATION_DIM, CLASS_NUM)
    critic = models.Critic(FEATURE_DIM, RELATION_DIM)

    #feature_encoder = torch.nn.DataParallel(feature_encoder)
    #actor = torch.nn.DataParallel(actor)
    #critic = torch.nn.DataParallel(critic)
    
    feature_encoder.train()
    actor.train()
    critic.train()
    
    feature_encoder.apply(models.weights_init)
    actor.apply(models.weights_init)
    critic.apply(models.weights_init)
    
    feature_encoder.to(device)
    actor.to(device)
    critic.to(device)

    agent = a2cAgent.A2CAgent(actor, critic, GAMMA, ENTROPY_WEIGHT, FEATURE_DIM, RELATION_DIM, CLASS_NUM, device)
    
    #feature_encoder.eval()
    #relation_network.eval()    
    
    if os.path.exists(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        feature_encoder.load_state_dict(torch.load(str("./models/miniimagenet_feature_encoder_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load feature encoder success")            
        
    if os.path.exists(str("./models/miniimagenet_actor_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        actor.load_state_dict(torch.load(str("./models/miniimagenet_actor_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load actor network success")
        
    if os.path.exists(str("./models/miniimagenet_critic_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")):
        critic.load_state_dict(torch.load(str("./models/miniimagenet_critic_network_" + str(CLASS_NUM) + "way_" + str(SAMPLE_NUM_PER_CLASS) + "shot.pkl")))
        print("load critic network success")

    
    max_accuracy_list = []
    mean_accuracy_list = []
    for episode in range(1):
        total_accuracy = []
        for i in range(TEST_EPISODE // ENV_LENGTH):
            # * Generate env
            env_states_list = []
            env_labels_list = []
            total_test_labels_nums = 0
            for j in range(ENV_LENGTH):
                number_of_query_image = 15
                task = tg.MiniImagenetTask(metatest_character_folders, CLASS_NUM, SAMPLE_NUM_PER_CLASS, number_of_query_image)
                sample_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=SAMPLE_NUM_PER_CLASS, split="train", shuffle=False)                
                test_dataloader = tg.get_mini_imagenet_data_loader(task, num_per_class=number_of_query_image, split="test", shuffle=True)
                
                sample_images, sample_labels = next(iter(sample_dataloader))
                test_images, test_labels = next(iter(test_dataloader))

                total_test_labels_nums += len(test_labels)
                
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
                
            test_env = a2cAgent.env(env_states_list, env_labels_list)
            rewards = agent.test(test_env)
            '''
            print(rewards)
            print(total_test_labels_nums)
            assert 1 == 2
            '''
            test_accuracy = rewards / total_test_labels_nums
            print(test_accuracy)
            total_accuracy.append(test_accuracy)
            
        mean_accuracy, conf_int = mean_confidence_interval(total_accuracy)        
        print(f"Total accuracy : {mean_accuracy:.4f}")
        print(f"confidence interval : {conf_int:.4f}")


if __name__ == "__main__":
    main()
