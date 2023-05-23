
   
#================================================================

#================================================================
import os
import math
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # -1:cpu, 0:first gpu
import random
import gym
import pylab
import numpy as np
import tensorflow as tf
from tensorboardX import SummaryWriter
#tf.config.experimental_run_functions_eagerly(True) # used for debuging and development
tf.compat.v1.disable_eager_execution() # usually using this for fastest performance
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense, LayerNormalization, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam, RMSprop, Adagrad, Adadelta

from tensorflow.keras import backend as K
import copy

from threading import Thread, Lock
from multiprocessing import Process, Pipe
import time

gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    print(f'GPUs {gpus}')
    try: tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError: pass

class Environment(Process):
    def __init__(self, env_idx, child_conn, env_name, state_size, action_size, visualize = False, double_reward=False):
        super(Environment, self).__init__()
        self.env = gym.make(env_name)
        self.is_render = visualize
        self.env_idx = env_idx
        self.child_conn = child_conn
        self.state_size = state_size
        self.action_size = action_size
        self.double_reward_ = double_reward
        

    def run(self):
        super(Environment, self).run()
        state = self.env.reset()
        state = np.reshape(state, [1, self.state_size])
        self.child_conn.send(state)
        
        while True:
            action = self.child_conn.recv()
            #if self.is_render and self.env_idx == 0:
                #self.env.render()
            if self.double_reward_ == False:
                state, reward, done, info = self.env.step(action)
                state = np.reshape(state, [1, self.state_size])

                if done:
                    state = self.env.reset()
                    state = np.reshape(state, [1, self.state_size])
                self.child_conn.send([state, reward, done, info])
                
            else:
                state, reward, reward2, done, info = self.env.double_reward_step(action)
                
                state = np.reshape(state, [1, self.state_size])

                if done:
                    state = self.env.reset()
                    state = np.reshape(state, [1, self.state_size])

                self.child_conn.send([state, reward, reward2, done, info])


class Actor_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        self.action_space = action_space
        X_1 = Dense(512, activation="sigmoid", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        X = LayerNormalization(epsilon = 1e-6)(X_1)
        
        X = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dropout(rate=.1)(X + X_1)
        X = LayerNormalization(epsilon = 1e-6)(X)
        
        X = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        X = Dropout(rate=.1)(X )
        X = LayerNormalization(epsilon = 1e-6)(X)
        
        X = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X)
        
        output = Dense(self.action_space, activation="tanh")(X)

        self.Actor = Model(inputs = X_input, outputs = output)
        self.Actor.compile(loss=self.ppo_loss_continuous, optimizer=optimizer(learning_rate=lr))
        #print(self.Actor.summary())

    def ppo_loss_continuous(self, y_true, y_pred):
        advantages, actions, logp_old_ph, = y_true[:, :1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]
        LOSS_CLIPPING = 0.2
        logp = self.gaussian_likelihood(actions, y_pred)

        ratio = K.exp(logp - logp_old_ph)

        p1 = ratio * advantages
        p2 = tf.where(advantages > 0, (1.0 + LOSS_CLIPPING)*advantages, (1.0 - LOSS_CLIPPING)*advantages) # minimum advantage

        actor_loss = -K.mean(K.minimum(p1, p2))

        return actor_loss

    def gaussian_likelihood(self, actions, pred): # for keras custom loss
        log_std = -0.5 * np.ones(self.action_space, dtype=np.float32)
        pre_sum = -0.5 * (((actions-pred)/(K.exp(log_std)+1e-8))**2 + 2*log_std + K.log(2*np.pi))
        return K.sum(pre_sum, axis=1)

    def predict(self, state):
        return self.Actor.predict(state)


class Critic_Model:
    def __init__(self, input_shape, action_space, lr, optimizer):
        X_input = Input(input_shape)
        old_values = Input(shape=(1,))
        
        V_1 = Dense(512, activation="sigmoid", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(X_input)
        V = LayerNormalization(epsilon = 1e-6)(V_1)
        
        V = Dense(512, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dropout(rate=.15)(V)
        V = LayerNormalization(epsilon = 1e-6)(V + V_1)
        
        V = Dense(256, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        V = Dropout(rate=.15)(V)
        V = LayerNormalization(epsilon = 1e-6)(V)
        
        V = Dense(64, activation="relu", kernel_initializer=tf.random_normal_initializer(stddev=0.01))(V)
        value = Dense(1, activation=None)(V)

        self.Critic = Model(inputs=[X_input, old_values], outputs = value)
        self.Critic.compile(loss=[self.critic_PPO2_loss(old_values)], optimizer=optimizer(learning_rate=lr))

    def critic_PPO2_loss(self, values):
        def loss(y_true, y_pred):
            LOSS_CLIPPING = 0.2
            clipped_value_loss = values + K.clip(y_pred - values, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            #value_loss = K.mean((y_true - y_pred) ** 2) # standard PPO loss
            return value_loss
        return loss

    def predict(self, state):
        return self.Critic.predict([state, np.zeros((state.shape[0], 1))])
    

class PPOAgent:
    # PPO Main Optimization Algorithm
    def __init__(self, env_name, model_name=""):
        # Initialization
        # Environment and PPO parameters
        self.env_name = env_name       
        self.env = gym.make(env_name)
        self.action_size = self.env.action_space.shape[0]
        self.state_size = self.env.observation_space.shape
        self.EPISODES = 5000 # total episodes to train through all environments
        self.episode = 0 # used to track the episodes total count of episodes played through all thread environments
        self.max_average = 0 # when average score is above 0 model will be saved
        self.lr = 0.00025
        self.epochs = 10 # training epochs
        self.shuffle = True
        self.Training_batch = 2000
        #self.optimizer = RMSprop
        self.optimizer = Adam
        self.replay_count = 0
        self.writer = SummaryWriter(comment="_"+self.env_name+"_"+self.optimizer.__name__+"_"+str(self.lr))
        self.Epsilon_greedy = True
        self.algorithm_action = False
        self.Epsilon = 1
        self.increase_reward_reliability = True
        self.gamma = 0.97
        self.double_reward = True
        self.short_time_gamma = 0.9
        self.long_time_gamma = 1
        self.short_time_conv_mask = []
        self.long_time_conv_mask = []
    

        
        # Instantiate plot memory
        self.scores_, self.episodes_, self.average_ = [], [], [] # used in matplotlib plots
        Actor_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
          self.lr,
          decay_steps = 100,
          decay_rate = 0.9,
          staircase=True
        )
        Critic_learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(
          self.lr,
          decay_steps = 100,
          decay_rate = 0.9,
          staircase=True
        )

        
        # Create Actor-Critic network models
        self.Actor = Actor_Model(input_shape = self.state_size, action_space = self.action_size, lr = Actor_learning_rate, optimizer = self.optimizer)
        
        self.Critic = Critic_Model(input_shape = self.state_size, action_space = self.action_size, lr = Critic_learning_rate, optimizer = self.optimizer)
       
        self.Actor_name = f"{self.env_name}_last_PPO_Actor.h5"
        self.Critic_name = f"{self.env_name}_last_PPO_Critic.h5"
        self.Actor_name2 = f"{self.env_name}_PPO_Actor.h5"
        self.Critic_name2 = f"{self.env_name}_PPO_Critic.h5"
        #self.load() # uncomment to continue training from old weights

        # do not change bellow
        self.log_std = -0.5 * np.ones(self.action_size, dtype = np.float32)
        self.std = np.exp(self.log_std)


    def act(self, state):
        # Use the network to predict the next action to take, using the model
        pred = self.Actor.predict(state)
        if self.Epsilon_greedy == True:
            if np.random.uniform(0, 1, size = 1) < self.Epsilon:
                training_remaining_rate = (self.EPISODES - self.episode) / self.EPISODES
                action_low = self.env.pygame.car.min_acceleration_action
                action_high = self.env.pygame.car.max_acceleration_action
                action = pred + np.random.uniform(action_low, action_high, size = pred.shape) * self.std * training_remaining_rate * (self.env.pygame.Hit_wall_times + 1)
                action = np.clip(action, action_low, action_high)
                logp_t = self.gaussian_likelihood(action, pred, self.log_std)
                return action, logp_t
                
            else:
                raining_remaining_rate = (self.EPISODES - self.episode) / self.EPISODES
                action_low = self.env.pygame.car.min_acceleration_action
                action_high = self.env.pygame.car.max_acceleration_action
                action = pred + np.random.uniform(action_low, action_high, size = pred.shape) * self.std * training_remaining_rate * (self.env.pygame.Hit_wall_times + 0.2 ) * 2
                action = np.clip(action, action_low, action_high)
                logp_t = self.gaussian_likelihood(action, pred, self.log_std)
                return action, logp_t            
        else:
            #print(self.state_size)
            training_remaining_rate = (self.EPISODES - self.episode) / self.EPISODES
            action_low = self.env.pygame.car.min_acceleration_action
            action_high = self.env.pygame.car.max_acceleration_action
            action = pred + np.random.uniform(action_low, action_high, size = pred.shape) * self.std * training_remaining_rate * (self.env.pygame.Hit_wall_times + 1)
            action = np.clip(action, action_low, action_high)
            logp_t = self.gaussian_likelihood(action, pred, self.log_std)
            return action, logp_t
            
    def power_algorithm_act(self, state):
        action_low = self.env.pygame.car.min_acceleration_action
        action_high = self.env.pygame.car.max_acceleration_action
        np.random.uniform(action_low, action_high, size = pred.shape)
        wight_dis = state[11] - state[9]
        height_dis = state[12] - state[10]
        #math.tan(

    def gaussian_likelihood(self, action, pred, log_std):
        # https://github.com/hill-a/stable-baselines/blob/master/stable_baselines/sac/policies.py
        pre_sum = -0.5 * (((action-pred)/(np.exp(log_std)+1e-8))**2 + 2*log_std + np.log(2*np.pi)) 
        return np.sum(pre_sum, axis=1)


    def get_gaes(self, rewards, dones, values, next_values, lamda = 0.90, normalize = True):
        deltas = [r + self.gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * self.gamma * lamda * gaes[t + 1]

        target = gaes + values
        
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)
        
    def get_gaes2(self, rewards, rewards2, dones, values, next_values, lamda = 0.90, normalize = True):
        rewards.reverse()
        rewards2.reverse()
        
        
        reg = np.convolve(self.short_time_conv_mask, rewards)
        reg2 = np.convolve(self.long_time_conv_mask, rewards2)
        gaes = reg[len(rewards):0:-1] + reg2[len(rewards2):0:-1]
        #rint(reg2[len(rewards2):0:-1])
        target = gaes + values
        
        if normalize:
            gaes = gaes - values + next_values
        return np.vstack(gaes), np.vstack(target)

    def replay(self, states, actions, rewards, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages

        #print(rewards)
        advantages, target = self.get_gaes(rewards, dones, np.squeeze(values), np.squeeze(next_values))
       
        #print("replay\n")
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        #print(rewards,advantages)
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
        
    def replay2(self, states, actions, rewards, rewards2, dones, next_states, logp_ts):
        # reshape memory to appropriate shape for training
        
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        actions = np.vstack(actions)
        logp_ts = np.vstack(logp_ts)

        # Get Critic network predictions 
        values = self.Critic.predict(states)
        next_values = self.Critic.predict(next_states)

        # Compute discounted rewards and advantages

        #print(rewards)
        advantages, target = self.get_gaes2(rewards, rewards2, dones, np.squeeze(values), np.squeeze(next_values))
       
        #print("replay\n")
        '''
        pylab.plot(adv,'.')
        pylab.plot(target,'-')
        ax=pylab.gca()
        ax.grid(True)
        pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
        pylab.show()
        if str(episode)[-2:] == "00": pylab.savefig(self.env_name+"_"+self.episode+".png")
        '''
        # stack everything to numpy array
        # pack all advantages, predictions and actions to y_true and when they are received
        # in custom loss function we unpack it
        y_true = np.hstack([advantages, actions, logp_ts])
        #print(rewards,advantages)
        # training Actor and Critic networks
        a_loss = self.Actor.Actor.fit(states, y_true, epochs=self.epochs, verbose=0, shuffle=self.shuffle)
        c_loss = self.Critic.Critic.fit([states, values], target, epochs=self.epochs, verbose=0, shuffle=self.shuffle)

        # calculate loss parameters (should be done in loss, but couldn't find working way how to do that with disabled eager execution)
        pred = self.Actor.predict(states)
        log_std = -0.5 * np.ones(self.action_size, dtype=np.float32)
        logp = self.gaussian_likelihood(actions, pred, log_std)
        approx_kl = np.mean(logp_ts - logp)
        approx_ent = np.mean(-logp)

        self.writer.add_scalar('Data/actor_loss_per_replay', np.sum(a_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/critic_loss_per_replay', np.sum(c_loss.history['loss']), self.replay_count)
        self.writer.add_scalar('Data/approx_kl_per_replay', approx_kl, self.replay_count)
        self.writer.add_scalar('Data/approx_ent_per_replay', approx_ent, self.replay_count)
        self.replay_count += 1
 
    def load(self):
        self.Actor.Actor.load_weights(f"{self.env_name}_best_PPO_Actor.h5")
        self.Critic.Critic.load_weights(f"{self.env_name}_best_PPO_Critic.h5")

    def save(self):

        self.Actor.Actor.save_weights(f"{self.env_name}_best_PPO_Actor.h5")
        self.Critic.Critic.save_weights(f"{self.env_name}_best_PPO_Critic.h5")

    pylab.figure(figsize=(18, 9))
    pylab.subplots_adjust(left=0.05, right=0.98, top=0.96, bottom=0.06)
    def PlotModel(self, score, episode, save=True):
        self.scores_.append(score)
        self.episodes_.append(episode)
        self.average_.append(sum(self.scores_[-50:]) / len(self.scores_[-50:]))
        if str(episode)[-2:] == "00":# much faster than episode % 100
            pylab.plot(self.episodes_, self.scores_, 'b')
            pylab.plot(self.episodes_, self.average_, 'r')
            pylab.ylabel('Score', fontsize=18)
            pylab.xlabel('Steps', fontsize=18)
            try:
                pylab.grid(True)
                pylab.savefig(self.env_name+".png")
            except OSError:
                pass
        # saving best models
        if self.average_[-1] >= self.max_average and save :
            if self.scores_[-1] >= self.average_[-1]: 
                self.max_average = self.average_[-1]
                self.save()
                SAVING = "SAVING"
                # decreaate learning rate every saved model
                #self.lr *= 0.99
                #K.set_value(self.Actor.Actor.optimizer.learning_rate, self.lr)
                #K.set_value(self.Critic.Critic.optimizer.learning_rate, self.lr)
            else:
                SAVING = ""
        else:
            SAVING = ""

        return self.average_[-1], SAVING
    
    def run_batch(self):
        #self.load()
        state = self.env.reset()

        state = np.reshape(state, [1, self.state_size[0]])
        done, score, SAVING = False, 0, ''
        self.env.pygame.current_step = 0
        self.env.pygame.max_iteration = self.Training_batch 
        self.env.pygame.total_step = self.EPISODES
        for t in range(self.Training_batch):
            self.short_time_conv_mask.append(max(pow(self.short_time_gamma, t), 0.0001))
            if self.double_reward == True:
                self.long_time_conv_mask.append(max(pow(self.long_time_gamma, t),0.0001))
        self.Training_batch = self.env.pygame.max_iteration
        while True:
            # Instantiate or reset games memory
            states, next_states, actions, rewards, rewards2, dones, logp_ts = [], [], [], [], [], [], []
            view_ = False
            for t in range(self.Training_batch):
                #print(state)      #observe
                # Actor picks an action
                self.env.pygame.current_step = self.episode
                reward2 = 0
                action, logp_t = self.act(state)
                # Retrieve new state, reward, and whether the state is terminal
                if self.double_reward == False:
                    next_state, reward, done, _ = self.env.step(action[0])
                    score += reward
                else:
                    next_state, reward, reward2, done, _ = self.env.double_reward_step(action[0])
                    rewards2.append(reward2)
                    score += reward2
                
                # Memorize (state, next_states, action, reward, done, logp_ts) for training
                states.append(state)
                next_states.append(np.reshape(next_state, [1, self.state_size[0]]))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                logp_ts.append(logp_t[0])
                # Update current state shape
                state = np.reshape(next_state, [1, self.state_size[0]])
                if done:
                    self.episode += 1
                    
                    average, SAVING = self.PlotModel(score, self.episode)
                    print("episode: {}/{}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, score, average, SAVING))
                    self.writer.add_scalar(f'Workers:{1}/score_per_episode', score,self.episode)
                    self.writer.add_scalar(f'Workers:{1}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{1}/average_score',  average, self.episode)
                    
                    state, done, score, SAVING = self.env.reset(), False, 0, ''
                    state = np.reshape(state, [1, self.state_size[0]])
                    if self.episode % 20 == 1:
                        view_ = True
                    else:
                        view_ = False
                    #view_ = True
                    
                #if view_ == True:
                   #self.env.render()
                
            if self.episode % 500 == 0:
                self.Actor.Actor.save_weights(f"{self.episode}_{self.Actor_name2}")
                self.Critic.Critic.save_weights(f"{self.episode}_{self.Critic_name2}")
            if self.double_reward == False:    
                self.replay(states, actions, rewards, dones, next_states, logp_ts)
            else:
                self.replay2(states, actions, rewards, rewards2, dones, next_states, logp_ts)
            if self.episode >= self.EPISODES:
                break
        self.Actor.Actor.save_weights(self.Actor_name)
        self.Critic.Critic.save_weights(self.Critic_name)
        self.env.close()


    def run_multiprocesses(self, num_worker = 4):
        #self.load()
        works, parent_conns, child_conns = [], [], []
        for t in range(self.Training_batch):
            self.short_time_conv_mask.append(max(pow(self.short_time_gamma, t), 0.0001))
            if self.double_reward == True:
                self.long_time_conv_mask.append(max(pow(self.long_time_gamma, t),0.0001))
        for idx in range(num_worker):
            parent_conn, child_conn = Pipe()
            work = Environment(idx, child_conn, self.env_name, self.state_size[0], self.action_size, True, self.double_reward)
            
            work.start()
            works.append(work)
            parent_conns.append(parent_conn)
            child_conns.append(child_conn)

        states =        [[] for _ in range(num_worker)]
        next_states =   [[] for _ in range(num_worker)]
        actions =       [[] for _ in range(num_worker)]
        rewards =       [[] for _ in range(num_worker)]
        rewards2 =      [[] for _ in range(num_worker)]
        dones =         [[] for _ in range(num_worker)]
        logp_ts =       [[] for _ in range(num_worker)]
        score =         [0 for _ in range(num_worker)]

        state = [0 for _ in range(num_worker)]
        for worker_id, parent_conn in enumerate(parent_conns):
            state[worker_id] = parent_conn.recv()

        while self.episode < self.EPISODES:
            # get batch of action's and log_pi's
            action, logp_pi = self.act(np.reshape(state, [num_worker, self.state_size[0]]))
            
            for worker_id, parent_conn in enumerate(parent_conns):
                parent_conn.send(action[worker_id])
                actions[worker_id].append(action[worker_id])
                logp_ts[worker_id].append(logp_pi[worker_id])
            

            for worker_id, parent_conn in enumerate(parent_conns):
                reward2 = 0
                if self.double_reward == False:
                    next_state, reward, done, _ = parent_conn.recv()
                    score[worker_id] += reward
                else:
                    next_state, reward, reward2, done, _ = parent_conn.recv()
                    score[worker_id] += reward2
                    
                states[worker_id].append(state[worker_id])
                next_states[worker_id].append(next_state)
                rewards[worker_id].append(reward)
                rewards2[worker_id].append(reward2)
                dones[worker_id].append(done)
                state[worker_id] = next_state
                

                if done:
                    average, SAVING = self.PlotModel(score[worker_id], self.episode)
                    print("episode: {}/{}, worker: {}, score: {}, average: {:.2f} {}".format(self.episode, self.EPISODES, worker_id, score[worker_id], average, SAVING))
                    self.writer.add_scalar(f'Workers:{num_worker}/score_per_episode', score[worker_id], self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/learning_rate', self.lr, self.episode)
                    self.writer.add_scalar(f'Workers:{num_worker}/average_score',  average, self.episode)
                    score[worker_id] = 0
                    if(self.episode < self.EPISODES):
                        self.episode += 1
                    if self.episode % 200 == 199:
                        self.Actor.Actor.save_weights(f"{self.episode}_{self.Actor_name2}")
                        self.Critic.Critic.save_weights(f"{self.episode}_{self.Critic_name2}")
                        drive.mount('/content/drive')

                        
                        
            for worker_id in range(num_worker):
                if len(states[worker_id]) >= self.Training_batch:
                    if self.double_reward == False:
                        self.replay(states[worker_id], actions[worker_id], rewards[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])
                    else:
                        self.replay2(states[worker_id], actions[worker_id], rewards[worker_id], rewards2[worker_id], dones[worker_id], next_states[worker_id], logp_ts[worker_id])

                    states[worker_id] = []
                    next_states[worker_id] = []
                    actions[worker_id] = []
                    rewards[worker_id] = []
                    rewards2[worker_id] = []
                    dones[worker_id] = []
                    logp_ts[worker_id] = []
            
                

        # terminating processes after a while loop
        works.append(work)
        for work in works:
            work.terminate()
            print('TERMINATED:', work)
            work.join()

    def test(self, test_episodes = 100):#evaluate
        self.load()
        for e in range(1):
            state = self.env.reset()
            state = np.reshape(state, [1, self.state_size[0]])
            done = False
            score = 0
            while not done:
                #self.env.render()
                action = self.Actor.predict(state)[0]
                state, reward, done, _ = self.env.step(action)
                state = np.reshape(state, [1, self.state_size[0]])
                score += reward
                if done:
                    average, SAVING = self.PlotModel(score, e, save=False)
                    print("episode: {}/{}, score: {}, average{}".format(e, test_episodes, score, average))
                    break
        self.env.close()
            


#drive.mount('/content/drive')
env_name = 'Car_cpu-test1'
agent = PPOAgent(env_name)
agent.run_batch() # train as PPO
#agent.run_multiprocesses(num_worker = 4)  # train PPO multiprocessed (fastest)
agent.test()


