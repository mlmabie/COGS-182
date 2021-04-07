# standard imports
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import random

# Open AI env imports
import gym
# from gym.envs.box2d import BipedalWalker

# neural network imports
import tensorflow as tf
from tensorflow import keras
from keras import backend as K
from tensorflow.keras import layers


# Controls agent, including actor-critic methods.
class Agent:
    def __init__(self, actions=4,states=14, batch=1, gamma=0.99, alpha=0.1):
        # self.s_link = s_link # h5 file to save model to
        self.action_space = actions

        self.state_space = states

        self.message = '\n making new model '
        
        # # set ACTOR
        # self.a_lidar_input, self.a_state_input, self.a_local = self.Actor()
        # _,_, self.a_target = self.Actor()

        # self.lr_actor = 1e-4 # actor learning rate
        # self.lr_critic = 3e-4 # critic learning rate
        self.batch = batch
        self.gamma = gamma # 0.99
        self.alpha = alpha #0.1
        # change epsilon over time as model learns
        self.e = 1
        self.e_ = .01
        self.dc = 0.9999
        
        # neural net gets to learn action selection:
        #   actor gets state, returns action and probability of action in space
        #   critic gets state and action (/ log probability?), returns value estimates. (Q function estimator)

        self.state_inputs = layers.Input(shape=14) # 14 state features cause no LIDAR

        # initialize actor and critic objects
        self.Actor = self.init_Actor(lr=1e-4)
        self.Critic= self.init_Critic(lr=3e-4)
    
    # we use gaussian likelihood for value function estimation and for calculating loss.
    def gaussian_likelihood(self, actions, pred):
        log_sigma = -0.5*np.ones(self.action_space, dtype=np.float32)
        log_prob = -0.5*(((actions - pred) / (K.exp(log_sigma)+1e-8))**2 + 2*log_sigma + K.log(2*np.pi))
        return K.sum(log_prob,axis=1)

    # action comprised of four torque values. also returns log prob of action.
    def choose_action(self, state):
        min,max = -1.0,1.0
        # get an action for Actor using policy, which is an NN with custom loss.
        prediction = self.Actor.predict(state)
        # Use epsilon-greedy with epsilon that shifts more greedy over time
        if np.random.rand() <= self.e :
            # semi exploratory
            action = prediction + np.random.uniform(min,max,4) * 0.5
        else:
            # rely wholy on prediction
            action = prediction
        # clip the outputs to be within valid torque control values
        action = np.clip(action, min,max)
        log_prob = self.gaussian_likelihood(action, prediction)
        return action[0], log_prob

    # single neural net value so far...TD step done in train
    def check_critic(self, state):
        return self.Critic(state)

    def store(state,action,reward,state_new,flag):
        pass

    # to better incorporate loss, i've separated the actor and critic.
    ## Actor:
    def init_Actor(self, lr):
        # using convenient memory sizes for first layer. activation function choice is relu.
        common = layers.Dense(512, activation="relu")(self.state_inputs)
        hidden = layers.Dense(256, activation="relu")(common)
        hidden = layers.Dense(64, activation="relu")(hidden)
        # 4 action features: activation funtion is tanh, which matches torque limits.
        action_choices = layers.Dense(self.action_space, activation="tanh")(hidden)
        # and so we have an actor model.
        model = keras.Model(inputs=self.state_inputs, outputs=action_choices)
        # optimizer is Adam, which I like because it calculates gradient descent with two moments.
        # keras nicely takes care of things from here.
        model.compile(loss=self.ppo_loss, optimizer=keras.optimizers.Adam(lr))
        return model

    # need to configure what makes an action choice accurate or not accurate.
    def ppo_loss(self, y_true,y_pred):
        # we're copying the proximal policy optimization loss formula from wikipedia.
        # first, decode y_true to get what we actually mean
        delta, action, old_log_prob, = y_true[:,:1], y_true[:, 1:1+self.action_space], y_true[:, 1+self.action_space]

        # loss clipping avoids exploding gradients & provides sense of action search width.
        CLIP = 0.2
        # action probability is generally found by calculating standard devation. We incorporate gaussian likelihood.
        # gaussian likelihood
        log_prob = self.gaussian_likelihood(action, y_pred)
        # ratio from the ppo formula
        ratio = K.exp(log_prob - old_log_prob)

        p1 = ratio * delta
        # find where there is some minimum delta to adjust from.
        p2 = tf.where(delta > 0, (1.0 + CLIP)*delta, (1.0 - CLIP)*delta)

        # adjust by the mean of the minimum action probability feedback.
        return -K.mean(K.minimum(p1,p2))
    
    def predict(self, state):
        return self.Actor.predict(state)

    # value function estimator that gives R(t+1) + gamma*V(St+1, w)
    # updates itself using a TD step
    ## Critic:
    def init_Critic(self, lr):
        state_h1 = layers.Dense(24, activation='relu')(self.state_inputs)
        state_h2 = layers.Dense(48)(state_h1)
                
        action_input = layers.Input(shape=self.action_space)
        action_h1 = layers.Dense(48)(action_input)

        merged = layers.Add()([state_h2, action_h1])
        merged_h1 = layers.Dense(24, activation='relu')(merged)

        # critic is optimizing for a single reward value per given state & action
        output = layers.Dense(1, activation='relu')(merged_h1)
        # and so we have a critic model.
        model  = keras.Model(inputs=[self.state_inputs,action_input], outputs=output)
        # keras takes care of loss optimization.
        model.compile(loss=keras.losses.Huber(), optimizer=keras.optimizers.Adam(lr))
        return model

    # def train(self):
    #     # actor critic share the first layer
    #     # num_inputs = BipedalWalker.observation_space.shape
    #     # num_outputs = BipedalWalker.action_space.shape
    #     batch_size = 32
    #     if len(self.memory) < batch_size:
    #         return
    #     rewards = []
    #     samples = random.sample(self.memory, batch_size)
    #     self._train_critic(samples)
    #     self._train_actor(samples)


def prep_state(state):
    state = state[:14]
    state = tf.convert_to_tensor(state)
    state = tf.expand_dims(state, 0) 
    return state

def prep_action(action):
    action = tf.convert_to_tensor(action)
    action = tf.expand_dims(action, 0)
    return action

# Run tests with agent object
if __name__ == '__main__':

    # set visual
    rendering = input("Visualize rendering? [y/n]: ")
    s_link = "BipedalWalker_model.h5"  
    # environment render preferences
    RENDER_REWARD_MIN = 5000
    RENDER_ENV = False
    if rendering == 'y': RENDER_ENV = True

    # Max number of episodes
    EPISODES = 10000
    # Max steps per episode
    max_steps_per_episode = 5000
     
    # make environment
    env = gym.make('BipedalWalker-v3')
    # set seed for reproduceability
    seed = 2138
    env.seed(seed)

    # set session and initialize agent
    agent = Agent()

    # epsilon: really small number for normalization
    epsilon = np.finfo(np.float32).eps.item()

    # Note: I want the agent type to be object oriented so we can easily insert different agents.

    # # track learning
    # error = []
    # epsilons = []
    # reward_var = []
    # reward_mean = []
    # mean_100 = []

    # track actor/critic behavior
    value_hist = []
    policy_hist = []
    reward_hist = []
    action_hist = []
    running_reward = 0
    episode_count = 0

    # # dynamics of neural net
    # optimizer = keras.optimizers.Adam(learning_rate=0.01)
    # huber_loss = keras.losses.Huber()

    print('Finished init, starting Bipedal Walker\n')
    print(agent.message)    
    print("Environment Observation_space: ", env.observation_space)
    print("Environment Action_space: ", env.action_space) 
    print("Number of Episodes: " + str(EPISODES))
    # print("\n:::::Algorithm_Parameters:::::")
    # print(list(agent.parameters.items()))
    # w = 0 # number of "wins"

    # run episodes until hits EPISODES limit or "solved":
    while True:
        # reset environment
        state = prep_state(env.reset()) # don't want LIDAR
        # init ep reward
        episode_reward = 0
        # track time
        start = time.time()

        # with tf.GradientTape() as tape:
        for timestep in range(1, max_steps_per_episode):
            if RENDER_ENV:
                env.render()

            # sample an action from Actor using NN policy.
            action, log_prob = agent.choose_action(state)
            policy_hist.append(log_prob)

            # CRITIC: get "value" of state
            value = agent.Critic.predict([state,prep_action(action)])
            # value_hist.append(value)

            # take action
            n_state, reward, done, _ = env.step(action)
            n_state = prep_state(n_state)
            reward_hist.append(reward)
            episode_reward += reward

            if not done:
                # get next action from same policy
                s_action, s_log_prob = agent.choose_action(n_state)
                # (TD step)
                s_step, s_reward, s_done, _ = env.step(s_action)
                # dv/dt on value function
                s_value = agent.Critic.predict([n_state,prep_action(s_action)])
                # get feedback on critic, which then optimizes policy!
                target = reward + agent.gamma*s_value 
            else:
                target = reward
            
            # train value function
            agent.Critic.fit([state, prep_action(action)], target, verbose=0, batch_size=1)

            # we now have all the ingredients necessary to train Actor:
            delta = target - value
            y_true = prep_action(np.concatenate((delta,action,log_prob), axis=None))
            agent.Actor.fit(state, y_true, verbose=0, batch_size=1)

            # break from episode
            if done:
                break

            # our definition of "good enough" / winning the running game
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # loop
            state = n_state

        # objective function is MSE between v_policy(s) and V(s,w) times state probability
        # Both maximize expected reward. 
        # forward pass (inner loop): activate units
        # back propogation: (loop per epoch): compute partial derivative for each weight
            
        # # now train Critic to update value function
        # # past rewards discounted with gamma
        # # we use this as label for critic's function estimation.
        # returns = []
        # discounted_sum = 0
        # for r in reward_hist[::-1]:
        #     discounted_sum = r + agent.gamma * discounted_sum
        #     returns.insert(0, discounted_sum)
        # # Normalize - acts like subtracting gamma * old V
        # returns = np.array(returns)
        # returns = (returns - np.mean(returns)) / (np.std(returns) + epsilon)
        # returns = returns.tolist() # delta

        # agent.Critic.fit(value_hist, returns)

        # if we have a sample of the policy, we can modify our gradient 
        # by dividing by the probability of sampling that action.
        # gradient of log prob times Q value? minimize the L2 norm.
        # we already did the work intergrating PPO into the keras
        # loss function. We just need to format it like:
        # delta, actions, old_log_prob
        # maximizes the probability it will select an action 
        # with the highest possible future rewards in each state

        # we want to determine what change in parameters (in the actor model) 
        # would result in the largest increase in the Q value (predicted by the critic model).
        # CHAIN RULE


        #It is recommended to periodically evaluate your agent for n test episodes 
        # (n is usually between 5 and 20) and average the reward per episode to have a good estimate.

        # Calculate TD error / loss values to update neural network
        
        # the critic estimated that we would get a total reward = `value` in the future.
        # the actor took an action of `log_prob` and got total reward = `ret`.
        # actor_losses = []
        # critic_losses = []
        # history = zip(policy_hist, value_hist, returns
        # for log_prob, value, ret in history:
        #     # one way to get actor loss is by choosing a gradient that compares value/prob with value.
        #     diff = ret - value
        #     actor_losses.append(-log_prob * diff)  # actor loss estimate
        #     # The Critic must be updated so that it predicts a better estimate of
        #     # the future rewards.
        

        # Backpropagation
        # loss_value = sum(actor_losses) + sum(critic_losses)
        # grads = tape.gradient(sum(actor_losses), agent.Actor.trainable_variables)
        # agent.Actor.apply_gradients(zip(grads, agent.Actor.trainable_variables))

        # Clear the loss and reward history
        # action_hist.clear()
        # value_hist.clear()
        # reward_hist.clear()

        # update epsilon for action choice
        if agent.e >= agent.e_:
            agent.e *= agent.dc

        # get time to flag stalls
        end = time.time()
        time_space = end - start
        print("Time: ", np.round(time_space, 2),"secs")
        # if time_space > 30:
        #     flag = True
        
        # # get total episode rewards
        # ep_rew_total = sum(agent.ep_rewards)
        # mean = np.mean(agent.ep_rewards)
        # var = np.var(agent.ep_rewards)
        # if ep_rew_total < -300:
        #     flag = True

        # if flag==True:
        #     reward_mean.append(mean)
        #     reward_var.append(var)
        #     max_reward = np.max(rewards)
        #     ep_max = np.argmax(rewards)
        #     if ep_rew_total >= 300:
        #         w += 1
        #         # save the agents that win
        #         agent.save(s_link)

        # print('\nEpisode: ', i)
        # print("Reward:", ep_rew_total)
        # print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(ep_max))
        # print("Times win: " + str(w))

        # # Give summary every 100 episodes
        # if i % 100 ==0:
        #     print("Mean reward of the past 100 episodes: ", str(np.mean(rewards[-100:])))
        #     mean_100.append(np.mean(rewards[-100:]))
        #     f = open('results.txt','a')
        #     f.write('\n' + str(np.mean(rewards[-100:])))
        #     f.close()

        # Log details within tf tape
        episode_count += 1
        if episode_count % 10 == 0:
            print("Time: ", np.round(time_space, 2),"secs")
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))

        if running_reward > 195:  # Condition to consider the task solved
            print("Solved at episode {}!".format(episode_count))
            template = "\nrunning reward: {:.2f}"
            print(template.format(running_reward))
            break

        if episode_count >= EPISODES: # over our max episode count
            print("Hit max episode count!\n")
            template = "running reward: {:.2f} at episode {}"
            print(template.format(running_reward, episode_count))
            break

            # # action comprised of four torque values
            # action = agent.choose_action(state)
            # action = action.reshape((4,))

            # state_new, reward, flag, inf = env.step(np.clip(action,-1,1))
            # state_new = state_new.reshape((1,24))

            # # store state observation
            # agent.store(state, action, reward, state_new, flag)
            # state = state_new

            # # get time to flag stalls
            # end = time.time()
            # time_space = end - start
            # if time_space > 30:
            #     flag = True
            
            # # get total episode rewards
            # ep_rew_total = sum(agent.ep_rewards)
            # mean = np.mean(agent.ep_rewards)
            # var = np.var(agent.ep_rewards)
            # if ep_rew_total < -300:
            #     flag = True

            # if flag==True:
            #     rewards.append(ep_rew_total)
            #     reward_mean.append(mean)
            #     reward_var.append(var)
            #     max_reward = np.max(rewards)
            #     ep_max = np.argmax(rewards)
            #     if ep_rew_total >= 300:
            #         w += 1
            #         # save the agents that win
            #         agent.save(s_link)

            #     print('\nEpisode: ', i)
            #     print("Time: ", np.round(time_space, 2),"secs")
            #     print("Reward:", ep_rew_total)
            #     print("Maximum Reward: " + str(max_reward) + "  on Episode: " + str(ep_max))
            #     print("Times win: " + str(w))

            #     # Give summary every 100 episodes
            #     if i % 100 ==0:
            #         print("Mean reward of the past 100 episodes: ", str(np.mean(rewards[-100:])))
            #         mean_100.append(np.mean(rewards[-100:]))
            #         f = open('results.txt','a')
            #         f.write('\n' + str(np.mean(rewards[-100:])))
            #         f.close()

            #     # start training the neural network to imporve policy
            #     training_time = agent.train()
            #     print("Time: " + str(list(training_time.items())))
            #     # tracking epsilons
            #     epsilons.append(agent.e)

            #     # terminal condition
            #     if max_reward > RENDER_REWARD_MIN: RENDER_ENV = True

            #     break

    # summary
    np.save("rewards_over_time", reward_hist)
    # np.save("mean100", mean_100)   

    # plots
    # plt.figure(figsize=(10,8))
    # plt.plot(epsilons)
    # plt.xlabel("Episodes")
    # plt.ylabel("Epsilon value")
    # plt.title("Epsilon Vs Episodes")
    # plt.savefig("Epsilon.png") 

    plt.figure(figsize=(10,8))            
    plt.plot(reward_hist, label="Rewards")
    # plt.plot(reward_mean, label="Mean")
    # plt.plot(reward_var, label="Variance")    
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards per Episode")
    plt.legend(loc=0)
    plt.savefig("Rewards.png")         
    
    # plt.figure(figsize=(10,8))
    # plt.plot(mean_100)
    # plt.xlabel("100_episodes")
    # plt.ylabel  ("Mean_value")
    # plt.title('Average Reward per 100 episodes')
    # plt.savefig("mean_100.png")  










