import os
import os.path as osp
from timeit import default_timer as timer

import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

from DRLH.agents import Agent
from DRLH.utils.log.tensorboard_logging import *
from pprint import pprint
from DRLH.utils.utils import \
    state_rep_reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen, \
    state_rep_reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen, \
    state_rep_reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen, \
    state_rep_reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen, \
    find_last_model_index


class PPOMemory:
    def __init__(self, batch_size):
        self.states_raw = []
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state_raw, state, action, probs, vals, reward, done):
        self.states_raw.append(state_raw)
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states_raw = []
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha,
                 fc1_dims=256, fc2_dims=256, use_cuda=True):
        super(ActorNetwork, self).__init__()

        # self.actor = nn.Sequential(
        #     nn.Linear(*input_dims, fc1_dims),
        #     nn.LayerNorm(fc1_dims),
        #     nn.ReLU(),
        #     nn.Linear(fc1_dims, fc2_dims),
        #     nn.LayerNorm(fc2_dims),
        #     nn.ReLU(),
        #     nn.Linear(fc2_dims, n_actions),
        #     nn.Softmax(dim=-1)
        # )

        self.actor = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, n_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() and use_cuda else 'cpu')
        self.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        dist_ = self.actor(state)
        dist = Categorical(dist_)
        return dist, dist_


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, use_cuda=True):
        super(CriticNetwork, self).__init__()

        # self.critic = nn.Sequential(
        #     nn.Linear(*input_dims, fc1_dims),
        #     nn.LayerNorm(fc1_dims),
        #     nn.ReLU(),
        #     nn.Linear(fc1_dims, fc2_dims),
        #     nn.LayerNorm(fc2_dims),
        #     nn.ReLU(),
        #     nn.Linear(fc2_dims, 1)
        # )

        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() and use_cuda else 'cpu')
        self.to(self.device)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state):
        value = self.critic(state)
        return value


class DRL_Agent(Agent):

    def __init__(self, problem, gamma=0.99, alpha=0.0003, gae_lambda=0.95,  # gae_lambda=0.5 # gae_lambda=0.95
                 policy_clip=0.2, batch_size=64, n_epochs=10, logdir=None, normalization_func="no_normalization",
                 use_cuda=True, n_steps_look_into_future=10, last_100k_size=1000000):
        super(DRL_Agent, self).__init__(n_actions=problem.action_space.n)
        self.problem = problem
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.n_steps_look_into_future = n_steps_look_into_future
        self.actor = ActorNetwork(problem.action_space.n, problem.observation_space.shape, alpha, use_cuda=use_cuda)
        self.critic = CriticNetwork(problem.observation_space.shape, alpha, use_cuda=use_cuda)
        self.memory = PPOMemory(batch_size)
        #
        self.welford_state_n = None
        self.welford_state_mean = None
        self.welford_state_mean_diff = None
        self.max_norm_values = None
        self.num_bool_feat = 3 + self.n_actions
        self.last_100k_states = np.tile(np.nan, (last_100k_size, problem.observation_space.shape[0]))
        self.last_100k_mean = np.tile(0, problem.observation_space.shape[0])
        self.last_100k_std = np.tile(1, problem.observation_space.shape[0])
        self.last_100k_counter = 0
        self.last_100k_size = last_100k_size
        self.logdir = logdir
        #
        self.baseline_rewards = []
        #
        self.state_rep = problem.state_rep
        self.normalization_func = {
            "max_normalize": self.max_normalize,
            # "welford_normalize": self.welford_normalize_state,
            "last_100k_normalize": self.last_100k_normalize,
            "no_normalization": lambda obs, up=True: np.array(obs)
            # T.tensor([obs], dtype=T.float).to(self.actor.device)
        }.get(normalization_func, "not a valid normalization func")

    def remember(self, state_raw, action, probs, vals, reward, done):
        state = self.normalization_func(state_raw)
        self.memory.store_memory(state_raw, state, action, probs, vals, reward, done)

    def save_model(self, i_or_path):
        print('... saving models ...')
        checkpoint = {
            'epoch': i_or_path if isinstance(i_or_path, int) else 0,
            'actor_state_dict': self.actor.state_dict(),
            'actor_optimizer': self.actor.optimizer.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_optimizer': self.critic.optimizer.state_dict(),
        }
        if self.normalization_func == self.last_100k_normalize:
            checkpoint['last_100k_mean'] = self.last_100k_mean
            checkpoint['last_100k_std'] = self.last_100k_std
        
        # Handle both path and episode number
        if isinstance(i_or_path, str):
            # Full path provided
            save_path = i_or_path
        else:
            # Episode number provided, construct path
            save_path = osp.join(self.logdir, "models", f"checkpoint_{i_or_path}.pt")
            
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        T.save(checkpoint, save_path)
        print("...also saved mean/std")

    def load_model(self, logdir_or_path=None, i=None):
        print('... loading models ...')
        
        # Handle both full file paths and directory+episode patterns
        if logdir_or_path.endswith('.pt'):
            # Full path provided
            model_path = logdir_or_path
        else:
            # Directory and episode number provided
            model_path = osp.join(logdir_or_path, "models", f"checkpoint_{i}.pt")
            
        checkpoint = T.load(model_path)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        if self.normalization_func == self.last_100k_normalize:
            self.last_100k_mean = checkpoint['last_100k_mean']
            self.last_100k_std = checkpoint['last_100k_std']
            print("...also loaded last_100k_mean=", self.last_100k_mean, "\nand last_100k_std=", self.last_100k_std)

    def load_model_legacy(self, actor_checkpoint=None, critic_checkpoint=None, last_100k_mean=None, last_100k_std=None):
        print('... loading models ...')

        self.actor.load_state_dict(T.load(actor_checkpoint))
        self.critic.load_state_dict(T.load(critic_checkpoint))
        # last_100k_checkpoint = '/'.join(actor_checkpoint.split("/")[:-1])
        if self.normalization_func == self.last_100k_normalize:
            self.last_100k_mean = T.load(f"{last_100k_mean}")  # T.load(f"{last_100k_checkpoint}/last_100k_mean.pt")
            self.last_100k_std = T.load(f"{last_100k_std}")  # T.load(f"{last_100k_checkpoint}/last_100k_std.pt")
        print("...also loaded last_100k_mean=", self.last_100k_mean, "\nand last_100k_std=", self.last_100k_std)

    def choose_action(self, observation):
        state = self.normalization_func(observation)
        state = T.tensor([state], dtype=T.float, device=self.actor.device)

        dist, dist_ = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = T.squeeze(dist.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value, dist_

    def learn(self, update_normalization=True):
        if self.normalization_func == self.last_100k_normalize and len(
                self.memory.states_raw) > 0 and update_normalization:
            print("updating mean/std for normalization...")
            statesx = np.stack(self.memory.states_raw)
            statesx[:, -self.num_bool_feat:] = 0  # in order to get mean=0 for bool features (should not be scaled)
            self.last_100k_states[self.last_100k_counter:self.last_100k_counter + len(statesx), :] = statesx
            self.last_100k_mean = np.nanmean(self.last_100k_states, axis=0)
            self.last_100k_std = np.nanstd(self.last_100k_states, axis=0)
            self.last_100k_std[
            -self.num_bool_feat:] = 1  # in order to get std=1 for bool features (should not be scaled)
            self.last_100k_counter = (self.last_100k_counter + len(statesx)) % self.last_100k_size

        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, min(t + self.n_steps_look_into_future,
                                      len(reward_arr) - 1)):  # range(t, min(t+10, len(reward_arr) - 1)):  # range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] *
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)

            values = T.tensor(values).to(self.actor.device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                dist, dist_ = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

    def seed(self, seed=None):
        T.manual_seed(seed)

    def reset(self, *info):
        if self.normalization_func == self.max_normalize:
            sd, ni = info
            max_norm_values = {  # would like to make this more general, but don't know how
                state_rep_reduced_dist___dist_from_min___no_improvement___index_step___was_changed___unseen: [sd, sd,
                                                                                                              ni, ni],
                state_rep_reduced_dist___dist_from_min___dist___min_dist___no_improvement___index_step___was_changed___unseen: [
                    sd, sd, sd, sd, ni, ni],
                state_rep_reduced_dist___dist_from_min___temp___cs___no_improvement___index_step___was_changed___unseen: [
                    sd,
                    sd, 1,
                    1, ni,
                    ni],
                state_rep_reduced_dist___dist_from_min___dist___min_dist___temp___cs___no_improvement___index_step___was_changed___unseen: [
                    sd, sd, sd, sd, 1, 1, ni, ni]
            }.get(self.state_rep, "not valid state representation")
            max_norm_values += [1] * (3 + self.n_actions)
            # self.max_norm_values = T.tensor(max_norm_values, dtype=T.float, device=self.actor.device)
            self.max_norm_values = np.array(max_norm_values)

    def max_normalize(self, observation, update=True):
        return observation / self.max_norm_values

    def last_100k_normalize(self, observation, update=True):
        state = (np.array(observation) - self.last_100k_mean) / self.last_100k_std
        return state

    def remember_baseline_reward(self, reward, n_steps=None):
        if n_steps is None:
            self.baseline_rewards.append(reward)
        else:
            self.baseline_rewards[n_steps] += reward

    def forget_baseline_rewards(self):
        self.baseline_rewards = []

    def train(self, max_samples_train, learning_rate=0.001, logging=False, log_single_interval=1000, save_model=True,
              save_model_interval=-1, baseline_results=None, resume=False, verbose=True, print_every_n_step=100):

        current_epoch = 1

        if logging:
            if not osp.exists(self.logdir):
                os.makedirs(self.logdir)
            writer = mywriter(self.logdir)

        if save_model:
            if resume:
                current_epoch = find_last_model_index(osp.join(self.logdir, "models"))
                self.load_model(self.logdir, current_epoch)
            if not osp.exists(osp.join(self.logdir, "models")):
                os.makedirs(osp.join(self.logdir, "models"))

        save_model_interval = max_samples_train if save_model_interval < 0 else save_model_interval

        for i in range(current_epoch, max_samples_train + 1):
            observation = self.problem.reset()
            self.reset(self.problem.start_distance, self.problem.max_steps)
            done = False
            score = 0
            n_steps = 0
            action_sequence = []
            dist_sequence = []
            min_dist_sequence = []
            tic = timer()
            print("instance:", i)
            while not done:
                action, prob, val, dist_ = self.choose_action(observation)
                observation_, reward, done, info = self.problem.step(action)

                self.remember(observation, action, prob, val, reward, done)
                n_steps += 1
                score += reward
                observation = observation_
                action_sequence.append(action)
                dist_sequence.append(info["distance"])
                min_dist_sequence.append(info["min_distance"])

                if n_steps % print_every_n_step == 0 and verbose:
                    print('Iteration {:03d}, Min Dist: {:.4f}, Start Dist: {:.4f}, Improved Dist: {:.4f}'
                          'Total Reward until now: {}'.format(n_steps,
                                                              info["min_distance"], info["start_distance"],
                                                              info["start_distance"] - info["min_distance"],
                                                              score
                                                              ))

                if i % log_single_interval == 0 and logging:
                    log_single_search(writer, info, i - 1, n_steps, dist_=dist_)
                    writer.flush()

            self.learn()
            if i % save_model_interval == 0 and save_model:
                self.save_model(i)
            total_time = timer() - tic
            baseline_scores = np.loadtxt(
                f"{baseline_results}/RESULT.txt").tolist() if baseline_results is not None else None  # refresh
            if logging:
                log_training(writer, info, i - 1, score=score, baseline_scores=baseline_scores)
                writer.flush()

            with open(osp.join(self.logdir, 'Result.txt'), "a") as file:
                file.write(str(info["min_distance"]) + "\n")
            with open(osp.join(self.logdir, 'ACTION_SEQUENCE.txt'), "a") as file:
                file.write(str(action_sequence) + "\n")
            with open(osp.join(self.logdir, 'DIST_SEQUENCE.txt'), "a") as file:
                file.write(str(dist_sequence) + "\n")
            with open(osp.join(self.logdir, 'MIN_DIST_SEQUENCE.txt'), "a") as file:
                file.write(str(min_dist_sequence) + "\n")
            with open(osp.join(self.logdir, 'TIME.txt'), "a") as file:
                file.write(str(total_time) + "\n")
            with open(osp.join(self.logdir, 'BEST_SOL.txt'), 'a') as file:
                file.write(str(info["best_solution"]) + "\n")
        if verbose:
            print('episode', i, 'min_distance %.3f' % info["min_distance"])
            pprint(info['action_counter'])

        if logging:
            writer.close()

    def solve(self, problem, logging=False, baseline_results=None, log_single_interval=100, verbose=True):
        if logging:
            if not os.path.exists(self.logdir):
                os.makedirs(self.logdir)
            writer = SummaryWriter(self.logdir)
        if not isinstance(problem, type(self.problem)):
            raise

        for i in range(len(problem)):
            # writer = SummaryWriter(logdir)
            observation = problem.reset()
            self.reset(problem.start_distance, problem.max_steps)
            done = False
            score = 0
            n_steps = 0
            action_sequence = []
            dist_sequence = []
            min_dist_sequence = []
            tic = timer()
            print("instance:", i)
            while not done:
                action, prob, val, dist_ = self.choose_action(observation)
                observation_, reward, done, info = problem.step(action)
                n_steps += 1
                score += reward
                observation = observation_
                action_sequence.append(action)
                dist_sequence.append(info["distance"])
                min_dist_sequence.append(info["min_distance"])
                if n_steps % 100 == 0 and verbose:
                    print('Iteration {:03d}, Min Dist: {:.4f}, Start Dist: {:.4f}, Improved Dist: {:.4f}'
                          'Total Reward until now: {}'.format(n_steps,
                                                              info["min_distance"], info["start_distance"],
                                                              info["start_distance"] - info["min_distance"],
                                                              score
                                                              ))
                if i % log_single_interval == 0 and logging:
                    log_single_search(writer, info, i, n_steps, dist_=dist_)
            total_time = timer() - tic
            # TODO
            # print the resutls for instance i and add them to a list for return at the end
            if logging:
                with open(f"{self.logdir}RESULT.txt", "a") as file:
                    file.write(str(info["min_distance"]) + "\n")
                with open(f"{self.logdir}ACTION_SEQUENCE.txt", "a") as file:
                    file.write(str(action_sequence) + "\n")
                with open(f"{self.logdir}DIST_SEQUENCE.txt", "a") as file:
                    file.write(str(dist_sequence) + "\n")
                with open(f"{self.logdir}MIN_DIST_SEQUENCE.txt", "a") as file:
                    file.write(str(min_dist_sequence) + "\n")
                with open(f"{self.logdir}TIME.txt", "a") as file:
                    file.write(str(total_time) + "\n")
                with open(osp.join(self.logdir, 'BEST_SOL.txt'), 'a') as file:
                    file.write(str(info["best_solution"]) + "\n")

        if logging:
            writer.close()
            # writer.close()
        if verbose:
            print('episode', i, 'min_distance %.3f' % info["min_distance"])
        
        # Return the best solution and cost found
        return info["best_solution"], info["min_distance"]
