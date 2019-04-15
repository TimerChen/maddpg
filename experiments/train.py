import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers
from maddpg.trainer.maml import MAML

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="default", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=50, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

    # About Opponent Modeling
    parser.add_argument("--use-maml", action="store_true", default=False)
    parser.add_argument("--use-om", action="store_true", default=False)
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers

experience_save_dir = "./experiences/"

def write_summary(writer, summary, fd, train_step):
    summary_tmp = tf.summary.merge(summary)
    summary_result = U.get_session().run(summary_tmp, feed_dict = fd)
    writer.add_summary(summary_result, train_step)

def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        agent_qloss = [[] for _ in range(env.n)]
        agent_ploss = [[] for _ in range(env.n)]
        opponnet_loss = [[] for _ in range(env.n)]

        # ??? obs_shape & num_action?
        print("obs_shape", obs_shape_n)
        obs_size = 0
        for i in obs_shape_n:
            obs_size = max(i[0], obs_size)

        print("My action space:", env.action_space)
        maml = MAML(U.get_session(), "", (obs_size,), 5, len(trainers))

        num_agents = len(trainers)



        # Set summary
        with tf.name_scope("summary"):
            summary_agent_qloss_ph = []
            summary_agent_ploss_ph = []
            summary_agent_reward_ph = []
            summary_total_reward_ph = []
            summary_opponent_loss_ph = []
            summary_opponent_test_loss_ph = []

            summary_only_done = dict()
            summary_every_step = dict()

            # Loss of Agent
            with tf.name_scope("agents"):
                a_qloss = []
                a_ploss = []
                a_rewards = []
                for i in range(num_agents):
                    qloss = tf.placeholder(tf.float32, None)
                    ploss = tf.placeholder(tf.float32, None)
                    reward = tf.placeholder(tf.float32, None)
                    summary_agent_qloss_ph.append(qloss)
                    summary_agent_ploss_ph.append(ploss)
                    summary_agent_reward_ph.append(reward)
                    qloss = tf.summary.scalar("agent_{}_qloss".format(i), qloss)
                    ploss = tf.summary.scalar("agent_{}_ploss".format(i), ploss)
                    reward = tf.summary.scalar("agent_{}_reward".format(i), reward)
                    a_qloss.append(qloss)
                    a_ploss.append(ploss)
                    a_rewards.append(reward)
                summary_only_done["agent_qloss"] = a_qloss
                summary_only_done["agent_ploss"] = a_ploss
                summary_only_done["agent_rewards"] = a_rewards

                reward = tf.placeholder(tf.float32, None)
                summary_total_reward_ph = reward
                reward = tf.summary.scalar("agent_total_reward".format(i), reward)
                summary_only_done["total_reward"] = reward

            # Loss of OM
            with tf.name_scope("opponnet"):
                o_loss = []
                t_loss = []
                for i in range(env.n):
                    loss = tf.placeholder(tf.float32, None)
                    summary_opponent_loss_ph.append(loss)
                    loss = tf.summary.scalar("agent_{}_loss".format(i), loss)
                    o_loss.append(loss)

                    loss = tf.placeholder(tf.float32, None)
                    summary_opponent_test_loss_ph.append(loss)
                    loss = tf.summary.scalar("agent_{}_test_loss".format(i), loss)
                    t_loss.append(loss)

                summary_opponent_total_test_loss_ph = \
                loss = tf.placeholder(tf.float32, None)
                loss = tf.summary.scalar("total_test_loss".format(i), loss)

                if arglist.use_om:
                    summary_only_done["opponnet_loss"] = o_loss
                    summary_only_done["opponnet_test_loss"] = t_loss
                    summary_only_done["opponnet_total_test_loss"] = [loss]

            # Loss of maml
            loss = tf.placeholder(tf.float32, None)
            summary_maml_loss_ph = loss
            loss = tf.summary.scalar("maml_loss", loss)
            if arglist.use_maml:
                summary_only_done["maml_loss"] = [loss]

            # Build FileWriter
            log_dir = "logs/" + arglist.exp_name + "/"
            if os.path.exists(log_dir):
                # os.removedirs(log_dir)
                print("Try to removing..")
                import shutil
                shutil.rmtree(log_dir)

            if os.path.exists(log_dir):
                raise "RM failed"
            writer = tf.summary.FileWriter(log_dir, U.get_session().graph)

        first_update = False

        def gather_experiences(num_episodes, store_func):
            nonlocal episode_step, train_step, obs_n
            for ii in range(num_episodes):
                while True:
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
                    # environment step
                    # print("action: ", action_n, np.sum(action_n, axis=1))
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    episode_step += 1
                    done = all(done_n)
                    terminal = (episode_step >= arglist.max_episode_len)
                    # collect experience
                    for i, agent in enumerate(trainers):
                        agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                        if agent.replay_buffer.check_need_save():
                            agent.replay_buffer.save(experience_save_dir, i)
                    # maml.store_data
                    store_func(obs_n, action_n)

                    obs_n = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards[-1] += rew
                        agent_rewards[i][-1] += rew

                    # for benchmarking learned policies
                    if arglist.benchmark:
                        for i, info in enumerate(info_n):
                            agent_info[-1][i].append(info_n['n'])
                        if train_step > arglist.benchmark_iters and (done or terminal):
                            file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                            print('Finished benchmarking, now saving...')
                            with open(file_name, 'wb') as fp:
                                pickle.dump(agent_info[:-1], fp)
                            break
                        continue

                    # for displaying learned policies
                    if arglist.display:
                        time.sleep(0.1)
                        env.render()
                        continue

                    if done or terminal:
                        obs_n = env.reset()
                        episode_step = 0
                        break

        print('Starting iterations...')
        while True:
            train_step += 1
            # MAML data
            gather_experiences(3, maml.store_data)

            # Train MAML
            if arglist.use_maml and first_update:
                maml_loss, _ = maml.train(train_step)
                if maml_loss is not None:
                    maml.update_real()

                    # Summary writer
                    feed_dict = dict()
                    feed_dict[summary_maml_loss_ph] = maml_loss
                    summary_tmp = summary_only_done["maml_loss"]
                    write_summary(writer, summary_tmp, feed_dict, train_step)

                    maml.clear_new()

            # Trainning data
            gather_experiences(1, maml.store_data)

            # Training
            mid_loss = 0
            if arglist.use_om and first_update:
                mid_loss = maml.train_real(train_step)
                if mid_loss is not None:
                    for i in range(env.n):
                        opponnet_loss[i].append(mid_loss[i])

                    # Summary writer
                    feed_dict = dict()
                    for i in range(num_agents):
                        feed_dict[summary_opponent_loss_ph[i]] = mid_loss[i]
                    summary_tmp = summary_only_done["opponnet_loss"]
                    write_summary(writer, summary_tmp, feed_dict, train_step)

            # Evaluation
            gather_experiences(2, maml.store_test_data)
            if arglist.use_om and first_update:
                loss = maml.evaluate()
                if loss is not None:
                    # Summary writer
                    feed_dict = dict()
                    for i in range(num_agents):
                        feed_dict[summary_opponent_test_loss_ph[i]] = loss[1][i]
                    feed_dict[summary_opponent_total_test_loss_ph] = loss[0]
                    summary_tmp = summary_only_done["opponnet_total_test_loss"] + summary_only_done["opponnet_test_loss"]
                    write_summary(writer, summary_tmp, feed_dict, train_step)

            # Summary writer of Data
            feed_dict = dict()
            feed_dict[summary_total_reward_ph] = episode_rewards[-1]
            for i in range(num_agents):
                feed_dict[summary_agent_reward_ph[i]] = agent_rewards[i][-1]

            summary_tmp = summary_only_done["agent_rewards"] + [summary_only_done["total_reward"]]
            write_summary(writer, summary_tmp, feed_dict, train_step)

            episode_rewards.append(0)
            for a in agent_rewards:
                a.append(0)
            agent_info.append([[]])

            # Update Each Agent
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                if arglist.use_om and first_update:
                    loss = agent.update(trainers, train_step, om=maml)
                else:
                    loss = agent.update(trainers, train_step)
                if loss is not None:
                    # Agent updated
                    if not first_update:
                        first_update = True
                        print("First update!!!")
                    agent_qloss[i].append(loss[0])
                    agent_ploss[i].append(loss[1])

            if loss is not None:
                # Summary writer
                feed_dict = dict()
                for i in range(num_agents):
                    feed_dict[summary_agent_qloss_ph[i]] = loss[0]
                    feed_dict[summary_agent_ploss_ph[i]] = loss[1]
                summary_tmp = summary_only_done["agent_qloss"] + summary_only_done["agent_ploss"]
                write_summary(writer, summary_tmp, feed_dict, train_step)

            # save model, display training output
            if (len(episode_rewards) % arglist.save_rate == 0):

                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                break


        # for i, agent in enumerate(trainers):
        #     if not agent.replay_buffer.check_need_save():
        #         agent.replay_buffer.save(experience_save_dir+arglist.exp_name, i)


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
