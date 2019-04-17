from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from mpi4py import MPI
from collections import deque

#
import random
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

#from tensorboardX import SummaryWriter

ob_reset = []

def path_segment_generator(pi, env, horizon_path, stochastic, start_observation=None, start_state=None):
    # generate a path, a small trajectory that has a defined start state.
    # ends by horizon_path or if done
    ac = env.action_space.sample()  # not used, just so we have the datatype

    #
    # TODO start_state !!!
    # simulation global forces and positions ?...
    #
    if start_observation == None:
        # use any default env start state
        env.metadata['gen_init_state'] = True
        ob = env.reset()
    else:
        # set start state
        env.metadata['gen_init_state'] = False
        env.metadata['new_init_ob']= start_observation
        ob = env.reset()


    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon_path)])
    rews = np.zeros(horizon_path, 'float32')
    vpreds = np.zeros(horizon_path, 'float32')
    news = np.zeros(horizon_path, 'int32')
    acs = np.array([ac for _ in range(horizon_path)])
    prevacs = acs.copy()

    i = 0
    while True:
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)

        obs[i] = ob  # observation
        vpreds[i] = vpred  # predicted value
        news[i] = new  # done
        acs[i] = ac  # action
        prevacs[i] = prevac  # previous action

        ob, rew, new, _ = env.step(ac)

        rews[i] = rew  # reward

        if new or i >= horizon_path:
            return {"ob": obs[:i], "rew": rews[:i], "vpred": vpreds[:i], "new": news[:i],
                    "ac": acs, "prevac": prevacs, "nextvpred": vpred * (1 - new)}

        i += 1




# vine
def traj_segment_generator(pi, env, horizon, stochastic):
    """
    vine
    generates a amount of horizon time steps
    does a episode with default init, then horizon_path_number paths with max horizon_path time steps

    :param pi:
    :param env:
    :param horizon:
    :param stochastic:
    :return:
    """


    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    env.metadata['gen_init_state'] = True
    ob = env.reset()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()


    #my TODO
    paths_per_episode = 0 # 3
    horizon_path = 100
    use_epV_at_rept = False

    path_start = 0

    cur_path_ret = 0
    cur_path_len = 0
    path_rets = []
    path_lens = []

    do_paths = False
    path_nr = 0
    last_ep_start = 0
    last_ep_end = 42

    while True:

        # get action
        prevac = ac
        ac, vpred = pi.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        # t timesteps
        #if t > 0 and t >= horizon:
        if t > 0 and t % horizon == 0:
            # my
            print("t: %i" % (t))

            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                   "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                   "ep_rets" : ep_rets, "ep_lens" : ep_lens,
                   "path_rets" : path_rets, "path_lens" : path_lens}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
            path_rets = []
            path_lens = []

        i = t % horizon

        # save data
        obs[i] = ob # observation
        vpreds[i] = vpred # predicted value
        news[i] = new # done
        acs[i] = ac # action
        prevacs[i] = prevac # previous action

        # run action
        ob, rew, new, _ = env.step(ac)

        rews[i] = rew  # reward

        #if int(t / horizon) % 5 == 0:
        #if t > 30000:
        #    env.render()

        if do_paths == False:
            # doing episode

            # in episode
            cur_ep_ret += rew
            cur_ep_len += 1

            if new == True:
                # end of episode
                ep_rets.append(cur_ep_ret)
                ep_lens.append(cur_ep_len)
                cur_ep_ret = 0
                cur_ep_len = 0

                # start doing paths
                do_paths = True
                last_ep_end = t
                new = True

                logger.log('End episode at t: %i, with episode length: %i' % (last_ep_end, ep_lens[-1]))
                # TODO reset to path

        if do_paths == True: # TODO check or elif??
            # doing paths

            if last_ep_end != t:
                # in path
                cur_path_ret += rew
                cur_path_len += 1

            if new == True or cur_path_len >= horizon_path:
                # do next path reset

                # dont add the start of first path, if then just do the init first
                if path_nr != 0:

                    # use_epV_at_rept
                    # set the reward to the value of the episode
                    if use_epV_at_rept:


                        p_end_t = path_start + cur_path_len
                        # if path passed the episode end or done the reward is the right value
                        # change the last reward!
                        if p_end_t < last_ep_end and (not new):
                            idx = (path_start + cur_path_len) % horizon
                            #rews[i] = vpreds[idx]
                            #cur_path_ret += -rew + rews[i]

                            # why not estimate the real V at path_end?
                            _, path_end_vpred = pi.act(stochastic, ob)
                            rews[i] = path_end_vpred - rew
                            cur_path_ret += rews[i] # -rew

                            logger.log("Path: Change reward %.2f to Value %.2f (episode had %.2f at that t) length: %i" % (rew, rews[i], vpreds[idx], cur_path_len))

                        else:
                            # testing
                            logger.log("Path: Done or use true reward %.2f ( is farther then episode) length: %i" %(rew, cur_path_len) )


                    path_rets.append(cur_path_ret)
                    path_lens.append(cur_path_len)

                cur_path_ret = 0
                cur_path_len = 0

                # stop path even env is not done
                new = True

                if path_nr < paths_per_episode:
                    # do new path init

                    # TODO
                    # set_global_seeds(seed)
                    # np_random, seed2 = seeding.np_random(seed)

                    # init random
                    #path_start = random.randint(last_ep_start, last_ep_end)

                    # init at mid
                    #path_start = last_ep_start + int((last_ep_end - last_ep_start) / 2)

                    # init at start and set path horizon to 2/3
                    #path_start = last_ep_start
                    #horizon_path = (last_ep_end - last_ep_start) / 2

                    # init at start and end at 1/2 and take V at end from episode
                    #path_start = last_ep_start
                    #horizon_path = int((last_ep_end - last_ep_start) / 2)
                    #use_epV_at_rept = True

                    # random path_horizon and take V
                    #path_start = last_ep_start
                    #horizon_path = random.randint(10, int((last_ep_end-path_start)*1.25))
                    #use_epV_at_rept = True

                    # random start, random path_horizon and take V
                    path_start = random.randint(last_ep_start, last_ep_end)
                    horizon_path = random.randint(0, last_ep_end-path_start+10)
                    use_epV_at_rept = True


                    init_idx = path_start % horizon
                    start_observation = obs[init_idx]

                    logger.log('Start path nr %i at: %i, t: %i, e-length: %i, e-start: %i, e-end: %i' % (
                        path_nr, path_start - last_ep_start, t, last_ep_end - last_ep_start, last_ep_start, last_ep_end))

                    path_nr += 1

                    env.metadata['gen_init_state'] = False
                    env.metadata['new_init_ob'] = start_observation
                    ob = env.reset()

                else:
                    # do new episode init
                    do_paths = False
                    path_nr = 0

                    # TODO
                    # path_t = -1
                    last_ep_start = t + 1

                    # use any default env start state
                    env.metadata['gen_init_state'] = True
                    logger.log('Start episode at t: %i' % (last_ep_start))
                    ob = env.reset()

        t += 1


def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]


        # TODO
        # # nextvpred is 0 ... why?, not good!
        if nonterminal == 0:
            # next is terminal
            #print("nextvpred", seg["nextvpred"])
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t] * nonterminal
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

            print('last 6 adv]',
                  seg["adv"][t-6],
                  seg["adv"][t-5],
                  seg["adv"][t-4],
                  seg["adv"][t-3],
                  seg["adv"][t-2],
                  seg["adv"][t-1],
                  seg["adv"][t-0])

            print('last 6 vpred]',
                  seg["vpred"][t - 6],
                  seg["vpred"][t - 5],
                  seg["vpred"][t - 4],
                  seg["vpred"][t - 3],
                  seg["vpred"][t - 2],
                  seg["vpred"][t - 1],
                  seg["vpred"][t - 0])
        else:
            # GAE
            delta = rew[t] + gamma * vpred[t + 1] * nonterminal - vpred[t]  # r_t + gamma*V(s_(t+1)) - V(s_t))
            gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam  # sum

    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(env, policy_func, *,
        timesteps_per_actorbatch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant' # annealing for stepsize parameters (epsilon and adam)
        ):
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
    pol_surr = - tf.reduce_mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
    vf_loss = tf.reduce_mean(tf.square(pi.vpred - ret))
    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)

    U.initialize()
    adam.sync()

    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator(pi, env, timesteps_per_actorbatch, stochastic=True)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=100) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    # my
    #sess = tf.get_default_session()
    #writer = tf.summary.FileWriter("/home/chris/openai_logdir/tb2")
    #placeholder_rews = tf.placeholder(tf.float32)
    #placeholder_vpreds = tf.placeholder(tf.float32)
    #placeholder_advs = tf.placeholder(tf.float32)
    #placeholder_news = tf.placeholder(tf.float32)
    #placeholder_ep_rews = tf.placeholder(tf.float32)

    #tf.summary.histogram("rews", placeholder_rews)
    #tf.summary.histogram("vpreds", placeholder_vpreds)
    #tf.summary.histogram("advs", placeholder_advs)
    #tf.summary.histogram("news", placeholder_news)
    #tf.summary.scalar("ep_rews", placeholder_ep_rews)

    #writer = SummaryWriter("/home/chris/openai_logdir/tb2")


    #placeholder_ep_rews = tf.placeholder(tf.float32)
    #placeholder_ep_vpred = tf.placeholder(tf.float32)
    #placeholder_ep_atarg = tf.placeholder(tf.float32)

    #sess = tf.get_default_session()
    #writer = tf.summary.FileWriter("/home/chris/openai_logdir/x_new/tb2")
    #tf.summary.scalar("EpRews", placeholder_ep_rews)
    #tf.summary.scalar("EpVpred", placeholder_ep_vpred)

    #tf.summary.scalar("EpRews", placeholder_ep_rews)
    #tf.summary.scalar("EpVpred", placeholder_ep_vpred)
    #tf.summary.scalar("EpAtarg", placeholder_ep_atarg)
    #summ = tf.summary.merge_all()

    time_step_idx = 0

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()

        # Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
        add_vtarg_and_adv(seg, gamma, lam)

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        vpredbefore = seg["vpred"] # predicted value function before udpate

        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate




        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult)
                losses.append(newlosses)
            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        # ep_lens and ep_rets
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        #TODO make MPI compatible
        path_lens = seg["path_lens"]
        path_rews = seg["path_rets"]

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        timesteps_so_far += sum(path_lens)# my add path lens
        print("timesteps_so_far: %i" % timesteps_so_far)

        iters_so_far += 1
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        #my
        logger.record_tabular("PathLenMean", np.mean(path_lens))
        logger.record_tabular("PathRewMean", np.mean(path_rews))


        # MY
        dir = "/home/chris/openai_logdir/"
        env_name = env.unwrapped.spec.id

        #plots

        plt.figure(figsize=(10, 4))
        # plt.plot(vpredbefore, animated=True, label="vpredbefore")
        #new_with_nones = seg["new"]
        #np.place(new_with_nones, new_with_nones == 0, [6])
        #plt.plot(new_with_nones, 'r.', animated=True, label="new")

        for dd in np.where(seg["new"] == 1)[0]:
            plt.axvline(x= dd, color='green', linewidth=1)
            #plt.annotate('asd', xy=(2, 1), xytext=(3, 1.5),)

        #np.place(new_episode_with_nones, new_episode_with_nones == 0, [90])
        #aaaa = np.where(seg["news_episode"] > 0)
        #for dd in aaaa[0]:
        #    plt.annotate('ne'+str(dd), xy=(0, 6), xytext=(3, 1.5),)
            #plt.axvline(x=dd, color='green', linewidth=1)

        #plt.plot(ac, animated=True, label="ac")

        plt.plot(vpredbefore, 'g', label="vpredbefore", antialiased=True)

        # plot advantage of episode time steps
        #plt.plot(seg["adv"], 'b', animated=True, label="adv")
        #plt.plot(atarg, 'r', animated=True, label="atarg")
        plt.plot(tdlamret, 'y', animated=True, label="tdlamret")

        plt.legend()
        plt.title('iters_so_far: ' + str(iters_so_far))
        plt.savefig(dir + env_name+'_plot.png', dpi=300)

        if iters_so_far % 2 == 0 or iters_so_far == 0:
            #plt.ylim(ymin=-10, ymax=100)
            #plt.ylim(ymin=-15, ymax=15)

            plt.savefig(dir+ '/plotiters/' + env_name + '_plot' + '_iter' + str(iters_so_far).zfill(3) + '.png', dpi=300)

        plt.clf()
        plt.close()

        # 3d V obs
        #ac, vpred = pi.act(True, ob)
        # check ob dim
        freq = 1 # 5
        # <= 3?
        if env.observation_space.shape[0] <= 3 and (iters_so_far % freq == 0 or iters_so_far == 1):

            figV, axV = plt.subplots()

            # surface
            #figV = plt.figure()
            #axV = Axes3D(figV)

            obs = env.observation_space

            X = np.arange(obs.low[0], obs.high[0], (obs.high[0] - obs.low[0]) / 30)
            Y = np.arange(obs.low[1], obs.high[1], (obs.high[1] - obs.low[1]) / 30)
            X, Y = np.meshgrid(X, Y)

            Z = np.zeros((len(X),len(Y)))

            for x in range(len(X)):
                for y in range(len(Y)):
                    #strange datatype needed??
                    myob = np.copy(ob[0])
                    myob[0] = X[0][x]
                    myob[1] = Y[y][0]
                    stochastic = True
                    ac, vpred = pi.act(stochastic, myob)
                    Z[x][y] = vpred

            plt.xlabel('First D')
            plt.ylabel('Second D')
            #plt.clabel('Value-Function')
            plt.title(env_name + ' iteration: ' + str(iters_so_far))

            # surface
            #axV.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm)

            # heat map
            imgplot = plt.imshow(Z, interpolation='nearest')
            imgplot.set_cmap('hot')
            plt.colorbar()

            figV.savefig(dir + env_name + '_V.png', dpi=300)

            if iters_so_far % 2 == 0 or iters_so_far == 0:
                figV.savefig(dir + '/plotiters/' + env_name + '_plot' + '_iter' + str(iters_so_far).zfill(3) + '_V' + '.png',
                            dpi=300)

            figV.clf()
            plt.close(figV)



        """
        # transfer timesteps of iterations into timesteps of episodes
        idx_seg = 0
        for ep in range(len(lens)) :
            all_ep = episodes_so_far - len(lens) + ep

            if all_ep%100==0:
                break

            ep_rews = seg["rew"][idx_seg:idx_seg+lens[ep]]
            ep_vpred = seg["vpred"][idx_seg:idx_seg+lens[ep]]
            ep_atarg = atarg[idx_seg:idx_seg+lens[ep]]

            idx_seg += lens[ep]
            #writer.add_histogram("ep_vpred", data, iters_so_far)
            #hist_dict[placeholder_ep_rews] = ep_rews[ep]
            #sess2 = sess.run(summ, feed_dict=hist_dict)

            #summary = tf.Summary()

            #if test_ep:
            #    break
            for a in range(len(ep_rews)):
                d = ep_rews[a]
                d2 = ep_vpred[a]
                d3 = ep_atarg[a]

                #summary.value.add(tag="EpRews/"+str(all_ep), simple_value=d)
                #writer.add_summary(summary, a)

                sess2 = sess.run(summ, feed_dict={placeholder_ep_rews: d, placeholder_ep_vpred: d2, placeholder_ep_atarg: d3})
            #writer.add_summary(sess2, global_step=a)
                writer.add_summary(sess2, global_step=time_step_idx)
                time_step_idx += 1

            writer.flush()
            #writer.close()
            #logger.record_tabular("vpred_e" + str(all_ep), ep_rews[ep])

        """

        """
        ###
        sess2 = sess.run(summ, feed_dict={placeholder_rews: seg["rew"],
                                          placeholder_vpreds: seg["vpred"],
                                          placeholder_advs: seg["adv"],
                                          placeholder_news: seg["new"]})
        writer.add_summary(sess2, global_step=episodes_so_far)
        """


        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
