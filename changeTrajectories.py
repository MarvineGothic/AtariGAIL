import os.path as osp
import pickle as pkl
import numpy as np
from tqdm import tqdm


def mergeFiles(fileNames, name):
    assert isinstance(fileNames, list)
    assert len(fileNames) > 0
    result = []

    for fn in range(len(fileNames)):
        if osp.exists(fileNames[fn]):
            with open(fileNames[fn], 'rb') as rfp:
                file = pkl.load(rfp)
                for w in range(len(file)):
                    result.append(file[w])

    taskName = 'deterministic.human.' + name + '.' + str(len(result)) + '_traj.1_window.pkl'
    pkl.dump(result, open(taskName, "wb"))


def shrink_trajectories(fileName, max_traj_length):
    with open(fileName, 'rb') as rfp:
        file = pkl.load(rfp)

    obs = []
    acs = []
    rets = []
    currentEpisodeReward = 0

    ob = "ob"
    ac = "ac"
    rw = "rew"
    ep_ret = "ep_ret"

    sampleTrajectory = []
    for traj in tqdm(file):

        try:
            traj[ep_ret]
        except:
            ob = "observations"
            ac = "actions"
            rw = "rewards"
            ep_ret = "episode_rewards"

        trajectoryData = traj
        if len(traj[ob]) > max_traj_length:
            print(len(traj[ob]))

            for i in range(max_traj_length):
                obs.append(traj[ob][i])
                acs.append(traj[ac][i])
                rets.append(traj[rw][i])
            currentEpisodeReward = sum(rets)
            trajectoryData = {ob: np.array(obs),
                              ac: np.array(acs),
                              rw: np.array(rets),
                              ep_ret: currentEpisodeReward
                              }
            obs = []
            acs = []
            rets = []
        sampleTrajectory.append(trajectoryData)

        ob = "ob"
        ac = "ac"
        rw = "rew"
        ep_ret = "ep_ret"

    taskName = fileName.split('.')
    taskName = taskName[0] + '.' + taskName[1] + '.'
    taskName = taskName + str(len(sampleTrajectory)) + '_traj_size_' + str(max_traj_length) + '.pkl'
    pkl.dump(sampleTrajectory, open(taskName, "wb"))
    return taskName


def printTrajLengths(fileName):
    with open(fileName, 'rb') as rfp:
        file = pkl.load(rfp)
        for trajNr in range(len(file)):
            traj = file[trajNr]
            try:
                print(len(traj['ob']))
            except:
                print(len(traj['observations']))


def cut_trajectories(fileName, numTraj):
    with open(fileName, 'rb') as rfp:
        file = pkl.load(rfp)
    sampleTrajectory = []
    for traj in range(numTraj):
        sampleTrajectory.append(file[traj])
    taskName = fileName.replace("pkl", "")
    # taskName = taskName[0] + '.' + taskName[1] + '.'
    taskName = taskName + str(numTraj) + '_traj_size_' + str(len(sampleTrajectory[0]['ob'])) + '.pkl'
    pkl.dump(sampleTrajectory, open(taskName, "wb"))
    return taskName


def main():
    fileName = 'data/expert/stochastic.trpo.Boxing-ram-v0.MD.1500.score_100-0.pkl'
    # files = ['human.MontezumaRevenge_MLP.10_traj.1_window.pkl','deterministic.human.MontezumaRevengeMLP.178_traj.1_window.pkl']

    # mergeFiles(files, "MontezumaRevengeMLP")

    #shrink_trajectories(fileName, 100)

    fileName = cut_trajectories(fileName, 1)
    printTrajLengths(fileName)

    #cut_trajectories(fileName)


if __name__ == '__main__':
    main()
