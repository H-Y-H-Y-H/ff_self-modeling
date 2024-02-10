import os
from torch.utils.data import Dataset, DataLoader
import time
import pybullet_data as pd
import pybullet as p
import gym
from model import *

class V000_sm_Env(gym.Env):
    def __init__(self, para, robot_camera=False, urdf_path='CAD2URDF'):

        self.stateID = None
        self.robotid = None
        # delta position
        self.v_p = None
        # orientation
        self.q = None
        # position
        self.p = None
        # initialize last position
        self.last_p = None
        # observation
        self.obs = [0] * 18
        # configurations for robot joints
        self.mode = p.POSITION_CONTROL
        self.maxVelocity = 1.5  # lx-224 0.20 sec/60degree = 5.236 rad/s
        self.force = 1.8
        self.sleep_time = 0  # decrease the value if it is too slow.

        self.joint_moving_idx = [0,1,2,3,4,5,6,7,8,9,10,11]
        self.n_sim_steps = 30
        self.motor_action_space = np.pi / 3
        self.urdf_path = urdf_path
        self.friction = 0.99
        self.robot_view_path = None
        # Each action contains 16 sub steps to achieve sin gait.
        self.sub_step_num = 16

        # CPG controller:
        self.initial_moving_joints_angle = self.sin_move(0,para)

        self.action_space = gym.spaces.Box(low=-np.ones(16, dtype=np.float32), high=np.ones(16, dtype=np.float32))
        self.observation_space = gym.spaces.Box(low=-np.ones(18, dtype=np.float32) * np.inf,
                                                high=np.ones(18, dtype=np.float32) * np.inf)

        self.log_obs = []
        self.log_action = []
        self.count = 0

        p.setAdditionalSearchPath(pd.getDataPath())

    def sin_move(self, ti, para, sep=16):
        # print(para)
        s_action = np.zeros(12)
        # print(ti)
        s_action[0] = para[0] * np.sin(ti / sep * 2 * np.pi + para[2]) + 0.2  # left   hind
        s_action[3] = para[1] * np.sin(ti / sep * 2 * np.pi + para[3]) + 0.45  # left   front
        s_action[6] = para[1] * np.sin(ti / sep * 2 * np.pi + para[4]) - 0.45  # right  front
        s_action[9] = para[0] * np.sin(ti / sep * 2 * np.pi + para[5]) - 0.2  # right  hind

        s_action[1] = para[6] * np.sin(ti / sep * 2 * np.pi + para[2]) - 0.5  # left   hind
        s_action[4] = para[7] * np.sin(ti / sep * 2 * np.pi + para[3]) - 0.7  # left   front
        s_action[7] = para[7] * np.sin(ti / sep * 2 * np.pi + para[4]) - 0.7  # right  front
        s_action[10] = para[6] * np.sin(ti / sep * 2 * np.pi + para[5]) - 0.5  # right  hind

        s_action[2] = para[8] * np.sin(ti / sep * 2 * np.pi + para[2]) + 0.28  # left   hind
        s_action[5] = para[9] * np.sin(ti / sep * 2 * np.pi + para[3]) + 0.54  # left   front
        s_action[8] = para[9] * np.sin(ti / sep * 2 * np.pi + para[4]) + 0.54  # right  front
        s_action[11] = para[8] * np.sin(ti / sep * 2 * np.pi + para[5]) + 0.28  # right  hind

        return s_action

    def get_obs(self):
        self.last_p = self.p
        # self.last_q = self.q
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)
        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        # Delta position and orientation
        self.v_p = self.p - self.last_p
        # self.v_q = self.q - self.last_q
        # if self.v_q[2] > 1.57:
        #     self.v_q[2] = self.q[2] - self.last_q[2] - 2 * np.pi
        # elif self.v_q[2] < -1.57:
        #     self.v_q[2] = (2 * np.pi + self.q[2]) - self.last_q[2]

        jointInfo = [p.getJointState(self.robotid, i) for i in self.joint_moving_idx]
        jointVals = np.array([[joint[0]] for joint in jointInfo]).flatten()
        self.obs = np.concatenate([self.v_p, self.q, jointVals]) # delta position, orientation and joint values.

        return self.obs

    def act(self, sin_para):
        for sub_step in range(self.sub_step_num):
            a = self.sin_move(sub_step, sin_para)
            a = np.clip(a, -1, 1)
            a *= self.motor_action_space

            for j in range(12):
                pos_value = a[j]
                p.setJointMotorControl2(self.robotid, self.joint_moving_idx[j], controlMode=self.mode,
                                        targetPosition=pos_value,
                                        force=self.force,
                                        maxVelocity=self.maxVelocity)

            for _ in range(self.n_sim_steps):
                p.stepSimulation()
            time.sleep(self.sleep_time)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath(pd.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        p.changeDynamics(planeId, -1, lateralFriction=self.friction)

        robotStartPos = [0, 0, 0.3]
        robotStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

        self.robotid = p.loadURDF(self.urdf_path, robotStartPos, robotStartOrientation, flags=p.URDF_USE_SELF_COLLISION,
                                  useFixedBase=0)

        p.changeDynamics(self.robotid, 2, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 5, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 8, lateralFriction=self.friction)
        p.changeDynamics(self.robotid, 11, lateralFriction=self.friction)

        for j in range(12):
            pos_value = self.initial_moving_joints_angle[j]
            p.setJointMotorControl2(self.robotid, self.joint_moving_idx[j], controlMode=self.mode,
                                    targetPosition=pos_value, force=self.force, maxVelocity=100)

        for _ in range(100):
            p.stepSimulation()
        self.stateID = p.saveState()
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)

        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        return self.get_obs()

    def resetBase(self):
        p.restoreState(self.stateID)

        self.last_p = 0
        self.p, self.q = p.getBasePositionAndOrientation(self.robotid)

        self.q = p.getEulerFromQuaternion(self.q)
        self.p, self.q = np.array(self.p), np.array(self.q)

        return self.get_obs()

    def step(self, a):

        self.act(a)
        obs = self.get_obs()

        r = 3 * obs[1] - abs(obs[5]) - 0.5 * abs(obs[0]) + 1
        Done = self.check()

        self.count += 1

        return obs, r, Done, {}

    def robot_location(self):
        position, orientation = p.getBasePositionAndOrientation(self.robotid)
        orientation = p.getEulerFromQuaternion(orientation)

        return position, orientation

    def check(self):
        pos, ori = self.robot_location()
        # if abs(pos[0]) > 0.5:
        #     abort_flag = True
        if abs(ori[0]) > np.pi / 6 or abs(ori[1]) > np.pi / 6: # or abs(ori[2]) > np.pi / 6:
            abort_flag = True
        # elif pos[1] < -0.04:
        #     abort_flag = True
        # elif abs(pos[0]) > 0.2:
        #     abort_flag = True
        else:
            abort_flag = False
        return abort_flag



class SAS_data(Dataset):
    def __init__(self, SAS_data):
        self.all_S = SAS_data[0]
        self.all_A = SAS_data[1]
        self.all_NS = SAS_data[2]
        self.all_DONE = SAS_data[3]
        self.all_R = SAS_data[4]

    def __getitem__(self, idx):
        S = self.all_S[idx]
        A = self.all_A[idx]
        NS = self.all_NS[idx]
        R = np.asarray([self.all_R[idx]])
        # DONE = np.asarray([self.all_DONE[idx]])

        self.S = torch.from_numpy(S.astype(np.float32)).to(device)
        self.A = torch.from_numpy(A.astype(np.float32)).to(device)
        self.NS = torch.from_numpy(NS.astype(np.float32)).to(device)
        self.R = torch.from_numpy(R.astype(np.float32)).to(device)
        # self.DONE = torch.from_numpy(DONE.astype(np.float32)).to(device)

        sample = {'S': self.S, 'A': self.A, "NS": self.NS, "R": self.R}

        return sample

    def __len__(self):
        return len(self.all_S)

    def add_data(self, SAS_data_):
        S_data, A_data, NS_data, all_DONE, all_R = SAS_data_
        self.all_S = np.vstack((self.all_S, S_data))
        self.all_A = np.vstack((self.all_A, A_data))
        self.all_NS = np.vstack((self.all_NS, NS_data))
        self.all_DONE = np.hstack((self.all_DONE, all_DONE))
        self.all_R = np.hstack((self.all_R, all_R))


def train_dyna_sm(sm_model, train_dataset, test_dataset):
    min_loss = + np.inf
    abort_learning = 0
    decay_lr = 0
    num_epoch = 100
    batchsize = 16
    all_train_L, all_valid_L = [], []

    train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    valid_dataloader = DataLoader(test_dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(sm_model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=5, verbose=False)

    for epoch in range(num_epoch):
        t0 = time.time()
        train_L, valid_L = [], []

        # Training Procedure
        sm_model.train()
        for batch in train_dataloader:
            S, A, NS = batch["S"], batch["A"], batch["NS"]
            pred_NS = sm_model.forward(S, A)
            loss = sm_model.loss(pred_NS, NS)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_L.append(loss.item())

        avg_train_L = np.mean(train_L)
        all_train_L.append(avg_train_L)

        sm_model.eval()
        with torch.no_grad():
            for batch in valid_dataloader:
                S, A, NS = batch["S"], batch["A"], batch["NS"]
                pred_NS = sm_model.forward(S, A)
                loss = sm_model.loss(pred_NS, NS)
                valid_L.append(loss.item())

        avg_valid_L = np.mean(valid_L)
        all_valid_L.append(avg_valid_L)

        if avg_valid_L < min_loss:
            # print('Training_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_train_L))
            # print('Testing_Loss At Epoch ' + str(epoch) + ':\t' + str(avg_valid_L))
            min_loss = avg_valid_L
            # PATH = log_path + '/best_model.pt'
            # torch.save(sm_model.state_dict(), PATH)
            abort_learning = 0
        else:
            abort_learning += 1
            decay_lr += 1
        # scheduler.step(avg_valid_L)
        # np.savetxt(log_path + "training_L.csv", np.asarray(all_train_L))
        # np.savetxt(log_path + "testing_L.csv", np.asarray(all_valid_L))

        if abort_learning > 5:
            break
        t1 = time.time()
        # print(epoch, "time used: ", (t1 - t0) / (epoch + 1), "lr:", lr)

    # print("valid_loss:", min_loss)
    return min_loss, sm_model


def  collect_robot_sasf(robot_name, URDF_PTH, data_save_PTH, num_epochs = 50,
    NUM_EACH_CYCLE = 6, render_flag=False):


    robot_info_pth = URDF_PTH + "%s/" % robot_name
    initial_joints_angle = np.loadtxt(robot_info_pth + "%s.txt" % robot_name)
    initial_joints_angle = initial_joints_angle[0] if len(
        initial_joints_angle.shape) == 2 else initial_joints_angle

    env = meta_sm_Env(initial_joints_angle,
                      urdf_path=robot_info_pth + "%s.urdf" % robot_name)
    env.sleep_time = 0
    sm_model = FastNN(18 + 10, 18)
    # pretrain sm:
    # if load_pretrain_sm:
    #     ...
    sm_model.to(device)
    choose_a = np.random.uniform(-1, 1, size=10)
    train_SAS, test_SAS, choose_a, sele_list = collect_dyna_sm_data(env,
                                                                    sm_model=sm_model,
                                                                    step_num=NUM_EACH_CYCLE,
                                                                    use_policy=0,
                                                                    choose_a=choose_a)
    train_data = SAS_data(SAS_data=train_SAS)
    test_data = SAS_data(SAS_data=test_SAS)

    log_valid_loss = []

    for epoch_i in range(num_epochs):
        sm_train_valid_loss, sm_model = train_dyna_sm(sm_model, train_data, test_data)

        # Collect one epoch data.
        train_SAS, test_SAS, choose_a, sub_sele_list = collect_dyna_sm_data(env,
                                                                            sm_model=sm_model,
                                                                            train_data=train_data,
                                                                            test_data=test_data,
                                                                            step_num=NUM_EACH_CYCLE,
                                                                            use_policy=1,
                                                                            choose_a=choose_a)
        # Add new data to Data Class
        train_data.add_data(train_SAS)
        test_data.add_data(test_SAS)

        log_valid_loss.append(sm_train_valid_loss)

        # Save dataset.

    PATH = data_save_PTH + "/model_%d" % num_epochs
    torch.save(sm_model.state_dict(), PATH)
    np.savetxt(data_save_PTH + "sm_valid_loss.csv", np.asarray(log_valid_loss))

    S0, A0, NS0, D0 = train_data.all_S, train_data.all_A, train_data.all_NS, train_data.all_DONE
    S1, A1, NS1, D1 = test_data.all_S, test_data.all_A, test_data.all_NS, test_data.all_DONE
    D0 = np.expand_dims(D0, axis=1)
    D1 = np.expand_dims(D1, axis=1)

    data_save0 = np.hstack((S0, A0, NS0, D0))
    data_save1 = np.hstack((S1, A1, NS1, D1))
    data_save = np.vstack((data_save0, data_save1))

    np.save(data_save_PTH + f'sasf_random_total{num_epochs * NUM_EACH_CYCLE}.npy', data_save)



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    print("Device:", device)
    robot_name = 'V000'


    p.connect(p.GUI)
    # p.connect(p.DIRECT)
    np.random.seed(2022)
    # Load parameters for gait generator
    para = np.loadtxt("CAD2URDF/para.csv")
    gait_gaussian = 0.2
    # Gait parameters and sans data path:
    log_pth = "data/babbling/"
    os.makedirs(log_pth, exist_ok=True)


    # Initialize an environment
    env = V000_sm_Env(para, urdf_path="CAD2URDF/%s/urdf/%s.urdf" % (robot_name, robot_name))
    env.sleep_time = 1/960

    obs = env.reset()
    # Every epoisde the robot runs 6 steps.
    num_step = 6
    rcmd_a = para
    step_times = 0
    r_record = -np.inf
    SANS_data = []
    while 1:
        all_rewards = []
        action_list = []
        obs = env.resetBase()
        action = np.random.normal(rcmd_a, scale=gait_gaussian)
        action_list.append(action)

        for i in range(num_step):
            step_times += 1
            next_obs, r, done, _ = env.step(action)
            sans = np.hstack((obs, action, next_obs))
            SANS_data.append(sans)
            obs = next_obs
            if done:
                break
        pos, ori = env.robot_location()
        all_rewards.append(r)

    np.savetxt(log_pth+"sans_%d.csv",SANS_data)
    p.disconnect()

