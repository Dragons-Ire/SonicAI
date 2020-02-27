import numpy as np
import gym

from retro_contest.local import make
from retro import make as make_retro

from baselines.common.atari_wrappers import FrameStack

import cv2

#cv2.oc1.setUseOpenCL(False)

class PreprocessFrame(gym.ObservationWrapper):
    
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)
    
    def observation(self, frame):
        # Set frame to gray
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # Resize the frame to 96x96x1
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame


class ActionsDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """
    def __init__(self, env):
        super(ActionsDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []

        """
        What we do in this loop:
        For each action in actions
            - Create an array of 12 False (12 = nb of buttons)
            For each button in action: (for instance ['LEFT']) we need to make that left button index = True
                - Then the button index = LEFT = True
            In fact at the end we will have an array where each array is an action and each elements True of this array
            are the buttons clicked.
        """
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()

class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """
    def reward(self, reward):

        return reward * 0.01
    
class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """
    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

def make_env(env_idx):
    """
    Create an environment with some standard wrappers.
    """

    dicts = [
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'AngelIslandZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HydrocityZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HydrocityZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MarbleGardenZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MarbleGardenZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'CarnivalNightZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'IceCapZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'IceCapZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LaunchBaseZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LaunchBaseZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'MushroomHillZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'FlyingBatteryZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'FlyingBatteryZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'SandopolisZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'SandopolisZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'LavaReefZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'HiddenPalaceZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'SkySanctuaryZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'DeathEggZone.Act1'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'DeathEggZone.Act2'},
        {'game': 'SonicAndKnuckles3-Genesis', 'state': 'TheDoomsdayZone.Act1'}
        

    ]
    # Make the environment
    print(dicts[env_idx]['game'], dicts[env_idx]['state'], flush=True)
    #record_path = "./records/" + dicts[env_idx]['state']
    env = make(game=dicts[env_idx]['game'], state=dicts[env_idx]['state'], bk2dir="./records")#record='/tmp')

    # Build the actions array, 
    env = ActionsDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)



    return env



def make_train_0():
    return make_env(0)

def make_train_1():
    return make_env(1)

def make_train_2():
    return make_env(2)

def make_train_3():
    return make_env(3)

def make_train_4():
    return make_env(4)

def make_train_5():
    return make_env(5)

def make_train_6():
    return make_env(6)

def make_train_7():
    return make_env(7)

def make_train_8():
    return make_env(8)

def make_train_9():
    return make_env(9)

def make_train_10():
    return make_env(10)

def make_train_11():
    return make_env(11)

def make_train_12():
    return make_env(12)

def make_train_13():
    return make_env(13)

def make_train_14():
    return make_env(14)

def make_train_15():
    return make_env(15)

def make_train_16():
    return make_env(16)

def make_train_17():
    return make_env(17)

def make_train_18():
    return make_env(18)

def make_train_19():
    return make_env(19)

def make_train_20():
    return make_env(20)

def make_train_21():
    return make_env(21)

def make_train_22():
    return make_env(22)

def make_train_23():
    return make_env(23)

def make_train_24():
    return make_env(24)

def make_test_level_Green():
    return make_test()


def make_test():
    """
    Create an environment with some standard wrappers.
    """


    # Make the environment
    env = make_retro(game='SonicTheHedgehog-Genesis', state='GreenHillZone.Act2', record="./records")

    # Build the actions array
    env = ActionsDiscretizer(env)

    # Scale the rewards
    env = RewardScaler(env)

    # PreprocessFrame
    env = PreprocessFrame(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance
    # head-on in the level.
    env = AllowBacktracking(env)

    return env