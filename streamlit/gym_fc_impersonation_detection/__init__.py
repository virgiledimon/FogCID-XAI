import gymnasium as gym
from gymnasium.envs.registration import register

try:
    register(
        id='FCImpersonationDetection-v0',
        entry_point='gym_fc_impersonation_detection.envs:FCSpoofingEnv',
    )
except gym.error.Error as e:
    # If the environment is already registered, ignore the error
    if not str(e).startswith('Cannot re-register id: '):
        raise e