# BabaIsRLAgent

- Create a virtual environment and run `pip install -r requirements.txt`.
- You can run things by running `run.py`.
    - See the code in `if __name__ == "__main__"` to see how things are run. Essentially, just specify a model to train, and evaluate it. The `run_manually` function is only used for debugging purposes (to make sure the enviroment and rewards were working correctly).
    - In the environment parameters, if `train=True` then randomization will be used. If `object_to_shuffle` is not set, then "complete" randomization is used; otherwise, only the specified object will be randomized.
    - To train with different reward weights, set reward_config=<CONFIG> on the BabaWorldEnv object. The set of configs can be found in envs/reward_schemes.py

    - Command-line arguments:
        - `--alg`: one of 'dqn', 'a2c', or 'ppo'
        - `--rewards`: a space-separated list of "winlose", "nochange", "movetext", "distance", or "all". See envs/BabaWorldEnv.py for implementation of each reward scheme


Sample command:
`python -m run 1 --alg dqn --rewards winlose movetext`