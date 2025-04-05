# BabaIsRLAgent

- Create a virtual environment and run `pip install -r requirements.txt`.
- You can run things by running `run.py`.
    - See the code in `if __name__ == "__main__"` to see how things are run. Essentially, just specify a model to train, and evaluate it. The `run_manually` function is only used for debugging purposes (to make sure the enviroment and rewards were working correctly).
    - In the environment parameters, if `train=True` then randomization will be used. If `object_to_shuffle` is not set, then "complete" randomization is used; otherwise, only the specified object will be randomized.
