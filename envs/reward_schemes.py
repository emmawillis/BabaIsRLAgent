
# Default
DEFAULT_REWARDS = {
    "win_bonus": 100,
    "lose_penalty": -1000,
    "nochange_penalty": -100,
    "text_move": 10,
    "text_rule": 25,
    "you_win_rule": 50,
    "distance_weight": -1.0
}

# Encourages exploration through text interaction, but still strongly rewards winning.
# Goal: Steady exploration with a mild penalty for doing nothing. Text movement is not overly rewarded.
EXPLORATION_FOCUS = {
    "win_bonus": 100,
    "lose_penalty": -1000,
    "nochange_penalty": -10,
    "text_move": 5,
    "text_rule": 20,
    "you_win_rule": 30,
    "distance_weight": -0.5,
}


# Loosen the punishment for not winning and encourage playing with rule blocks.
# Goal: Lower penalties and more incentive to change the world without needing to win fast.
TEXT_FOCUS = {
    "win_bonus": 100,
    "lose_penalty": -500,
    "nochange_penalty": -5,
    "text_move": 20,
    "text_rule": 50,
    "you_win_rule": 75,
    "distance_weight": -0.2,
}


# Bigger reward for changing rules, especially 'YOU' and 'WIN'.
# Goal: Push the agent to learn that manipulating rules is essential to winning.
RULE_FOCUS = {
    "win_bonus": 100,
    "lose_penalty": -100,
    "nochange_penalty": -25,
    "text_move": 10,
    "text_rule": 100,
    "you_win_rule": 150,
    "distance_weight": -0.1,
}


# Goal: Aggressively minimize distance to win objects to avoid random wandering
DISTANCE_FOCUS = {
    "win_bonus": 100,
    "lose_penalty": -300,
    "nochange_penalty": -50,
    "text_move": 2,
    "text_rule": 10,
    "you_win_rule": 20,
    "distance_weight": -1.0,
}
