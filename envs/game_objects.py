import numpy as np
from enum import Enum

class Actions(Enum):
    right = 0
    up = 1
    left = 2
    down = 3

class Object(Enum):
    BACKGROUND = 0
    BABA = 1
    FLAG = 2
    WALL = 3
    ROCK = 4
    PUSH_TEXT = 5
    STOP_TEXT = 6
    YOU_TEXT = 7
    WIN_TEXT = 8
    IS_TEXT = 9
    BABA_TEXT = 10
    ROCK_TEXT = 11
    FLAG_TEXT = 12
    WALL_TEXT = 13

class ObjectState: 
    def __init__(self, type, is_text=False, paired_object_key=None, rule_text=None, user_rules=None, immutable_rules=None):
        self.type = type
        self.user_rules = user_rules if user_rules is not None else []
        self.immutable_rules = immutable_rules if immutable_rules is not None else []

        # properties to handle the text objects
        self.is_text = is_text
        self.paired_object_key = paired_object_key if paired_object_key is not None else None
        self.rule_text = rule_text if rule_text is not None else None
    
    def rules(self):
        return self.user_rules + self.immutable_rules
 
    def clear_rules(self):
        self.user_rules = []

    def remove_rule(self, rule): # only removes one instance of the rule
        self.user_rules.remove(rule)
    
    def add_rule(self, new_rule):
        self.user_rules.append(new_rule)

    def is_you(self):
        return "you" in self.rules()
    
    def is_win(self):
        return "win" in self.rules()
    
    def is_stop(self):
        return "stop" in self.rules()
    
    def is_push(self):
        return "push" in self.rules()
    
    def is_free(self):
        return not self.rules()

