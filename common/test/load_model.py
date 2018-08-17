from agent.DQNModel import DQNModel as Model

if __name__ == "__main__":
    env_name    = "Breakout-v0"
    input_shape = (4, 84, 84, 1)
    nb_actions  = 4
    model = Model(input_shape, nb_actions, env_name)
