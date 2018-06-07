def train(env, agent, max_timesteps):
    x = env.reset()
    done = False
    timestep = 0

    while timestep < max_timesteps:
        if done:
            x = env.reset()
        a = agent.act(x)
        x_prime, reward, done, _ = env.step(a)
        agent.update(x, a, x_prime, reward, done)
        x = x_prime
        timestep += 1
