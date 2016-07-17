# a3c-rl
Use a slightly different algorithm to train a pong agent.
reach a mean reward about 12 in 2e7 steps with learning rate 1e-4.

Combine the cpu and gpu. Because I use a Monte-Carlo method, ie. n-steps, for such a episode game.
I put the backward part on the gpu which can speed up the training a lot.
