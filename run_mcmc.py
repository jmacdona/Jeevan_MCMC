import os
import math, sys
import numpy as np
import random

# import normal distribution:
from scipy.stats import multivariate_normal


mean = np.array([0., 0.])

cov = np.matrix([[1., 0.5],
                 [0.5, 1.]])
likelihood_dist = multivariate_normal(mean=mean, cov=cov)

prop_mean = np.array([0., 0.])
prop_cov = 0.1 * np.matrix([[1., 0.5],
                 [0.5, 1.]])
proposal_dist = multivariate_normal(mean=prop_mean, cov=prop_cov)


# x = np.array([0, 0])

#print(multivariate_normal.pdf(x, mean=mean, cov=1))


#for x in np.arange(-3, 3, 0.1):
#    for y in np.arange(-3, 3, 0.1):
#        pos = np.array([x, y])
#        # pdf = multivariate_normal.pdf(pos, mean=mean, cov=cov)
#        pdf = likelihood_dist.pdf(pos)
#        print(str(x) + "\t" + str(y) + "\t" + str(pdf))

# Markov chain Monte Carlo

# start position:
pos = np.array([5, 5])

steps = 100
step = 0
curr_likelihood = likelihood_dist.pdf(pos)

while step < steps:

    prop_pos = pos + proposal_dist.rvs()
    prop_likelihood = likelihood_dist.pdf(prop_pos)

    # probability of accepting random move (aka proposal)
    prob_acc = min(1, prop_likelihood/curr_likelihood)

    # pick random number from uniform distribution from 0 to 1
    random_number = random.random()

    if (random_number < prob_acc):
        # have accepted proposal
        pos = prop_pos
        curr_likelihood = prop_likelihood

    print("STEP:\t" + str(step) + "\t" + str(pos[0]) + "\t" + str(pos[1]) + "\t" + str(curr_likelihood))

    step += 1



