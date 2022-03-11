import numpy as np
import matplotlib.pyplot as plt

x1 = [0.5, 0.1, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1, 0.35, 0.25]
x2 = [0.9, 0.8, 0.75, 1.0]
# for class 1
mean_1 = np.mean(x1)
std_2 = np.std(x2)
pi_1 = len(x1) / (len(x1) + len(x2))    # class probability: y=1
# for class 2
std_1 = np.std(x1)
mean_2 = np.mean(x2)
pi_2 = 1 - pi_1     # class probability: y=2

x_grid = np.linspace(0, 1, 101, endpoint=True)
# PDF for Gaussian distribution
Gaussian = lambda xx, mean, std: 1/((2 * np.pi)**1/2 * std)\
     * np.exp(-(xx - mean)**2 / (2 * std**2))
# score p(x,y=1)
prob_y1 = Gaussian(x_grid, mean_1, std_1) * pi_1
# score p(x,y=2)
prob_y2 = Gaussian(x_grid, mean_2, std_2) * pi_2
plt.plot(x_grid, prob_y1, label='Class y=1')
plt.plot(x_grid, prob_y2, label='Class y=2')
plt.xlabel('x')
plt.ylabel('scores')

# p(y=1 | x=0.6) ∝ p(x=0.6|y=1) * p(y=1) 
prob_y1 = Gaussian(0.6, mean_1, std_1) * pi_1 
prob_y2 = Gaussian(0.6, mean_2, std_2) * pi_2 
Prob_y1 = prob_y1 / (prob_y1 + prob_y2)
print('The probability that point x=0.6 belings to class 1:',Prob_y1)
plt.legend()
plt.show()