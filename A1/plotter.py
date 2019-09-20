import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
    
def hypothesis(theta, x):
    return theta[0] + theta[1] * x


def derivative0(theta, x, y):
    m = len(x)
    sum = 0.0

    for i in range(0, m):
        sum += hypothesis(theta, x[i]) - y[i]

    return (sum / m)


def derivative1(theta, x, y):
    m = len(x)
    sum = 0.0

    for i in range(0, m):
        sum += (hypothesis(theta, x[i]) - y[i]) * x[i]

    return (sum / m)


def cost(theta, x, y):
    m = len(x)
    sum = 0.0

    for i in range(0, m):
        sum += (hypothesis(theta, x[i]) - y[i])**2

    return (sum / (2.0 * m))
    
def run(x, y):
    iteration = 0
    delta = 1
    limit = 1e-6
    alpha = 0.01
    theta = [0, 0]
    new_theta = [0, 0]
    while delta > limit:
        iteration += 1
        old = cost(theta, x, y)
        
        df0 = alpha * derivative0(theta, x, y)
        df1 = alpha * derivative1(theta, x, y)
        
        new_theta[0] = theta[0] - df0
        new_theta[1] = theta[1] - df1
        
        new = cost(new_theta, x, y)
        delta = old - new
        
        while old < new:
            df0 *= 0.1
            df1 *= 0.1
            new_theta[0] = theta[0] - df0
            new_theta[1] = theta[1] - df1
            new = cost(new_theta, x, y)
            delta = old - new
            # print(f'Delta {delta}')
            
        theta[0] = new_theta[0]
        theta[1] = new_theta[1]
        # print(f'Cost {cost(theta, x, y)}')
    
    print(f'Total Iterations: {iteration}')
    return theta

x = []
y = []

df = pd.read_csv("housing.csv")
cols = df.columns
number_of_columns = len(cols)

for ci in range(3, number_of_columns-9):
    print('====================================================')
    print(f'For {cols[ci]}:')
    x = df[cols[ci]]
    y = df[cols[number_of_columns-1]]
    thetas = [
        [24.672369100931952, -1.25231796073369],
        [20.875974042653038, 0.14302536896944076],
        [29.71855951512921, -0.6461197192776257],
        [22.105338229786106, 6.190613095776837],
        [40.86769901803369, -33.062816806932396],
        [-33.849138601123336, 8.972967818563113],
        [22.635465770235303, -0.018265605516869607],
        [18.346784470531375, 1.1004476235382143],
        [26.360189150441347, -0.4018363854964163],
        [0.13029935906648632, 0.04333494495399045],
        [20.43977112678136, 0.08745764946354015],
        [0.000291045563985186, 0.06138322393011101],
        [32.16296577805981, -0.8190799560000949]]
    xwidth = int(max(x)-min(x))
    xmargin = (xwidth + 1) * 0.1
    xmin = min(x) - xmargin
    xmax = max(x) + xmargin
    yheight = int(max(y)-min(y))
    ymargin = yheight * 0.05
    ymin = min(y) - ymargin
    ymax = max(y) + ymargin
    plt.axis([xmin, xmax, ymin, ymax])
    xline = [i for i in range(-2, int(3*xmax))]
    yline = [xi*thetas[ci][1]+thetas[ci][0] for xi in xline]
    plt.plot(xline, yline, 'r-')
    plt.scatter(x, y, alpha=0.5)
    plt.title(cols[ci])
    plt.xlabel(cols[ci])
    plt.ylabel(cols[number_of_columns-1])
    plt.show()
    
    




