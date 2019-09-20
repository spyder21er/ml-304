import numpy as np
import math

def load(filename):
    data = np.loadtxt(open(filename), dtype="float")
    m, n = data.shape
    # create column with ones
    first_col = np.ones((m, 1))
    # stack the columns before the data
    data = np.hstack((first_col, data))
    return data
    
def hypothesis(theta, x):
    x = np.mat(x)
    theta = np.mat(theta)
    z = theta * x.T
    z = z * -1.0
    
    return 1.0 / (1.0 + math.exp(z))

def cost(theta, data):
    sum = 0.0
    m, n = data.shape
    last = n - 1
    for x in range(0, m):
        yone = data[x, last] * np.log(hypothesis(theta, data[x, :-1]))
        yzero = (1.0 - data[x, last]) * np.log(1.0 - hypothesis(theta, data[x, :-1]))
        sum += yone + yzero

    return -1.0 * (sum / float(m))

# derivative of theta WRT to variable data[x, j]
def derivative(theta, data, j):
    sum = 0.0
    m, n = data.shape
    last = n - 1
    for x in range(0, m):
        sum += (hypothesis(theta, data[x, :-1]) - data[x, last]) * data[x, j]
    return (sum / m)
    
def saveThetaToFile(filename, iteration_number, cost, theta):
    f = open(filename, "a+")
    f.write("================================================================\n")
    f.write("Iteration %d: \n" % iteration_number)
    f.write("Cost: %s\n" % cost)
    f.write("Theta: ")
    for x in theta:
        f.write("%s, " % x)
    f.write("\n")
    f.close
    
data = load("bc_minMax.txt")
m, n = data.shape
hundreds = 1842
#184200 iteration
theta = [-7.240947686147089, 4.822916523691607, -0.01689672608134557, 2.8641978165317346, 2.966914659270142, 0.8680668443953029, 3.4481291086001473, 4.015915879203883, 1.9168261954282448, 4.66351643508138]
new_theta = [0.0] * (n-1)
df = [0.0] * (n-1)
alpha = 0.1
limit = 1e-25
delta = 0.001
iterations = 0
outputfile = "result_minMax.txt"

while delta >= limit:
    iterations += 1
    print("=============================================")
    
    old = cost(theta, data)
    for x in range(0, n - 1):
        df[x] = alpha * derivative(theta, data, x)
    
    for x in range(0, n - 1):
        new_theta[x] = theta[x] - df[x]
        
    new = cost(new_theta, data)
    delta = old - new
    print(f"Cost: {old}\nNew: {new}\nDelta: {delta}\n")
        
    while old < new:
        for x in range(0, n - 1):
            df[x] *= 0.1
        for x in range(0, n - 1):
            new_theta[x] = theta[x] - df[x]
        new = cost(new_theta, data)
        delta = old - new
    
    for x in range(0, n - 1):
        theta[x] = new_theta[x]
        
    if iterations == 100:
        iterations = 0
        hundreds += 1
        saveThetaToFile(outputfile, hundreds*100, new, theta)
        

print(theta)
print(cost(theta, data))

print(hundreds*100+iterations)
