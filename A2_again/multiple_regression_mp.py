import numpy as np
import multiprocessing as mp

def load(filename):
    data = np.loadtxt(open(filename), dtype="float")
    m, n = data.shape
    # create column with ones
    first_col = np.ones((m, 1))
    # stack the colums before the data
    data = np.hstack((first_col, data))
    return data
    
def hypothesis(theta, x):
	x = np.mat(x)
	theta = np.mat(theta)
	return theta * x.T
	
def cost(theta, data):
    sum = 0.0
    m, n = data.shape
    last = n - 1
    for x in range(0, m):
    	# theta and data except the last column
        sum += (hypothesis(theta, data[x, :-1]) - data[x, last]) ** 2
    # this returns a mat but we need a single value so we take the only element of this mat
    return (sum / (2.0 * m))[0,0]

# derivative of theta WRT to variable data[x, j]
def derivative(theta, data, j):
    sum = 0.0
    m, n = data.shape
    last = n - 1
    for x in range(0, m):
        sum += (hypothesis(theta, data[x, :-1]) - data[x, last]) * data[x, j]
    return (sum / (2.0 * m))[0,0]
    
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
    
data = load("housing.csv")
m, n = data.shape
hundreds = 11
#1100th iteration
theta = [0.008138359683488363, -0.01594908626041901, 0.13068018278744128, -0.04586650931878984, 0.00538766248383133, 0.0029208921806709426, 0.09657915248220844, 0.07096640423476626, 0.016317460633819374, 0.003051199569667387, 0.0006272995166802606, 0.06571275638383509, 0.048453292882681145, -0.2620020970365925]
new_theta = [0.0] * (n-1)
df = [0.0] * (n-1)
alpha = 0.1
limit = 1e-11
delta = 0.001
iterations = 0
outputfile = "result.txt"

if __name__ == '__main__':
    pool = mp.Pool(6)
    while delta >= limit:
        iterations += 1
        old = cost(theta, data)
        # new_theta = pool.map(get_new_theta, range(0, n-1))
        for x in range(0, n - 1):
            df[x] = alpha * derivative(theta, data, x)
            
        for x in range(0, n - 1):
            new_theta[x] = theta[x] - df[x]
            
        new = cost(new_theta, data)
        delta = old - new
        
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
print(hundreds*100+iterations)