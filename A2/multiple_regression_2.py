import numpy as np

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
    
data = load("test.csv")
m, n = data.shape
theta = [0.0] * (n-1)
new_theta = [0.0] * (n-1)
df = [0.0] * (n-1)
alpha = 0.000000000000000000001
limit = 0.000000000000000000001
delta = 0.0000001
iterations = 0

while delta >= limit:
    iterations += 1
    old = cost(theta, data)
    
    for x in range(0, n - 1):
        new_theta[x] = alpha * derivative(theta, data, x)
    for x in range(0, n - 1):
        theta[x] = new_theta[x]
        
    new = cost(theta, data)
    print(old)
    print(new)
    delta = old - new
    
print(theta)
print(iterations)
#3062 iterations