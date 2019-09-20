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
hundreds = 53998
#5399800 iterations
theta = [0.8153050407908885, -0.24023534058602747, 0.0471576106975837, 0.009900074436971064, 1.1710530133311983, 0.20420348704975436, 5.732643253834673, -0.006477385094260471, -0.9134626478799869, 0.19470673024282908, -0.010721052501982416, -0.44242716785141745, 0.015811638826258117, -0.46158425458534974]
new_theta = [0.0] * (n-1)
df = [0.0] * (n-1)
alpha = 0.1
limit = 1e-15
delta = 0.001
iterations = 0
outputfile = "return2.txt"

while delta >= limit:
    iterations += 1
    old = cost(theta, data)
    print(old)
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
    
    # saveThetaToFile(outputfile, (hundreds*100+iterations), new, theta)
        
    if iterations == 100:
        iterations = 0
        hundreds += 1
        saveThetaToFile(outputfile, hundreds*100, new, theta)
        
print(theta)
print(cost(theta, data))

print(hundreds*100+iterations)