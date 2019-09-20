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
    
data = load("scaled.csv")
m, n = data.shape
hundreds = 801
#80100th iteration
theta = [25.889051919962267, -0.821009291870132, 4.27997876857469, 0.8106611577346373, 2.8275249745089712, -8.113369519343387, 20.159258102590584, 0.06960857654018164, -15.353681594017562, 6.10237913974391, -6.370685740002467, -8.878004315960528, 4.134111039347878, -19.875714644292067]
new_theta = [0.0] * (n-1)
df = [0.0] * (n-1)
alpha = 0.1
limit = 1e-11
delta = 0.001
iterations = 0
outputfile = "result2.txt"

while delta >= limit:
    iterations += 1
    old = cost(theta, data)
    print("===============================================================")
    print(f'Iteration {iterations}: ')
    print(f'Cost: {old} ')
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
    
    print(f'theta: {theta} ')  
    if iterations < 0:
        iterations = 0
        hundreds += 1
        #saveThetaToFile(outputfile, hundreds*100, new, theta)
        