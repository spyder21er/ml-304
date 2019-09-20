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
    
def check(theta, data):
    m, n = data.shape
    correct = 0
    incorrect = 0
    for i in range(0, m):
        hypo = hypothesis(theta, data[i, :-1])
        prediction = int(hypo + 0.5)
        trueValue = int(data[i, n-1])
        if(prediction == trueValue):
            correct += 1
        else:
            incorrect += 1
            print(f"{i} => {trueValue}|{prediction}|{hypo}")

    print(f"Total correct: {correct}")
    print(f"Total incorrect: {incorrect}")
    
orig_data = load("bc_orig.txt")
minMax_data = load("bc_minMax.txt")
meanNorm_data = load("bc_meanNorm.txt")
unique_data = load("bc_unique_orig.txt")
orig = [-10.103871422831364, 0.5350089841389074, -0.006277326548799986, 0.3227056834892231, 0.3306340556506056, 0.09663345240447464, 0.38302371841166033, 0.447183614620244, 0.21302991587474793, 0.5348280705996659]
minmax = [-7.242385592137217, 4.820999260169416, -0.02724338890432401, 2.8748072316326287, 2.9690257152811137, 0.8684003317695358, 3.4478336281221216, 4.018007322996153, 1.9169424164407511, 4.697971571962601]
meanNorm = [-1.098792990212348, 4.821691515425138, -0.028973723121121956, 2.876457846862837, 2.969358875568506, 0.8685622916360043, 3.447784425408492, 4.018521653266351, 1.917037820738809, 4.696527395146644]
unique = [-9.40317084870481, 0.513442029361829, 0.010307226326166184, 0.28313213137382787, 0.305922605978011, 0.08986813220945644, 0.3576764895743022, 0.4100148645916569, 0.20224509331575477, 0.4906837830486626]

# print("=========================================")
# check(orig, orig_data)
# print("=========================================")
# check(minmax, minMax_data)
# print("=========================================")
# check(meanNorm, meanNorm_data)
print("=========================================")
check(unique, orig_data)
print("=========================================")
check(unique, unique_data)