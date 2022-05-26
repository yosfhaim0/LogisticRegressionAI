import math

'''
Computes the dot product of the vectors x and y (lists of numbers).
Assumes that x and y have the same length.
'''


def dot_prod(Q, x):
    return sum([Q[i] * x[i] for i in range(len(x))])


# Q= (Q0,Q1,Q2)
# X = (1,X1,X2)
#
# Q.X
#
# A.B = 1*4 + 2*5 + 3*6
def sigmoid(Q, x):
    return 1 / (1 + math.exp(-dot_prod(Q, x)))


def gradient_descent(function, derivative, epsilon, alpha, Q):
    f = function(Q)
    prevf = f + 1 + epsilon
    while abs(prevf - f) > epsilon:
        # print([derivative(i, Q) for i in range(len(Q))])
        # print(function(Q))
        Q = [Q[i] - alpha * derivative(i, Q) for i in range(len(Q))]
        prevf = f
        f = function(Q)
        # print(Q)
    return Q


def function(t):
    global ds
    return -sum([ds[i][-1] * math.log(sigmoid(t, [1] + ds[i][:-1])) + \
                 (1 - ds[i][-1]) * math.log(1 - sigmoid(t, [1] + ds[i][:-1])) \
                 for i in range(len(ds))]) / len(ds)


def derivative(j, Q):  # derivative in Q[j] (e.g. Q0, Q1, Q2)
    global ds
    return sum([((sigmoid(Q, [1] + ds[i][:-1])) - ds[i][-1]) * ([1] + ds[i])[j] \
                for i in range(len(ds))]) / len(ds)


def classify(Q, x):
    return round(sigmoid(Q, [1] + x))


# symbol = the separate symbol between value in the file by default " "
def readDS(DsFileName, symbol=" "):
    DSfile = open(DsFileName)
    # reading each line of the file and printing to the console
    _ds=[[eval(i) for i in line.split(symbol)] for line in DSfile]
    DSfile.close()
    return _ds


def save_model(filename, t):
    # open the file in write mode
    myfile = open(filename, 'w')
    myfile.write(str(t))
    # close the file
    myfile.close()


ds = []
