import tempfile

import LogisticRegression


def classify(modelfile, dsfile):
    modelFile = open(modelfile)
    Q = eval(modelFile.readline())
    modelFile.close()

    # Create temporary file
    t = tempfile.NamedTemporaryFile(mode="r+")

    # Open input file in read-only mode
    i = open(dsfile, 'r')

    # Copy input file to temporary file
    for line in i:
        value = [int(j) for j in line.split(" ")]
        t.write(line.rstrip() + " " + str(LogisticRegression.classify(Q, value)) + " \n")
    t.seek(0)  # Rewind temporary file

    o = open(dsfile, "w")  # Reopen input file writable

    # Overwriting original file with temp file contents
    for line in t:
        o.write(line)

    t.close()  # Close temporary file


if __name__ == "__main__":
    LogisticRegression.ds = LogisticRegression.readDS("dataSet.txt")
    Q = [1, 1, 1]
    t = LogisticRegression.gradient_descent(LogisticRegression.function, LogisticRegression.derivative, 0.000001, 0.001,
                                            Q)
    LogisticRegression.save_model("model.txt", t)
    classify("model.txt", "classfiy.txt")

    # classify("model.txt", "dataSet.txt", "classfiy.txt")
    # global ds
    # # ds = [[1, 1, 0], [2, 3, 1], [3, 2, 1]]  # = [x1,x2,y]
    # ds = LogisticRegression.readDS("C:\\Users\\yosef\\IdeaProjects\\LogisticRegression\\src\\dataSet.txt", " ")
    # Q = [1, 1, 1]
    #
    # Q = gradient_descent(function, derivative, 0.000001, 0.001, Q)
    #
    # print(Q)
    # print("+++++++++++++++++")
    # print(classify(Q, [1, 1]))
    # print(classify(Q, [2, 3]))
    # print(classify(Q, [3, 2]))
    # # print(classify(Q,[1.52,1.52]))
    # # -4.9 + 1.6 x1 + 1.6 x2 = 0
