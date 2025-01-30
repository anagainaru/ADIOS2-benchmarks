import numpy as np
import time
import argparse
import adios2.bindings as adios2

supportedExpr = ["add", "multiply", "magnitude", "curl"]
numSteps = 100
derivedName = "derived"

# generate random dataset
# nElem dimensions of the array [n1, n2, n3] for 3D arrays
# nVar number of variables created
# seed value for generating the random numbers
# returns: [np.array(var 1, n dimensions), ...]
def generante_dataset(nElem, nVar, seed):
    high = 100
    low = -100
    data = []
    np.random.seed(seed)
    for _ in range(nVar):
        varData = np.random.rand(*nElem) * (high - low) + low
        data.append(varData)
    return data

# read dataset from a bp file
# fileName path to the bp file being read
# listVarNames a list of adios variables
# returns: the same as generate_datase
def read_dataset(fileName, listVarNames):
    return []

# applies a given expression using numpy logic
# returns: average time it took to apply the expression once
def apply_expression(expression, listArrays):
    print("Apply %s on the dataset" %(expression))
    ts = []
    for step in range(numSteps):
        start_ts = time.time()
        if expression == "add":
            res = np.zeros(listArrays[0].shape)
            # add all numpy arrays in the list
            for var in listArrays:
                res = res + var
        end_ts = time.time()
        if step > 10:
            ts.append(end_ts - start_ts)
    return np.mean(ts)

# write a bp file asking for the expression to be
# computed on the fly by ADIOS
# type_write: writeData, statsOnly or exprOnly
# returns: average time it took to write one step
def write_with_derived(fileName, expression, type_write, listArrays):
    adios = adios2.ADIOS()
    ioWriter = adios.DeclareIO("derivedWriter")
    # define adios variables
    adiosVars = []
    ts = []
    cnt = 0
    # create adios variables for each array part of the derived expression
    for var in listArrays:
        cnt += 1
        adiosVars.append(ioWriter.DefineVariable(
            "var"+str(cnt), var, var.shape, [0]*len(var.shape), var.shape))
    # add the derived variable
    if expression == "add":
        ioWriter.DefineDerivedVariable(derivedName, "a=var1\nb=var2\na+b",
                                       type_write)
    elif expression == "multiply":
        ioWriter.DefineDerivedVariable("derived", "a=var1\nb=var2\na*b",
                                       type_write)
    wStream = ioWriter.Open(fileName, adios2.Mode.Write)
    for step in range(numSteps):
        start_ts = time.time()
        wStream.BeginStep()
        cnt = 0
        for var in listArrays:
            wStream.Put(adiosVars[cnt], var)
            cnt += 1
        wStream.EndStep()
        end_ts = time.time()
        if step > 10: # skip warm-up steps
            ts.append(end_ts - start_ts)
    wStream.Close()

    return np.mean(ts)

# read variables written by the write_with_derived function
# from a bp file for derived variables
# returns: average time it took to read one step
def read_with_derived(fileName, shape):
    adios = adios2.ADIOS()
    ioReader = adios.DeclareIO("derivedReader")
    ts = []
    rStream = ioReader.Open(fileName, adios2.Mode.Read)
    for step in range(numSteps):
        buffer = np.zeros(shape=shape)
        start_ts = time.time()
        rStream.BeginStep()
        adiosVar = ioReader.InquireVariable(derivedName)
        rStream.Get(adiosVar, buffer)
        rStream.EndStep()
        end_ts = time.time()
        if step > 10: # skip warm-up steps
            ts.append(end_ts - start_ts)
    rStream.Close()

    return np.mean(ts)

# filter the datasets so it contain the number of variables expected by the 
# expression (e.g. magnitude will need 3 etc)
def filter_dataset(dataset, expression):
    if len(dataset) == 0:
        print("ERR the expression used (%s) needs to be applied on at least"
              "one variable" %(expression))
        exit()
    if expression == "add" or expression == "multiply":
        if len(dataset) < 2:
            print("The expression used (%s) needs to have at least two variables" %(
                expression))
            exit()
        return dataset[:2]
    if expression == "magnitude" or expression == "curl":
        if len(dataset) < 3:
            print("The expression used (%s) needs to have at least three variables" %(
                expression))
            exit()
        return dataset[:3]
    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('testName', type=str,
                        help='name for the tests being performed')
    parser.add_argument('-f', '--file', type=str, default='n/a',
                        help='input file containing the arrays used for the test')
    parser.add_argument('-v', '--variables', type=str, default='n/a', nargs='+',
                        help='variables inside the input file used for testing')
    parser.add_argument('-s', '--size', type=int, dest='size', nargs='+', default=0,
                        help='the size of each dimension for the generated arrays')
    parser.add_argument('-n', '--numvars', type=int, default=1,
                        help='number of variables that are generated (default 1)')
    parser.add_argument('-r', '--randseed', type=int, default=0,
                        help='seed for generating the random numbers (default 0)')
    parser.add_argument('-e', '--expr', choices=supportedExpr, dest='expression',
                        default="add", help='the derived expression under investigation')
    args = parser.parse_args()

    if args.size != 0 and args.file != 'n/a':
        print("ERR The script cannot use generated arrays (given by -s) at the same"
              " time with an input file (given by -f)")
        exit()

    if args.file != 'n/a' and args.variables == 'n/a':
        print("ERR variables not provided for the input file", args.file)
        exit()

    dataset = []
    # if we are generating the arrays used for testing
    if args.size != 0:
        seed = 100
        dataset = generante_dataset(args.size, args.numvars, args.randseed)
        print("Generated a dataset of %d variables of shape %s" %(
            len(dataset), dataset[0].shape))

    dataset = filter_dataset(dataset, args.expression)
    ts_apply = apply_expression(args.expression, dataset)
    print("Time to apply expression:", ts_apply)

    ts_write = write_with_derived("out_%s.bp" %(args.testName), args.expression,
                       adios2.DerivedVarType.StatsOnly, dataset)
    print("Time to write expression:", ts_write)

    ts_read = read_with_derived("out_%s.bp" %(args.testName), dataset[0].shape)
    print("Time to read expression:", ts_read)
