import numpy as np
import time
import argparse
from adios2 import FileReader
import adios2.bindings as adios2

supportedExpr = ["add", "multiply", "magnitude"]
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
def read_dataset(fileName, listVarNames, verbose=True):
    data = []
    with FileReader(fileName) as ibpFile:
        for varName in listVarNames:
            buffer = ibpFile.read(varName)
            data.append(buffer)
        if verbose:
            print("Reading variable %s shape %s" %(varName, buffer.shape))
    return data

# applies a given expression using numpy logic
# returns: average time it took to apply the expression once
def apply_expression(expression, listArrays, verbose=True):
    if verbose:
        print("Apply %s on the dataset" %(expression))
    ts = []
    for step in range(numSteps):
        res = np.zeros(listArrays[0].shape)
        if expression == "multiply":
            res = np.ones(listArrays[0].shape)
        start_ts = time.time()
        if expression == "add":
            for var in listArrays:
                res = res + var
        elif expression == "multiply":
            for var in listArrays:
                res = res * var
        elif expression == "magnitude":
            for var in listArrays:
                res = res + (var * var)
            res = np.sqrt(res)
        end_ts = time.time()
        if step > 10:
            ts.append(end_ts - start_ts)
    return res, np.mean(ts)

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
    elif expression == "magnitude":
        ioWriter.DefineDerivedVariable(
                "derived", "a=var1\nb=var2\nc=var3\nmagnitude(a,b,c)",
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
def read_with_derived(fileName):
    adios = adios2.ADIOS()
    ioReader = adios.DeclareIO("derivedReader")
    ts = []
    buffer = np.zeros(1)
    rStream = ioReader.Open(fileName, adios2.Mode.Read)
    for step in range(numSteps):
        start_ts = time.time()
        rStream.BeginStep()
        adiosVar = ioReader.InquireVariable(derivedName)
        buffer = np.zeros(adiosVar.Shape())
        rStream.Get(adiosVar, buffer)
        rStream.EndStep()
        end_ts = time.time()
        if step > 10: # skip warm-up steps
            ts.append(end_ts - start_ts)
    rStream.Close()
    return buffer, np.mean(ts)

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
    if expression == "magnitude":
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
    parser.add_argument('--verbose', action='store_true',
                        dest='verbose', help='turn on information messages')
    parser.add_argument('--only-read', action='store_true',
                        dest='onlyRead', help='run only the read side test.'
                        ' The mandatory parameter testName will represent the file'
                        ' containing the data and -v the derived variable name.'
                        ' If -v is not used, the file needs to contain a variable'
                        ' called `derived`')
    args = parser.parse_args()

    if args.onlyRead:
        # overwrite the normal logic and measure the time to read and compute
        # derived variables in the input file
        if args.variables != 'n/a':
            derivedName = args.variables[0]
        _, ts_read = read_with_derived(args.testName)
        print("Time to read expression:", ts_read)
        exit()

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
        if args.verbose:
            print("Generated a dataset of %d variables of shape %s" %(
                len(dataset), dataset[0].shape))
    elif args.file != 'n/a': # if we are reading the data from a file
        dataset = read_dataset(args.file, args.variables,
                               verbose=args.verbose)
    else:
        print("ERR There is no option chosen for generating data."
              "Use either -s (random arrays) or -f (data from a file)")
        exit()

    dataset = filter_dataset(dataset, args.expression)
    res_apply, ts_apply = apply_expression(args.expression, dataset,
                                           verbose=args.verbose)
    if args.verbose:
        print("Time to apply expression:", ts_apply)

    ts_write = write_with_derived("out_%s.bp" %(args.testName), args.expression,
                       adios2.DerivedVarType.StatsOnly, dataset)
    if args.verbose:
        print("Time to write expression:", ts_write)

    res_read, ts_read = read_with_derived("out_%s.bp" %(args.testName))
    if args.verbose:
        print("Time to read expression:", ts_read)

    if not np.allclose(res_apply, res_read, rtol=1e-03):
        print("ERR Correctness test failed")

    print("%s %.4f %.4f %.4f" %(dataset[0].shape, ts_apply, ts_write, ts_read))
