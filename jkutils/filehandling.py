import numpy as np
import re
from os import path
from random import random, sample
import csv

def mkdir_p(path):
    """Like mkdir -p it creates all directories
    in a path if they do not exist"""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def parse_header(headers):
    """
    In Python3, this method could be done in one line:
    return {name:i for i, name in enumerate(headers)}
    """
    header_names = {}
    for idx, name in enumerate(headers):
        header_names[name] = idx

    return header_names

def read_data_file(filename, separator = ','):
    """Columns are data dimensions, rows are sample data.
    Whitespace separates the columns. Returns a python list [[]].
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f.readlines(), delimiter=separator)
        inputs = [row for row in reader]

    #Make sure it has a consistent structure
    col_len = len(inputs[0])
    for line in inputs:
        assert(len(line) == col_len)
    return inputs

def parse_file(filename, inputcols = None, ignorecols = [], ignorerows = [],
               normalize = True, separator = None, use_header = False,
               fill_average = True):
    return parse_data(np.array(read_data_file(filename, separator = separator)),
                      inputcols, ignorecols, ignorerows, normalize, use_header,
                      fill_average)

def parse_data(inputs, inputcols = None, ignorecols = [], ignorerows = [],
               normalize = True, use_header = False, fill_average = True):
    """inputs is an array of data columns. targetcols is either an int
    describing which column is a the targets or it's a list of several ints
    pointing to multiple target columns.
    Input columns follows the same pattern, but are not necessary if the inputs
    are all that's left when target columns are subtracted.
    Ignorecols can be used instead if it's easier to specify which columns to
    ignore instead of which are inputs.
    Ignorerows specify which, if any, rows should be skipped.

    if useHeader is True, the first line is taken to be the header containing
    column names. This will be parsed and inputcols and targetcols must now
    specify the columns with names instead.
    The first line (the header) is subsequently ignored from the dataset so this
    doesn't have to be specified seperately.
    """

    if use_header:
        #Parse the header line, get a hash where the keys are the names and the values are the column numbers.
        col_names = parse_header(inputs[0])
        #If not present in ignore list, add it
        if 0 not in ignorerows:
            ignorerows.append(0)
        #Also verify that the names specified are indeed valid column names, otherwise throw an exception about it.
        for name in inputcols:
            if name not in col_names:
                raise ValueError(str(name) + ' is not a column name ({})'.format(col_names))

        #Now use a translated array from names to numbers. Carry on as before
        named_inputcols = inputcols
        inputcols = [col_names[name] for name in named_inputcols]

    try:
        inputs[:, 0]
    except (TypeError, IndexError):
        #Slicing failed, inputs is not a numpy array. Alert user
        raise TypeError('Slicing of inputs failed, \
it is probably not a numpy array: ' + str(inputs))

    if not inputcols:
        inputcols = list(range(len(inputs[0])))
        destroycols = []
        try:
            destroycols.append(int(targetcols)) #Only if it's an int
        except TypeError:
            destroycols.extend(targetcols)
        try:
            destroycols.append(int(ignorecols)) #Only if it's an int
        except TypeError:
            destroycols.extend(ignorecols)

        inputcols = np.delete(inputcols, destroycols, 0)

    if fill_average:
        replace_empty_with_avg(inputs, inputcols)

    for line in range(len(inputs)):
        keep_only_numbers(line, inputcols, ignorerows)

    inputs = np.delete(inputs, ignorerows, 0)

    inputs = np.array(inputs[:, inputcols], dtype = 'float64')

    if normalize:
        inputs = normalizeArray(inputs)

    #Now divide the input into test and validation parts

    return inputs

def keep_only_numbers(line, all_cols, ignorerows):
    for col in all_cols: #check only valid columns
        try:
            float(col)
        except ValueError: #This row contains crap, get rid of it
            ignorerows.append(line)
            break #skip to next line

def replace_empty_with_avg(inputs, inputcols):
    for col in inputcols:
        binary = False
        valid_inputs = np.array([], dtype = 'float64')
        for val in inputs[:, col]:
            try:
                if float(val) != 0 and float(val) != 1:
                    binary = False
                valid_inputs = np.append(valid_inputs, float(val))
            except ValueError:
                pass
        avg_val = valid_inputs.mean()
        for i in range(len(inputs)):
            try:
                float(inputs[i, col])
            except ValueError:
                if binary:
                    inputs[i, col] = sample(valid_inputs,
                                            1)[0]
                else:
                    inputs[i, col] = avg_val


def replace_empty_with_random(inputs, inputcols):
    for col in inputcols:
        valid_inputs = np.array([], dtype = 'float64')
        for val in inputs[:, col]:
            try:
                valid_inputs = np.append(valid_inputs, float(val))
            except ValueError:
                pass
        for i in range(len(inputs)):
            try:
                float(inputs[i, col])
            except ValueError:
                #Sample returns a list, access first and only element
                inputs[i, col] = sample(valid_inputs, 1)[0]

def normalizeArray(array):
    '''Returns a new array, will not modify existing array.
    Normalization is simply subtracting the mean and dividing by the
    standard deviation (for non-binary arrays).'''

    inputs = np.copy(array)
    #First we must determine which columns have real values in them
    #Basically, if it isn't a binary value by comparing to 0 and 1
    for col in range(len(inputs[0])):
        real = False
        for value in inputs[:, col]:
            if value != 0 and value != 1:
                real = True
                break #No point in continuing now that we know they're real
        if real:
            #Subtract the mean and divide by the standard deviation
            inputs[:, col] = (inputs[:, col] - np.mean(inputs[:, col])) / np.std(inputs[:, col])

    return inputs

def normalizeArrayLike(test_array, norm_array):
    ''' Takes two arrays, the first is the test set you wish to have
    normalized as the second array is normalized.
    Normalization is simply subtracting the mean and dividing by the
    standard deviation (for non-binary arrays).

    So what this method does is for every column in array1, subtract
    by the mean of array2 and divide by the STD of
    array2. Mean that both arrays have been subjected
    to the same transformation.'''
    if test_array.shape[1] != norm_array.shape[1] or len(test_array.shape) != 2 or len(norm_array.shape) != 2:
        #Number of columns did not match
        raise ValueError('Number of columns did not match in the two arrays.')
    test_inputs = np.copy(test_array)
    #First we must determine which columns have real values in them
    #Basically, if it isn't a binary value by comparing to 0 and 1
    for col in range(norm_array.shape[1]):
        real = False
        for value in norm_array[:, col]:
            if value != 0 and value != 1:
                real = True
                break #No point in continuing now that we know they're real
        if real:
            #Subtract the mean and divide by the standard deviation of the other array
             test_inputs[:, col] = (test_inputs[:, col] - np.mean(norm_array[:, col])) / np.std(norm_array[:, col])

    return test_inputs


def print_output(outfile, net, filename, targetcols, inputcols, ignorerows, normalize):
    '''
    Take a network and a file, outputting the inputs to that network and
    its output for said input on each line.
    '''
    inputs = read_data_file(filename)
    P, T = parse_file(filename, targetcols = targetcols, inputcols = inputcols,
                      ignorerows = ignorerows, normalize = normalize)
    outputs = net.sim(P).tolist()
    while len(inputs) > len(outputs):
        outputs.insert(0, ["net_output"])

    if len(inputs) < len(outputs):
        raise TypeError('Input is smaller than output!')

    lines = []
    for rawline in zip(inputs, outputs):
        line = ''
        for col in rawline[0]:
            line += str(col)
            line += ','
        for col in rawline[1]:
            line += str(col)

        lines.append(line + '\n')

    with open(outfile, 'w') as f:
        f.writelines(lines)

def get_validation_set(inputs, targets, validation_size = 0.2,
                       binary_column = None):
    '''
    Use binary column to specify a column of the targets which is binary,
    and can be used for
    stratified division of the dataset.
    '''

    if validation_size < 0 or validation_size > 1:
        raise TypeError('validation_size not between 0 and 1')
    test_inputs = []
    test_targets = []
    validation_inputs = []
    validation_targets = []

    #if the target has two values, assume one is a binary indicator. we want an equal share of both
    #matching the diversity of the dataset
    if binary_column is not None:
        zeros = targets[:, binary_column] == 0
        ones = targets[:, binary_column] == 1
    else:
        zeros = [True for x in range(len(targets))]
        ones = []

    #First zeros
    if len(zeros) > 0:
        inputs_zero = inputs[zeros]
        targets_zero = targets[zeros]
        for row in range(len(inputs_zero)):
            if random() > validation_size:
                test_inputs.append(inputs_zero[row])
                test_targets.append(targets_zero[row])
            else:
                validation_inputs.append(inputs_zero[row])
                validation_targets.append(targets_zero[row])
    #Then ones
    if len(ones) > 0:
        inputs_ones = inputs[ones]
        targets_ones = targets[ones]
        for row in range(len(inputs_ones)):
            if random() > validation_size:
                test_inputs.append(inputs_ones[row])
                test_targets.append(targets_ones[row])
            else:
                validation_inputs.append(inputs_ones[row])
                validation_targets.append(targets_ones[row])

    test_inputs = np.array(test_inputs, dtype = 'float64')
    test_targets = np.array(test_targets, dtype = 'float64')
    validation_inputs = np.array(validation_inputs, dtype = 'float64')
    validation_targets = np.array(validation_targets, dtype = 'float64')

    #shuffle the lists
    #BIG FUCKING ERROR HERE. NOTICE HOW YOU SHUFFLE INPUT AND TARGETS DIFFERENTLY? YOU FUCKING IDIOT !
    #np.random.shuffle(test_inputs)
    #np.random.shuffle(test_targets)
    #np.random.shuffle(validation_inputs)
    #np.random.shuffle(validation_targets)

    return ((test_inputs, test_targets), (validation_inputs, validation_targets))

def get_cross_validation_sets(inputs, targets, pieces, binary_column = None, return_indices = False):
    '''
    pieces is the number of validation sets that the data set should be divided into.
    '''
    totalcols = len(inputs[0, :]) + len(targets[0, :])

    training_sets = []
    validation_sets = []

    training_indices_sets = [[] for piece in range(pieces)]
    validation_indices_sets = [[] for piece in range(pieces)]

    training_input_sets = []
    training_target_sets = []
    validation_input_sets = []
    validation_target_sets = []

    #if (pieces < 2): #No validation set can be divided, return empty set
    #    training_indices_sets[0].extend(range(totalrows))
    #else:

    #if the target has two values, assume one is a binary indicator. we want an equal share of both
    #matching the diversity of the dataset
    all = np.arange(len(targets))

    if binary_column is not None:
        zeros = all[targets[:, binary_column] == 0]
        ones = all[targets[:, binary_column] == 1]
    else:
        zeros = all
        ones = []

    #Make sure to randomize them before division
    np.random.shuffle(zeros)
    np.random.shuffle(ones)

    def divide_sets(indices):
        sets = np.array_split(indices, pieces)
        k = 0
        for set in range(len(sets)):
            validation_indices_sets[set].extend(sets[k])
            #validation_input_sets[set].extend(inputs[sets[k]])
            #validation_target_sets[set].extend(targets[sets[k]])
            k += 1
            k %= pieces
            for piece in range(pieces - 1):
                training_indices_sets[set].extend(sets[k])
                #training_input_sets[set].extend(inputs[sets[k]])
                #raining_target_sets[set].extend(targets[sets[k]])
                k += 1
                k %= pieces
            #Do one final incrase, to make validation start at +1 next round
            k += 1
            k %= pieces

    #First zeros
    if len(zeros) > 0:
        divide_sets(zeros)

    #Then ones
    if len(ones) > 0:
        divide_sets(ones)

    #convert types
    for set in range(len(training_indices_sets)):
        trows = training_indices_sets[set]
        training_sets.append(np.zeros((len(trows), totalcols), dtype = 'float64'))

        training_sets[set][:, 0:len(inputs[0, :])] = inputs[trows]
        training_sets[set][:, len(inputs[0, :]):totalcols] = targets[trows]

        vrows = validation_indices_sets[set]
        validation_sets.append(np.zeros((len(vrows), totalcols), dtype = 'float64'))

        validation_sets[set][:, 0:len(inputs[0, :])] = inputs[vrows]
        validation_sets[set][:, len(inputs[0, :]):totalcols] = targets[vrows]

        #don't shuffle again, we need the indices
        #np.random.shuffle(training_sets[set])
        #np.random.shuffle(validation_sets[set])

        #Make return slices
        training_input_sets.append(training_sets[set][:, 0:len(inputs[0, :])])
        training_target_sets.append(training_sets[set][:, len(inputs[0, :]):totalcols])
        validation_input_sets.append(validation_sets[set][:, 0:len(inputs[0, :])])
        validation_target_sets.append(validation_sets[set][:, len(inputs[0, :]):totalcols])

    # Return a list of tuple, (Training, Validation)
    training = list(zip(training_input_sets, training_target_sets))
    validation = list(zip(validation_input_sets, validation_target_sets))

    if pieces == 1:
        #There will be nothing in training, and everything in validaiton. Swap them before return.
        training, validation = validation, training
        training_indices_sets, validation_indices_sets = validation_indices_sets, training_indices_sets

    if not return_indices:
        return list(zip(training, validation))
    else:
        # Return indices if they are of interest
        indices = list(zip(training_indices_sets, validation_indices_sets))
        data_sets = list(zip(training, validation))
        return (data_sets, indices)
