import os
import numpy as np
import re
from collections import OrderedDict

def compute_hdi(array, interval=[2.5, 50, 97.5]):
    '''Return the high-density interval

    Parameters
    ----------
    array : numpy array
        Array containing the sampled values

    interval : list 
        Interval range for HDI

    Returns
    -------
    hdi : array
        Array of size equivalent to interval with specified percentiles
    '''    
    hdi = numpy.percentile(array, interval)

def softmax(matrix):
    '''Calculates the row-wise softmax

    Calculate the row-wise softmax for a matrix of size NxM,
        matrix[N,:] = exp(matrix[N,:]) / (sum(matrix[N,:]), axis=1)

    Parameters
    ----------
    array : array
        Array of inputs

    Returns
    -------
    softmax_array : array
        Array with the softmax function applied row-wise
    
    Taken from a SO anwer: http://tinyurl.com/angxm5
    '''
    e = np.exp(w)
    softmax_array = e / np.sum(e, axis=1)[:, np.newaxis]
    return softmax_array


def f7(seq):
    '''Return the unique entries in list in same order

    Parameters
    ----------
    seq : list
        list of inputs

    Returns
    -------
    seen_uniq : list
        List containing only the unique entries in the same order as the
        original list
    
    Taken from a SO anwer: http://tinyurl.com/angxm5
    '''
    seen = set()
    seen_add = seen.add
    seen_uniq = [x for x in seq if not (x in seen or seen_add(x))]
    return seen_uniq

def read_one_stan_csv(csvfile, summary=False):
    """Read one Stan file produced by CmdStan.

    Read the output from a Variational fit that only produces one
    csv file
    
    Parameters
    ----------
    csvfile : str
        csvfile produced by CmdStan.

    summary : boolean
        Include the first line of values, which is the mean of the 
        variational approximation. 
    
    Returns
    -------
    Extract : OrderedDict
        OrderedDict containing the samples. Ordering follows the header.

    Attributes : dictionary of attributes extracted from the comment section
    
    Raises
    ------
    ValueError
        Input is empty
    OSError
        File does not exist, or cannot be read.
    KeyError
        Something is wrong with the comment section
    """
    if len(csvfile) < 1:
        raise ValueError("Input is empty!")
    if not os.path.exists(csvfile):
        raise os.OSError("File does not exist!")

    # RegEx for comment matching
    find_equal = re.compile('=')
    find_replace = re.compile('\#|\(Default\)')


    # Read file, line by line
    comment_line = True
    header = None
    attributes = {}
    draws = None
    n_current_iter = 0

    # Read comments

    with open(csvfile, 'r') as fh:
        for line in fh:
            if not line.startswith('#') and comment_line:
                # Grab header
                header = line.strip().split(',')

                # Assume that output samples has been seen
                try:
                    niter = int(attributes['output_samples'])
                except KeyError:
                    raise KeyError("output_samples not found in attributes!")

                # Initialize array
                # +1 for the first iteration which summarises
                draws = np.empty((niter+1, len(header))) 

                # No more comments
                comment_line = False
                continue

            if comment_line and find_equal.search(line):
                # Comment + equals sign
                line = find_replace.sub('', line)
                key, val = line.strip().split('=')
                attributes[key.strip()] = val.strip()
            elif not comment_line:
                # Draws.
                draws[n_current_iter, :] = line.strip().split(',')
                n_current_iter += 1
        header = np.array(header)

    name_without_index = f7([x.split('.')[0] for x in header])

    Extract = OrderedDict()
    for i in name_without_index:
        # TODO: fix this tomorrow
        re_search = re.compile(i+'(.+)?$')
        # Assume that names are not too close...
        match_idx = [m.group(0) for l in header for m in [re_search.search(l)] if m]




    for idx, i in enumerate(header):
        Extract[i] = draws[:, idx]

    return (Extract, attributes)


def read_stan_csv(csvfiles):
    """Read Stan files produced by CmdStan.

    Reads the files specified in csvfiles and reports back an OrderedDict.
    Mimicks the behavior of PyStan.    
    
    Parameters
    ----------
    csvfiles : list
        List of csvfiles produced by CmdStan. One for each chain.
    
    Returns
    -------
    Extract : OrderedDict
        OrderedDict containing the samples. Ordering follows the header.

    Attributes : dictionary of attributes extracted from the comment section
    
    Raises
    ------
    ValueError
        If a list is not provided
    """
    if not type(csvfiles) == list:
        raise ValueError("Input must be a list!")
    nchains = len(csvfiles)

    find_equal = re.compile('=')
    find_replace = re.compile('\#|\(Default\)')
    attributes = {}

    # Read first file to catch comments and header
    with open(csvfiles[0], 'r') as fh:
        for line in fh:
            if not line.startswith('#'):
                header = line.strip().split(',')
                break
            if find_equal.search(line):
                # Comment + equals sign
                line = find_replace.sub('', line)
                key, val = line.strip().split('=')
                attributes[key.strip()] = val.strip()

    # Initialize array to hold values
    niter = int(attributes['num_samples']) + int(attributes['num_warmup'])
    draws = np.empty((niter*nchains, len(header))) # Iteration, Chain, Value

    n_current_iter = 0
    for mcmc_file in csvfiles:
        with open(mcmc_file, 'r') as fh:
            for line in fh:
                # Skip comments and header
                if line.startswith('#') or line.startswith('lp'):
                    continue

                vals = line.strip().split(',')
                if len(vals) <= 1:
                    continue

                draws[n_current_iter, :] = vals
                n_current_iter += 1

    Extract = OrderedDict()
    for idx, i in enumerate(header):
        Extract[i] = draws[:, idx]

    return (Extract, attributes)














