import os
import numpy as np
import re
from collections import defaultdict, OrderedDict

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
    hdi = np.percentile(array, interval)
    return hdi


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


def prepare_dict(header, nsamples):
    name_without_index = f7([x.split('.')[0] for x in header])

    # Prepare dict
    Extract = OrderedDict()
    header_name_to_merged = defaultdict(list) # Name, [dimension]
    for i in name_without_index:
        re_search = re.compile(i+'(\..+)?$')
        # Assume that names are not too close...
        match_idx = [m.group(0) for l in header for m in [re_search.search(l)] if m]

        # Assume last entry represents size and dimensions
        e = match_idx[-1]
        re_dim = [int(x) for x in re.findall('\.(\d+)', e)]
        re_dim.append(nsamples)
        Extract[i] = np.zeros((re_dim))

        # Map it
        for j in match_idx:
            re_dim = [int(x)-1 for x in re.findall('\.(\d+)', j)]
            header_name_to_merged[j] = [i, re_dim]

    return (Extract, header_name_to_merged)




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
    summary_flag = True
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
                if summary:
                    draws = np.empty((niter+1, len(header)))
                else:
                    draws = np.empty((niter, len(header)))

                # No more comments
                comment_line = False
                continue

            if comment_line and find_equal.search(line):
                # Comment + equals sign
                line = find_replace.sub('', line)
                key, val = line.strip().split('=')
                attributes[key.strip()] = val.strip()
            elif not comment_line:
                if not summary and summary_flag:
                    summary_flag = False
                    continue

                # Draws.
                draws[n_current_iter, :] = line.strip().split(',')
                n_current_iter += 1
        header = np.array(header)

    
    # Translate to OrderedDict
    Extract, header_map = prepare_dict(header, niter)
    for idx, i in enumerate(header):
        l = header_map[i]
        arr = Extract[l[0]]
        ix = l[1]
        draw_values = draws[:, idx]
        

        # This is a bit stupid. Currently only works with stan-objects of dim<2
        if len(ix) == 2:
            arr[ix[0]][ix[1]] = draw_values
        elif len(ix) == 1:
            arr[ix[0]] = draw_values
        elif len(ix) == 0:
            arr = draw_values
        else:
            raise ValueError("Not implemented for arrays with dim>2")

    return (Extract, attributes)


def read_stan_csv(csvfiles, warmup=False):
    """Read Stan files produced by CmdStan.

    Reads the files specified in csvfiles and reports back an OrderedDict.
    Mimicks the behavior of PyStan.    
    
    Parameters
    ----------
    csvfiles : list
        List of csvfiles produced by CmdStan. One for each chain.

    warmup : boolean
        Return warmup samples as the first Nwarmup in arrays
    
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

    # Read samples
    nsamples = int(attributes['num_samples'])
    nwarmup = int(attributes['num_warmup'])
    niter =  nsamples
    if warmup:
        niter += nwarmup
    draws = np.empty((niter*nchains, len(header))) # Iteration, Chain, Value

    n_current_iter = 0
    for mcmc_file in csvfiles:
        WARMUP_STEP = True
        with open(mcmc_file, 'r') as fh:
            for line in fh:
                # Skip comments and header
                if line.startswith('# Adaptation terminated'):
                    # Sampling started
                    WARMUP_STEP = False
                elif line.startswith('#') or line.startswith('lp'):
                    continue
                elif not warmup and WARMUP_STEP:
                        continue

                vals = line.strip().split(',')
                if len(vals) <= 1:
                    continue

                draws[n_current_iter, :] = vals
                n_current_iter += 1

    # Translate to OrderedDict
    Extract, header_map = prepare_dict(header, niter*nchains)
    for idx, i in enumerate(header):
        l = header_map[i]
        arr = Extract[l[0]]
        ix = l[1]
        draw_values = draws[:, idx]
        

        # This is a bit stupid. Currently only works with stan-objects of dim<2
        if len(ix) == 2:
            arr[ix[0]][ix[1]] = draw_values
        elif len(ix) == 1:
            arr[ix[0]] = draw_values
        elif len(ix) == 0:
            arr = draw_values
        else:
            raise ValueError("Not implemented for arrays with dim>2")

    return (Extract, attributes)

