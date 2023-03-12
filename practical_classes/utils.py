from functools import reduce
import numpy as np
import pdb

# define the function to convert a binary vector to decimal
def bin2dec(x, width):
    """Convert a numpy array from binary to decimal.
    IF the input is an array of binary arrays, 
    the returned output is an array of the corresponding 
    decimals in their corresponding indexes.
    Parameters:
        x: numpy array
        b: base of the number system
    """
    x = np.array(x)
    if x.size == width:
        assert len(x) == width, "The length of the vector must be equal to the number of bits"
        return reduce(lambda x,b: 2*x + b, x)
    assert len(x[0]) == width, "The length of the vector must be equal to the number of bits"
    return np.array(np.array([reduce(lambda xval,b: 2*xval + b, xval) for xval in x]))
def dec2bin(x, width):
    """Convert a numpy array from decimal to binary
    If the input is an array of decimals, the returned 
    binary arrays are the codes corresponding to 
    each decimal in its corresponding index.
    Parameters:
        x: numpy array
        b: base of the number system
    """
    x = np.array(x)
    if x.size == 1:
        return np.array([int(c) for c in np.binary_repr(x, width=width)])
    return np.array([np.array([int(c) for c in np.binary_repr(subx, width=width)]) for subx in x])

def ideal_dac(vref:float, n_bits:int):
    """Define the transfer function of an ideal 
    DAC biased by vref and presenting an n_bits resolution.
    Parameters:
        vref (float): The biasing voltage of the electronic system.
        n_bits (int): The resolution of the DAC.
    Returns:
        function(Din): the lambda function defining the transfer function of the DAC
    """
    vlsb = vref/(2**n_bits) # compute the fundamental step voltage between quantized levels
    return lambda x: bin2dec(x, n_bits)*vlsb # return the converter funtion

def ideal_adc(vref:float, nbits:int, roundf):
    """Define the transfer function of an ideal 
    ADC biased by vref and presenting an n_bits resolution.
    Parameters:
        vref (float): The biasing voltage of the electronic system.
        n_bits (int): The resolution of the DAC.
        roundf (function): The rounding function to be used.
    Returns:
        function(Vin): the lambda function defining the transfer function of the ADC
    """
    assert roundf in [np.round, np.ceil, np.floor], "The round function must be numpy.floor, numpy.ceil or numpy.round"
    vlsb = vref/(2**nbits)
    return lambda x: dec2bin(roundf(x/vlsb).astype(int), nbits)

def nonideal_adc(vref, n_bits, ofst=0, gain=1, vnq=0, roundf=np.round):
    """Implements a non-ideal ADC with linear errors.
    The ADC is modeled as a comparator with a hysteresis.
    The input voltage is compared with the transition voltage
    and if the input voltage is greater than the transition voltage
    a transition occurs.
    Parameters:
        vref (float): The biasing voltage of the electronic system.
        n_bits (int): The resolution of the ADC.
        ofst (float): The offset of the ADC.
        gain (float): The gain of the ADC.
        vnq (float): The quantization noise voltage level of the ADC.
        roundf (function): The rounding function to be used.
    Returns:
        function(Vin): the lambda function defining the transfer function of the ADC
        vtrans (numpy array): the transition voltages
    """
    assert n_bits > 0, "The number of bits must be greater than zero"
    assert roundf in [np.round, np.floor, np.ceil], "The rounding function must be one of the following: np.round, np.floor, np.ceil"
    # compute the fundamental step voltage between quantized levels
    vlsb = vref/(2**n_bits)
    # compute the transition voltages
    vtrans = np.arange(vlsb, vref, vlsb)*gain + ofst
    # define the quantization noise to be added 
    # to the transition voltages
    qnoise = np.random.uniform(-vnq, vnq, len(vtrans)) if vnq > 0 else 0
    vtrans += qnoise
    def _transfer_function(x, vtrans=vtrans, n_bits=n_bits):
        #pdb.set_trace()
        assert len(vtrans) == 2**n_bits-1, "The number of transition voltages must be equal to the number of transitions between output codes of the ADC"
        # compute the number of transitions
        if x.size == 1:
            ntrans = np.sum(x > vtrans)
        else:
            ntrans = np.array([np.sum(xval > vtrans) for xval in x])	
        # compute the output code and return it 
        return dec2bin(ntrans, n_bits)
    return lambda x: _transfer_function(x, vtrans, n_bits), vtrans