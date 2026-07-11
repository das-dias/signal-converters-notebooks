from functools import reduce
import numpy as np
import pdb

# define the function to convert a binary vector to decimal
def bin2dec(x, width, reverse=False):
    """Convert a numpy array from binary to decimal.
    IF the input is an array of binary arrays, 
    the returned output is an array of the corresponding 
    decimals in their corresponding indexes.
    Parameters:
        x: numpy array
        b: base of the number system
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    if len(x) == width:
        if reverse:
            x = np.flip(x)
        #assert len(x) == width, "The length of the vector must be equal to the number of bits"
        return reduce(lambda x,b: 2*x + b, x)
    assert len(x[0]) == width, "The length of each binary vector must be equal to the number of bits"
    if reverse:
        x = np.flip(x, axis=1)
    return np.array(np.array([reduce(lambda xval,b: 2*xval + b, xval) for xval in x]))

def dec2bin(x, width, reverse=False):
    """Convert a numpy array from decimal to binary
    If the input is an array of decimals, the returned 
    binary arrays are the codes corresponding to 
    each decimal in its corresponding index.
    Parameters:
        x: numpy array
        b: base of the number system
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    x = x.astype(int)
    if x.size == 1:
        arr = np.array([int(c) for c in np.binary_repr(x, width=width)])
        return np.flip(arr) if reverse else arr
    arr = np.array([[int(c) for c in np.binary_repr(subx, width=width)] for subx in x])
    return np.flip(arr, axis=1) if reverse else arr

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


def binsub(
    a, b
):
    """Binary word subtraction.
    Args:
        a (_type_): _description_
        b (_type_): _description_
    """
    width_a = len(a) if a.size == 1 else a.shape[1]
    width_b = len(b) if b.size == 1 else b.shape[1]
    return dec2bin(bin2dec(a, width_a) - bin2dec(b, width_b), np.max([width_a, width_b]))

def binadd(
    a, b
):
    """Binary word addition.
    Args:
        a (_type_): _description_
        b (_type_): _description_
    """
    width_a = len(a) if a.size == 1 else a.shape[1]
    width_b = len(b) if b.size == 1 else b.shape[1]
    return dec2bin(bin2dec(a, width_a) + bin2dec(b, width_b), np.max([width_a, width_b]))
    

def digital_error_correction(
    douts,
    scale_factors,
    reverse=False,
    bin=True
):
    """Perform digital error correction
    
    Args:
        douts (list): The list of output codes of each stage of the pipeline.
        scale_factors (list): The list of scale factors between the output codes of each stage of the pipeline.
        reverse (bool): The flag to indicate if the output word must be reversed.
        bin (bool): The flag to indicate if the output word must be returned as a binary word.
    """
    if douts[0].size == 1:
        douts = [ np.array(codes).reshape(1, -1) for codes in douts ]
    # assert if there are equal number of codes in each dout
    assert len(set([ codes.shape[0] for codes in douts ])) == 1, "The number of codes in each dout must be equal"
    
    width = np.sum( [codes.shape[1] for codes in douts] ) - len(douts) + 1
    #word_out = np.zeros( (douts[0].shape[0], width) )
    # subtract 1 bit from each dout word
    # pad the first dout code word to the total output width, by adding zeros to the right
    #word_out[:, :douts[0].shape[1]] = douts[0]
    # align the msb of the next dout code word with the lsb of the previous dout code word
    # and add the next dout code word to the output word
    #word = bin2dec(word_out, width, reverse=reverse)
    
    #for i in range(1, len(douts)):
        # create padded auxiliar word
        #aux_word = np.zeros( (douts[i].shape[0], word_out.shape[1]) )    
        #aux = bin2dec(douts[i], douts[i].shape[1]) - 2**(douts[i].shape[1]-2) # subtract a shifted 1 from the output word of each stage to perform digital error correction
        #pdb.set_trace()
        #word += aux*2**(  ) # add the auxiliar word to the output word while aligning the lsb of the auxiliar word with the msb of the output word
    widths = [codes.shape[1] for codes in douts]
    #widths = widths
    word = bin2dec(douts[0], widths[0])*2**(width - widths[0])
    #pdb.set_trace()
    prev_exp = width - widths[0]
    for k in range(1,len(widths)):
        aux = bin2dec(douts[k], widths[k])
        #pdb.set_trace()
        aux -= 2**(widths[k] - 2)
        word += aux*2**(prev_exp + 1 - widths[k])
        #pdb.set_trace()
        prev_exp = widths[k-1] - widths[k]
        
    # take care of overflows
    word[word > 2**width - 1]   = 2**width - 1
    word[word < 0]              = 0
    # return the output word
    return dec2bin(word, width, reverse=reverse) if bin else word

    