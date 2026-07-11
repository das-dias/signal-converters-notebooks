from functools import reduce
import numpy as np


def bin2dec(x, width, reverse=False):
    """Convert a numpy array from binary to decimal.
    If the input is an array of binary arrays,
    the returned output is an array of the corresponding
    decimals in their corresponding indexes.
    Parameters:
        x (numpy.ndarray): Binary vector or array of binary vectors.
        width (int): Number of bits in each binary word.
        reverse (bool): If True, interpret input as LSB-first.
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    if len(x) == width:
        if reverse:
            x = np.flip(x)
        return reduce(lambda acc, bit: 2 * acc + bit, x)
    assert len(x[0]) == width, (
        "The length of each binary vector must be equal to the number of bits"
    )
    if reverse:
        x = np.flip(x, axis=1)
    return np.array([reduce(lambda acc, bit: 2 * acc + bit, xval) for xval in x])


def dec2bin(x, width, reverse=False):
    """Convert a numpy array from decimal to binary.
    If the input is an array of decimals, the returned
    binary arrays are the codes corresponding to
    each decimal in its corresponding index.
    Parameters:
        x (numpy.ndarray): Decimal value or array of decimal values.
        width (int): Number of bits in the output binary word.
        reverse (bool): If True, return output as LSB-first.
    """
    x = np.array(x) if not isinstance(x, np.ndarray) else x
    x = x.astype(int)
    if x.size == 1:
        arr = np.array([int(c) for c in np.binary_repr(x, width=width)])
        return np.flip(arr) if reverse else arr
    arr = np.array([[int(c) for c in np.binary_repr(subx, width=width)] for subx in x])
    return np.flip(arr, axis=1) if reverse else arr


def ideal_dac(vref: float, n_bits: int):
    """Define the transfer function of an ideal
    DAC biased by vref and presenting an n_bits resolution.
    Parameters:
        vref (float): The biasing voltage of the electronic system.
        n_bits (int): The resolution of the DAC.
    Returns:
        function(Din): the lambda function defining the transfer function of the DAC
    """
    vlsb = vref / (2**n_bits)
    return lambda x: bin2dec(x, n_bits) * vlsb


def ideal_adc(vref: float, nbits: int, roundf):
    """Define the transfer function of an ideal
    ADC biased by vref and presenting an nbits resolution.
    Parameters:
        vref (float): The biasing voltage of the electronic system.
        nbits (int): The resolution of the ADC.
        roundf (function): The rounding function to be used.
    Returns:
        function(Vin): the lambda function defining the transfer function of the ADC
    """
    assert roundf in [np.round, np.ceil, np.floor], (
        "The round function must be numpy.floor, numpy.ceil or numpy.round"
    )
    vlsb = vref / (2**nbits)
    maxcode = 2**nbits - 1
    return lambda x: dec2bin(np.clip(roundf(x / vlsb).astype(int), 0, maxcode), nbits)


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
    assert roundf in [np.round, np.floor, np.ceil], (
        "The rounding function must be one of the following: np.round, np.floor, np.ceil"
    )
    vlsb = vref / (2**n_bits)
    vtrans = np.arange(vlsb, vref, vlsb) * gain + ofst
    qnoise = np.random.uniform(-vnq, vnq, len(vtrans)) if vnq > 0 else 0
    vtrans += qnoise

    def _transfer_function(x, vtrans=vtrans, n_bits=n_bits):
        assert len(vtrans) == 2**n_bits - 1, (
            "The number of transition voltages must be equal to the number of transitions between output codes of the ADC"
        )
        if x.size == 1:
            ntrans = np.sum(x > vtrans)
        else:
            ntrans = np.array([np.sum(xval > vtrans) for xval in x])
        return dec2bin(ntrans, n_bits)

    return lambda x: _transfer_function(x, vtrans, n_bits), vtrans


def binsub(a, b):
    """Binary word subtraction.
    Args:
        a (numpy.ndarray): First binary word or array of binary words.
        b (numpy.ndarray): Second binary word or array of binary words.
    """
    width_a = len(a) if a.size == 1 else a.shape[1]
    width_b = len(b) if b.size == 1 else b.shape[1]
    return dec2bin(
        bin2dec(a, width_a) - bin2dec(b, width_b), np.max([width_a, width_b])
    )


def binadd(a, b):
    """Binary word addition.
    Args:
        a (numpy.ndarray): First binary word or array of binary words.
        b (numpy.ndarray): Second binary word or array of binary words.
    """
    width_a = len(a) if a.size == 1 else a.shape[1]
    width_b = len(b) if b.size == 1 else b.shape[1]
    return dec2bin(
        bin2dec(a, width_a) + bin2dec(b, width_b), np.max([width_a, width_b])
    )


def digital_error_correction(
    douts,
    scale_factors,
    reverse=False,
    as_binary=True,
):
    """Perform digital error correction.

    Args:
        douts (list): The list of output codes of each stage of the pipeline.
        scale_factors (list): The list of scale factors between the output codes of each stage of the pipeline.
        reverse (bool): The flag to indicate if the output word must be reversed.
        as_binary (bool): The flag to indicate if the output word must be returned as a binary word.
    """
    if douts[0].size == 1:
        douts = [np.array(codes).reshape(1, -1) for codes in douts]
    assert len(set([codes.shape[0] for codes in douts])) == 1, (
        "The number of codes in each dout must be equal"
    )

    width = np.sum([codes.shape[1] for codes in douts]) - len(douts) + 1
    widths = [codes.shape[1] for codes in douts]
    word = bin2dec(douts[0], widths[0]) * 2 ** (width - widths[0])
    prev_exp = width - widths[0]
    for k in range(1, len(widths)):
        aux = bin2dec(douts[k], widths[k])
        aux -= 2 ** (widths[k] - 2)
        word += aux * 2 ** (prev_exp + 1 - widths[k])
        prev_exp = widths[k - 1] - widths[k]

    word[word > 2**width - 1] = 2**width - 1
    word[word < 0] = 0
    return dec2bin(word, width, reverse=reverse) if as_binary else word
