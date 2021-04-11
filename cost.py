import numpy as np

state_0 = np.array([[1],[0]])
not_gate = np.array([[0,1],[1,0]])
h_gate = np.array([[1,1],[1,-1]])/np.sqrt(2)

def real_parameters(complex_array):
    n = (len(complex_array)-1)/2
    real_part = np.real(complex_array)
    imag_part = np.imag(complex_array)

    real_array = np.zeros((4*n+1,))
    real_array[0] = real_part[0]
    real_array[1:2*n] = real_part[1:]
    real_array[2*n+1:] = imag_part[1:]

    return real_array

def complex_parameters(real_array):
    n = int((len(real_array)-1)/4)
    T = real_array[0] + 0j
    real_part = real_array[1:2*n+1]
    imag_part = real_array[2*n+1:]
    complex_array = np.zeros((2*n+1,), dtype=np.complex128)

    complex_array[0] = T
    complex_array[1:] = real_part + 1j*imag_part

    return complex_array

def signal_concatenate(not_signal, h_signal, pattern):
    '''pattern is an array of 0s and 1s, representing the order of not_signal (0) and h_signal (1) '''
    signal_list = [not_signal, h_signal]
    signal = np.array([])
    for which in pattern:
        signal = np.append(signal, signal_list[which])
    return signal

def generate_patterns(n):
    repeated_array_not = np.zeros((n,), dtype=int)
    repeated_array_h = np.ones((n,), dtype=int)
    random_array = np.random.randint(0,2,n)
    return (repeated_array_not, repeated_array_h, random_array)

def cost_determination(measurement_counts, pattern):
    '''pattern is an array of 0s and 1s, representing the order of not_signal (0) and h_signal (1) 
    measurement_counts is the object from the measurement list given by qctrl.functions.calculate_qchack_measurements '''
    
    n = pattern.size
    
    gates = [not_gate, h_gate]
    operator = np.array([[1,0],[0,1]])
    for which in pattern:
        operator = gates[which] @ operator
    final_state = operator @ state_0

    probability_of_one_ideal = np.abs(final_state[1][0])**2
    probablity_of_one_measured = measurement_counts.count(1)/len(measurement_counts)
    cost_value = np.abs(probability_of_one_ideal-probablity_of_one_measured)/n

    return cost_value