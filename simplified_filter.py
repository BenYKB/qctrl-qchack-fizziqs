import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from qctrl import Qctrl

import os
from dotenv import load_dotenv

load_dotenv()
email = os.getenv('EMAIL')
password = os.getenv('PASS')

qctrl = Qctrl(email=email, password=password)

def get_filtered_results(duration=1,
                         values=np.array([np.pi]),
                         shots=1024,
                         repetitions=1):
    # 1. Limits for drive amplitudes
    assert np.max(values) <= 1.0
    assert np.min(values) >= -1.0
    max_drive_amplitude = 2 * np.pi * 20  # MHz

    # 2. Dephasing error
    dephasing_error = -2 * 2 * np.pi  # MHz

    # 3. Amplitude error
    amplitude_i_error = 0.98
    amplitude_q_error = 1.03

    # 4. Control line bandwidth limit
    cut_off_frequency = 2 * np.pi * 10  # MHz
    resample_segment_count = 1000

    # 5. SPAM error confusion matrix
    confusion_matrix = np.array([[0.99, 0.01], [0.02, 0.98]])

    # Lowering operator
    b = np.array([[0, 1], [0, 0]])
    # Number operator
    n = np.diag([0, 1])
    # Initial state
    initial_state = np.array([[1], [0]])

    with qctrl.create_graph() as graph:
        # Apply 1. max Rabi rate.
        values = values * max_drive_amplitude

        # Apply 3. amplitude errors.
        values_i = np.real(values) * amplitude_i_error
        values_q = np.imag(values) * amplitude_q_error
        values = values_i + 1j * values_q

        print(len(values))
        # Apply 4. bandwidth limits
        drive_unfiltered = qctrl.operations.pwc_signal(duration=duration, values=values)
        drive_filtered = qctrl.operations.convolve_pwc(
            pwc=drive_unfiltered,
            kernel_integral=qctrl.operations.sinc_integral_function(cut_off_frequency),
        )
        #trying to convert to number
        drive_unfiltered_num = qctrl.operations.discretize_stf(drive_unfiltered, duration, resample_segment_count)
        drive_filtered_num = qctrl.operations.discretize_stf(drive_unfiltered, duration, resample_segment_count)
        #print(drive_unfiltered_num.values)
        print(type(drive_filtered_num))
        #print(drive_filtered_num)
        print(drive_filtered_num.values)

        return drive_filtered

if __name__ == "__main__":
    max_rabi_rate = 20 * 2 * np.pi  # MHz
    not_duration = np.pi / (max_rabi_rate)  # us
    h_duration = np.pi / (2 * max_rabi_rate)  # us
    shots = 1024

    values = np.array([1.0, 0.5, 0.7, 0.9, 0.8, 0.3, 0.1])
    get_filtered_results(duration=not_duration, values=values, shots=shots)