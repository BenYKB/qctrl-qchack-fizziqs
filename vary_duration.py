# create a pulse and send it to the Q-CTRL cloud
from qctrl import Qctrl
import os

import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from typing import List, Tuple

from scipy.optimize import curve_fit

qctrl = Qctrl(email=os.getenv('EMAIL'), password=os.getenv('PASSWORD'))

def estimate_probability_of_one(measurements):
    size = len(measurements)
    probability = np.mean(measurements)
    standard_error = np.std(measurements) / np.sqrt(size)
    return probability, standard_error


def get_qubit_population(duration: float, control_count: int = 1, ai: float = 0, a_q: float = 0) -> List[
    Tuple[float, float]]:
    """
    Gets the qubit population that corresponds to a certain pulse_duration and pulse (a_i, a_q)
    :return: a list of qubit populations (probability, standard_error)
    """
    segment_count = 16
    shot_count = 32

    controls = []
    for k in range(control_count):
        # Create a random string of complex numbers for each controls.
        real_part = np.ones(segment_count) * ai
        imag_part = np.ones(segment_count) * a_q
        values = real_part + 1j * imag_part
        controls.append({"duration": duration, "values": values})

    # Obtain the results of the experiment.
    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    qubit_population = []
    experiment_measurements = experiment_results.measurements
    for k, measurement_counts in enumerate(experiment_measurements):
        print(f"control #{k}: {measurement_counts}")
        # print(estimate_probability_of_one(measurement_counts))
        qubit_population.append(estimate_probability_of_one(measurement_counts))

    return qubit_population


duration_interval = 5
min_duration = 100
max_duration = 300
pulse_durations = np.arange(min_duration, max_duration, duration_interval).tolist()
a_i = 0.7  # pulse amplitude (real)

qubit_populations = []

for pulse_duration in pulse_durations:
    probability, standard_error = get_qubit_population(pulse_duration, control_count=1, ai=a_i, a_q=0)[0]
    print(probability, standard_error)
    qubit_populations.append(probability)

plt.plot(pulse_durations, qubit_populations, 'o')

def fit_function_bounds(x_values, y_values, function, bound_values):
    fitparams, conv = curve_fit(function, x_values, y_values, bounds=bound_values)
    y_fit = function(x_values, *fitparams)
    return fitparams, y_fit


fit_parameters, y_fit = fit_function_bounds(
    np.array(pulse_durations),
    np.array(qubit_populations),
    lambda x, A, rabi_freq, phi: A * np.cos(2 * np.pi * rabi_freq * x + phi) ** 2,
    (
        [0.8, np.abs(a_i * 8 * 1e7), -4],
        [1, np.abs(a_i * 11 * 1e7), 4],
    ),
    )

print("Drive amplitude:", a_i)
print("Fitted Rabi frequency [Hz]:", fit_parameters[1])

plt.plot(
    pulse_durations,
    fit_parameters[0]
    * np.cos(2 * np.pi * fit_parameters[1] * np.array(pulse_durations) + fit_parameters[2]) ** 2,
    color="red",
    )

plt.show()
