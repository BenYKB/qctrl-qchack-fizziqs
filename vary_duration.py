# create a pulse and send it to the Q-CTRL cloud
from qctrl import Qctrl
import os

import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import get_qctrl_style, plot_controls
from typing import List, Tuple

qctrl = Qctrl(email=os.getenv('EMAIL'), password=os.getenv('PASSWORD'))


def estimate_probability_of_one(measurements):
    size = len(measurements)
    probability = np.mean(measurements)
    standard_error = np.std(measurements) / np.sqrt(size)
    return probability, standard_error


def get_qubit_population(pulse_duration: float, control_count: int = 1, a_i: float = 0, a_q: float = 0) -> List[
    Tuple[float, float]]:
    segment_count = 16
    shot_count = 32

    controls = []
    for k in range(control_count):
        # Create a random string of complex numbers for each controls.
        real_part = np.ones(segment_count) * a_i
        imag_part = np.ones(segment_count) * a_q
        values = 0.15 * k * (real_part + 1j * imag_part)

        controls.append({"duration": pulse_duration, "values": values})

    # Obtain the results of the experiment.
    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls=controls,
        shot_count=shot_count,
    )

    qubit_population = []
    experiment_measurements = experiment_results.measurements
    for k, measurement_counts in enumerate(experiment_measurements):
        print(f"control #{k}: {measurement_counts}")
        print(estimate_probability_of_one(measurement_counts))
        qubit_population.append(estimate_probability_of_one(measurement_counts))

    return qubit_population


print(get_qubit_population(30, 1, 1, 0))
