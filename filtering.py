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

def filter_values(alpha_1_values=np.array([0.5]*50)):
    # Define standard matrices
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Define physical constraints
    alpha_max = 2 * np.pi * 8.5e6  # Hz
    nu = 2 * np.pi * 6e6  # Hz
    sinc_cutoff_frequency = 2 * np.pi * 48e6  # Hz
    segment_count = 50
    duration = 250e-9  # s

    # Create graph object
    with qctrl.create_graph() as graph:
        # Create alpha_1(t) signal
        # alpha_1_values = qctrl.operations.bounded_optimization_variable(
        #     count=segment_count,
        #     lower_bound=-alpha_max,
        #     upper_bound=alpha_max,
        # )
        #alpha_1_values = np.array([0.5]*50)
        #print(alpha_1_values)
        #print(type(alpha_1_values))
        alpha_1 = qctrl.operations.pwc_signal(
            values=alpha_1_values,
            duration=duration,
            name="alpha_1",
        )
        # Create filtered signal
        alpha_1_filtered = qctrl.operations.convolve_pwc(
            alpha_1,
            qctrl.operations.sinc_integral_function(sinc_cutoff_frequency),
        )

        # Similarly, create filtered alpha_2(t) signal
        alpha_2_values = qctrl.operations.bounded_optimization_variable(
            count=segment_count,
            lower_bound=-alpha_max,
            upper_bound=alpha_max,
        )
        alpha_2 = qctrl.operations.pwc_signal(
            values=alpha_2_values,
            duration=duration,
            name="alpha_2",
        )
        alpha_2_filtered = qctrl.operations.convolve_pwc(
            alpha_2,
            qctrl.operations.sinc_integral_function(sinc_cutoff_frequency),
        )

        # Create drive term (note the use of STF functions instead of PWC functions,
        # because we are dealing with smooth signals instead of piecewise-constant
        # signals).
        drive = qctrl.operations.stf_operator(alpha_1_filtered, sigma_x / 2)
        # Create clock shift term
        shift = qctrl.operations.stf_operator(alpha_2_filtered, sigma_z / 2)

        # Create dephasing noise term
        dephasing = qctrl.operations.constant_stf_operator(sigma_z / duration)

        # Create target
        target_operator = qctrl.operations.target(operator=sigma_x)

        # Create infidelity (note that we pass an array of sample times, which
        # governs the granularity of the integration procedure)
        infidelity = qctrl.operations.infidelity_stf(
            np.linspace(0, duration, 150),
            qctrl.operations.stf_sum([drive, shift]),
            target_operator,
            [dephasing],
            name="infidelity",
        )

        # Sample filtered signals (to output and plot)
        alpha_1_smooth = qctrl.operations.discretize_stf(
            stf=alpha_1_filtered,
            duration=duration,
            segments_count=500,
            name="alpha_1_filtered",
        )
        alpha_2_smooth = qctrl.operations.discretize_stf(
            stf=alpha_2_filtered,
            duration=duration,
            segments_count=500,
            name="alpha_2_filtered",
        )

    # Run the optimization
    optimization_result = qctrl.functions.calculate_optimization(
        cost_node_name="infidelity",
        output_node_names=["alpha_1", "alpha_2", "alpha_1_filtered", "alpha_2_filtered"],
        graph=graph,
    )

    print("Optimized cost:\t", optimization_result.cost)

    # Plot the optimized controls
    plot_controls(
        plt.figure(),
        controls={
            "$\\alpha_1$": optimization_result.output["alpha_1"],
            "$\\alpha_2$": optimization_result.output["alpha_2"],
        },
    )
    plt.suptitle("Unfiltered pulses")

    plot_controls(
        plt.figure(),
        controls={
            "$L(\\alpha_1)$": optimization_result.output["alpha_1_filtered"],
            "$L(\\alpha_2)$": optimization_result.output["alpha_2_filtered"],
        },
    )
    plt.suptitle("Filtered pulses")

    plt.show()
    print(alpha_1_filtered)
    return(alpha_1_filtered)

if __name__ == "__main__":
    alpha_1_values = np.array([0.5, 0.5, 0.5, -0.5, -0.5, -0.5,0.25, 0.25, 0.7,0.7,0.7])
    filter_values(alpha_1_values=alpha_1_values)