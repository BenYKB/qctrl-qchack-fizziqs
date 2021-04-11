import matplotlib.pyplot as plt
import numpy as np
from qctrlopencontrols import new_corpse_control, new_primitive_control
from qctrlvisualizer import plot_controls
import os
from dotenv import load_dotenv
from qsim import *
from scipy_filter import filtering
from taper_window import planck_taper_window
from cost import signal_concatenate, generate_patterns, cost_determination, complex_parameters
from initial_pulse import get_initial_pulse

# get_initial_pulse()

from qctrl import Qctrl

load_dotenv()
# Starting a session with the API
qctrl = Qctrl(email=os.getenv('EMAIL'), password=os.getenv('PASS'))

def simulate_more_realistic_qubit(
    duration=1, values=np.array([np.pi]), shots=1024, repetitions=1
):

    # 1. Limits for drive amplitudes
    assert np.amax(values) <= 1.0
    assert np.amin(values) >= -1.0
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

        # Apply 4. bandwidth limits
        drive_unfiltered = qctrl.operations.pwc_signal(duration=duration, values=values)
        drive_filtered = qctrl.operations.convolve_pwc(
            pwc=drive_unfiltered,
            kernel_integral=qctrl.operations.sinc_integral_function(cut_off_frequency),
        )
        drive = qctrl.operations.discretize_stf(
            drive_filtered, duration=duration, segments_count=resample_segment_count
        )

        # Construct microwave drive
        drive_term = qctrl.operations.pwc_operator_hermitian_part(
            qctrl.operations.pwc_operator(signal=drive, operator=b)
        )

        # Construct 2. dephasing term.
        dephasing_term = qctrl.operations.constant_pwc_operator(
            operator=dephasing_error * n,
            duration=duration,
        )

        # Construct Hamiltonian.
        hamiltonian = qctrl.operations.pwc_sum(
            [
                drive_term,
                dephasing_term,
            ]
        )

        # Solve Schrodinger's equation and get total unitary at the end
        unitary = qctrl.operations.time_evolution_operators_pwc(
            hamiltonian=hamiltonian,
            sample_times=np.array([duration]),
        )[-1]
        unitary.name = "unitary"

        # Repeat final unitary
        repeated_unitary = np.eye(2)
        for _ in range(repetitions):
            repeated_unitary = repeated_unitary @ unitary
        repeated_unitary.name = "repeated_unitary"

        # Calculate final state.
        state = repeated_unitary @ initial_state

        # Calculate final populations.
        populations = qctrl.operations.abs(state[:, 0]) ** 2
        # Normalize populations
        norm = qctrl.operations.sum(populations)
        populations = populations / norm
        populations.name = "populations"

    # Evaluate graph.
    result = qctrl.functions.calculate_graph(
        graph=graph,
        output_node_names=["unitary", "repeated_unitary", "populations"],
    )

    # Extract outputs.
    unitary = result.output["unitary"]["value"]
    repeated_unitary = result.output["repeated_unitary"]["value"]
    populations = result.output["populations"]["value"]

    # Sample projective measurements.
    true_measurements = np.random.choice(2, size=shots, p=populations)
    measurements = np.array(
        [np.random.choice(2, p=confusion_matrix[m]) for m in true_measurements]
    )

    results = {"unitary": unitary, "measurements": measurements}

    return results



# duration = 10
# values = np.array([-1, 3, 2, 3, -2, -1])


# def get_pulse_plot_dict(name="default", duration=1, values=np.array([1.0])):
#     segments = len(values)
#     segment_durations = duration / segments
#     pulse_plot_dict = {
#         name: [{"duration": segment_durations, "value": v} for v in values]
#     }
#     return pulse_plot_dict


# example_pulse = get_pulse_plot_dict(name="$\Omega$", duration=duration, values=values)



# duration = 10
# values = np.array([-1,5,6,4,-2])

# example_pulse = get_pulse_plot_dict(name='$\Omega$', duration=duration, values=values)



# fig = plt.figure()
# plot_controls(fig, example_pulse, polar=False)
# plt.show()

t_min = 10
t_max = 40


def duration_from_T(T):
    return (t_min + t_max)/2 + (t_max - t_min) / 2 * T
N = 18
test_point_count = 2
segment_count = 2*(2*N)+1

sigma = 0.01




# Define the number of test points obtained per run.


def cost_function(results, expected_result):
    return np.count_nonzero(results==expected_result)/results.size

def run_experiments(parameters_set):
    shot_count = 1024

    rets = []

    set_counter = 0

    for parameter in parameter_set:
        if set_counter == 0:
            set_counter +=1
            print(f'first params {parameter}')

        parameter = complex_parameters(parameter)
        #print(parameter)

        T = np.real(parameter[0])

        gate_N = parameter[1:1+N]

        gate_H = parameter[1+N:]

        duration = duration_from_T(T)

        #print(f'current duration: {duration}')

        #filter
        #print('applying filter')
        N_filtered = filtering(gate_N)
        H_filtered = filtering(gate_H)
        #print(N_filtered)

        #window

        N_windowed = planck_taper_window(N_filtered)
        H_windowed = planck_taper_window(H_filtered)

        abs_N = np.abs(N_windowed)
        abs_H = np.abs(H_windowed)

        for i in range(abs_N.size):
            if abs_N[i] > 1:
                N_windowed[i] = N_windowed[i]/(abs_N[i] +.00001)

        for i in range(abs_H.size):
            if abs_H[i] > 1:
                H_windowed[i] = H_windowed[i]/(abs_H[i]+ .00001)

        #print(f'done filtering + taper:  {N_windowed}')

        #determine concatenation pattern
        
        n = np.random.randint(2,7)

        one_not = np.array([0],dtype=int)
        one_h = np.array([1],dtype=int)
        nots, hs, randpattern = generate_patterns(n)

        patterns = [one_not, one_h, nots, hs, randpattern]

        print(f'patterns are {patterns}')

        controls = []
        for pattern in patterns:
            sig = signal_concatenate(N_windowed, H_windowed, pattern)
            controls.append({"duration":duration*pattern.size, "values": sig})
            #print(f'added control of duration {duration}')
            #print(f'added control vals of lenght {sig.size}')
        # Obtain the results of the experiment.
        #print(f"sending shot")
        experiment_results = qctrl.functions.calculate_qchack_measurements(
            controls=controls,
            shot_count=shot_count,
        )
        #print("done shot")

        measurements = experiment_results.measurements

        costs = []
        for i in range(5):
            costs.append(cost_determination(measurements[i], patterns[i]))
        

        rets.append(5*np.mean(np.abs(costs)))

    return rets


# Define parameters as a set of controls with piecewise constant segments.
# parameter_set = (
#     1.0
#     * (np.linspace(-1, 1, test_point_count)[:, None])
#     * np.random.rand(test_point_count, segment_count)
# )


# Guess from Rabi Rate 
initial_guess = np.concatenate((
    np.array([-.8]),
    0.7*np.ones(N),
    np.zeros(3),0.7*np.ones(6),np.zeros(9),
    np.zeros(N),
    -0.7 *np.ones(3), np.zeros(6), np.zeros(9))
)

random_guess = (np.random.rand(segment_count)-0.5)*1.7


parameter_set = np.stack((initial_guess, random_guess))



print(f'initial parameter set{parameter_set}')


bound = qctrl.types.closed_loop_optimization_step.BoxConstraint(
    lower_bound=-1,
    upper_bound=1,
)

initializer = qctrl.types.closed_loop_optimization_step.GaussianProcessInitializer(
    bounds=[bound] * segment_count,
    rng_seed=0,
)
optimizer = qctrl.types.closed_loop_optimization_step.Optimizer(
    gaussian_process_initializer=initializer,
)


experiment_results = run_experiments(parameter_set)


# def get_exp_results(run_parameters):
#     return simulate_more_realistic_qubit

#xperimental_results = get_exp_results(run_parameters)

best_cost, best_controls = min(zip(experiment_results, parameter_set), key=lambda params:params[0])


optimization_count = 0

# Run the optimization loop until the cost (infidelity) is sufficiently small.
while best_cost > 3 * sigma:
    # Print the current best cost.
    optimization_steps = (
        "optimization step" if optimization_count == 1 else "optimization steps"
    )
    print(
        f"Best infidelity after {optimization_count} BOULDER OPAL {optimization_steps}: {best_cost}"
    )

    # Organize the experiment results into the proper input format.
    results = [
        qctrl.types.closed_loop_optimization_step.CostFunctionResult(
            parameters=list(parameters),
            cost=cost,
            cost_uncertainty=sigma,
        )
        for parameters, cost in zip(parameter_set, experiment_results)
    ]

    # Call the automated closed-loop optimizer and obtain the next set of test points.
    optimization_result = qctrl.functions.calculate_closed_loop_optimization_step(
        optimizer=optimizer,
        results=results,
        test_point_count=test_point_count,
    )
    optimization_count += 1

    # Organize the data returned by the automated closed-loop optimizer.
    parameter_set = np.array(
        [test_point.parameters for test_point in optimization_result.test_points]
    )
    optimizer = qctrl.types.closed_loop_optimization_step.Optimizer(
        state=optimization_result.state
    )

    # Obtain experiment results that the automated closed-loop optimizer requested.
    experiment_results = run_experiments(parameter_set)

    # Record the best results after this round of experiments.
    cost, controls = min(
        zip(experiment_results, parameter_set), key=lambda params: params[0]
    )
    if cost < best_cost:
        best_cost = cost
        best_controls = controls


    if optimization_count % 2 == 0:
        print(f'current best {best_controls} at best cost {best_cost}')


# Print final best cost.
print(f"Infidelity: {best_cost}")

# Plot controls that correspond to the best cost.
plot_controls(
    figure=plt.figure(),
    controls={
        r"$\Omega(t)$": [
            {"duration": duration / len(best_controls), "value": value}
            for value in best_controls
        ]
    },
)




