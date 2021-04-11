import jsonpickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from qctrlvisualizer import get_qctrl_style, plot_controls
from scipy import interpolate
from scipy.optimize import curve_fit

import os

# Q-CTRL imports
from qctrl import Qctrl

# Starting a session with the API
qctrl = Qctrl(email=os.getenv('EMAIL'), password=os.getenv('PASSWORD'))

# Choose to run experiments or to use saved data
use_saved_data = False

# Plotting parameters
plt.style.use(get_qctrl_style())
prop_cycle = plt.rcParams["axes.prop_cycle"]
colors = prop_cycle.by_key()["color"]
markers = {"x": "x", "y": "s", "z": "o"}
lines = {"x": "--", "y": "-.", "z": "-"}

# Definition of operators and functions
sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex)
sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex)
sigma_y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex)
X90_gate = np.array([[1.0, -1j], [-1j, 1.0]], dtype=np.complex) / np.sqrt(2)
bloch_basis = ["x", "y", "z"]


def save_var(file_name, var):
    # saves a single var to a file using jsonpickle
    f = open(file_name, "w+")
    to_write = jsonpickle.encode(var)
    f.write(to_write)
    f.close()


def load_var(file_name):
    # retuns a var from a json file
    f = open(file_name, "r+")
    encoded = f.read()
    decoded = jsonpickle.decode(encoded)
    f.close()
    return decoded


def fit_function_bounds(x_values, y_values, function, bound_values):
    fitparams, conv = curve_fit(function, x_values, y_values, bounds=bound_values)
    y_fit = function(x_values, *fitparams)
    return fitparams, y_fit


def movingaverage(x, w):
    return np.convolve(x, np.ones(w), "valid") / w


def simulation_coherent(control, time_samples):
    durations = [segment["duration"] for segment in control["I"]]
    I_values = np.array([segment["value"] for segment in control["I"]])
    Q_values = np.array([segment["value"] for segment in control["Q"]])
    duration = sum(durations)

    # Define shift controls
    shift_I = qctrl.types.coherent_simulation.Shift(
        control=[
            qctrl.types.RealSegmentInput(duration=d, value=v)
            for d, v in zip(durations, I_values)
        ],
        operator=sigma_x,
    )
    shift_Q = qctrl.types.coherent_simulation.Shift(
        control=[
            qctrl.types.RealSegmentInput(duration=d, value=v)
            for d, v in zip(durations, Q_values)
        ],
        operator=sigma_y,
    )

    # Define sample times for the output
    sample_times = np.linspace(0, duration, time_samples)

    # Define the target (optional)
    target = qctrl.types.TargetInput(operator=X90_gate)

    # Perform simulation
    simulation_result = qctrl.functions.calculate_coherent_simulation(
        duration=duration,
        sample_times=sample_times,
        shifts=[shift_I, shift_Q],
        initial_state_vector=np.array([1.0, 0.0]),
        target=target,
    )

    # Extract results
    gate_times = np.array([sample.time for sample in simulation_result.samples])
    state_vectors = np.array(
        [sample.state_vector for sample in simulation_result.samples]
    )
    infidelities = np.array([sample.infidelity for sample in simulation_result.samples])

    bloch_vector_components = {
        "x": np.real(
            np.array(
                [
                    np.linalg.multi_dot([np.conj(state), sigma_x, state])
                    for state in state_vectors
                ]
            )
        ),
        "y": np.real(
            np.array(
                [
                    np.linalg.multi_dot([np.conj(state), sigma_y, state])
                    for state in state_vectors
                ]
            )
        ),
        "z": np.real(
            np.array(
                [
                    np.linalg.multi_dot([np.conj(state), sigma_z, state])
                    for state in state_vectors
                ]
            )
        ),
    }

    return infidelities, bloch_vector_components, gate_times


import warnings

warnings.simplefilter("ignore")
use_IBM = False

if use_IBM == True:
    # IBM-Q imports
    import qiskit.pulse as pulse
    import qiskit.pulse.pulse_lib as pulse_lib
    from qiskit import IBMQ
    from qiskit.compiler import assemble
    from qiskit.pulse import Acquire, Play, Schedule
    from qiskit.tools.jupyter import *
    from qiskit.tools.monitor import job_monitor

    # IBM credentials and backend selection
    provider = IBMQ.enable_account(
        "ac80b9064c0b54273c37ba81edd4a569b5a1a82ded80cbbb9964a23aced5ff82a9b0839ef8ca4b97783c340e86fa1af51dac3bae8f5b94ea066ba94f03dc7b21")
    # provider = IBMQ.get_provider(
    #     hub="your hub", group="your group", project="your project"
    # )

    backend = provider.get_backend("ibmq_valencia")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    # Backend properties
    dt = backend_config.dt
    print(f"Hardware sampling time: {dt / 1e-9} ns")

    qubit_freq_est = []
    for qubit in backend_config.meas_map[0]:
        qubit_freq_est.append(backend_defaults.qubit_freq_est[qubit])
        print(f"Qubit [{qubit}] frequency estimate: {qubit_freq_est[qubit] / 1e9} GHz")

# Setting up calibration experiments
qubit = 0
dt = 2 / 9 * 1e-9
num_shots_per_point = 1024
pulse_amp_array = np.linspace(0.05, 0.2, 7)
pulse_times = np.array(
    [4 + np.arange(0, int(3.6 / (amplitude)), 1) for amplitude in pulse_amp_array]
)
pulse_times = pulse_times * 16

if use_saved_data == False:
    """
    backend.properties(refresh=True)
    qubit_frequency_updated = backend.properties().qubit_property(qubit, "frequency")[0]

    meas_map_idx = None
    for i, measure_group in enumerate(backend_config.meas_map):
        if qubit in measure_group:
            meas_map_idx = i
            break
    assert meas_map_idx is not None, f"Couldn't find qubit {qubit} in the meas_map!"

    inst_sched_map = backend_defaults.instruction_schedule_map
    measure_schedule = inst_sched_map.get("measure", qubits=[qubit])
    drive_chan = pulse.DriveChannel(qubit)
    """

    rabi_programs_dic_I = {}
    for idx, pulse_amplitude in enumerate(pulse_amp_array):
        rabi_schedules_I = []
        for duration_pulse in pulse_times[idx]:
            drive_pulse = pulse_lib.gaussian_square(
                duration=duration_pulse,
                sigma=1,
                amp=pulse_amplitude,
                risefall=1,
                name=f"square_pulse_{duration_pulse}",
            )
            """
            schedule = pulse.Schedule(name=str(duration_pulse))
            schedule |= (
                    Play(drive_pulse, pulse.DriveChannel(qubit)) << schedule.duration
            )
            schedule += measure_schedule << schedule.duration
            rabi_schedules_I.append(schedule)
            
        rabi_experiment_program_I = assemble(
            rabi_schedules_I,
            backend=backend,
            meas_level=2,
            meas_return="single",
            shots=num_shots_per_point,
            schedule_los=[{drive_chan: qubit_frequency_updated}]
                         * len(pulse_times[idx]),
        )
        """
        rabi_programs_dic_I[pulse_amplitude] = rabi_experiment_program_I

    # Running calibration experiments
    rabi_calibration_exp_I = []
    rabi_oscillations_results = []
    for idx, pulse_amplitude in enumerate(pulse_amp_array):
        job = backend.run(rabi_programs_dic_I[pulse_amplitude])
        job_monitor(job)
        rabi_results = job.result(timeout=120)
        rabi_values = []
        time_array = pulse_times[idx] * dt
        for time_idx in pulse_times[idx]:
            counts = rabi_results.get_counts(str(time_idx))
            excited_pop = 0
            for bits, count in counts.items():
                excited_pop += count if bits[::-1][qubit] == "1" else 0
            rabi_values.append(excited_pop / num_shots_per_point)

        rabi_oscillations_results.append(rabi_values)
        fit_parameters, y_fit = fit_function_bounds(
            time_array,
            rabi_values,
            lambda x, A, rabi_freq, phi: A
                                         * np.cos(2 * np.pi * rabi_freq * x + phi) ** 2,
            (
                [0.8, np.abs(pulse_amplitude * 8 * 1e7), -4],
                [1, np.abs(pulse_amplitude * 11 * 1e7), 4],
            ),
        )

        rabi_calibration_exp_I.append(fit_parameters[1])

    save_var(
        "resources/superconducting-qubits-pulse-calibration/rabi_calibration_Valencia_qubit_0",
        rabi_calibration_exp_I,
    )
    save_var(
        "resources/superconducting-qubits-pulse-calibration/fit_parameters",
        fit_parameters,
    )
    save_var(
        "resources/superconducting-qubits-pulse-calibration/rabi_values", rabi_values
    )
else:
    rabi_calibration_exp_I = load_var(
        "resources/superconducting-qubits-pulse-calibration/rabi_calibration_Valencia_qubit_0"
    )
    fit_parameters = load_var(
        "resources/superconducting-qubits-pulse-calibration/fit_parameters"
    )
    rabi_values = load_var(
        "resources/superconducting-qubits-pulse-calibration/rabi_values"
    )

time_array = pulse_times[-1] * dt
print("Drive amplitude:", pulse_amp_array[-1])
print("Fitted Rabi frequency [Hz]:", fit_parameters[1])
plt.title("Exemplary Rabi oscillation data with fitting", fontsize=16, y=1.05)
plt.xlabel("Time [s]", fontsize=14)
plt.ylabel("Measured signal [a.u.]", fontsize=14)
plt.scatter(time_array, np.real(rabi_values), color="black")
plt.xlim(0, time_array[-1])
plt.ylim(0, 1)
plot_times = np.linspace(0, time_array[-1], 100)
plt.plot(
    plot_times,
    fit_parameters[0]
    * np.cos(2 * np.pi * fit_parameters[1] * plot_times + fit_parameters[2]) ** 2,
    color="red",
)
plt.show()

amplitude_interpolated_list = np.linspace(-0.2, 0.2, 100)
pulse_amp_array = np.concatenate((-pulse_amp_array[::-1], pulse_amp_array))
rabi_calibration_exp_I = np.concatenate(
    (-np.asarray(rabi_calibration_exp_I[::-1]), np.asarray(rabi_calibration_exp_I))
)
f_amp_to_rabi = interpolate.interp1d(pulse_amp_array, rabi_calibration_exp_I)
rabi_interpolated_exp_I = f_amp_to_rabi(amplitude_interpolated_list)

f_rabi_to_amp = interpolate.interp1d(
    rabi_interpolated_exp_I, amplitude_interpolated_list
)

plt.title("IBMQ Valencia: Rabi rates calibration", fontsize=16, y=1.1)
plt.xlabel("Hardware input amplitude", fontsize=14)
plt.ylabel("Rabi rate [Hz]", fontsize=14)
plt.scatter(pulse_amp_array, rabi_calibration_exp_I)
plt.tick_params(axis="both", which="major", labelsize=14)
plt.plot(amplitude_interpolated_list, rabi_interpolated_exp_I)
plt.axvline(0, color="black", linestyle="dashed")
plt.axhline(0, color="black", linestyle="dashed")
plt.show()
