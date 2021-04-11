import jsonpickle
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from qctrlvisualizer import get_qctrl_style, plot_controls
from scipy import interpolate
from scipy.optimize import curve_fit
import numpy as np

# Q-CTRL imports
from qctrl import Qctrl

# Starting a session with the API
import os
from dotenv import load_dotenv

load_dotenv()
em = os.getenv('EMAIL')
pw = os.getenv('PASS')
qctrl = Qctrl(email=em, password=pw)

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
    #import qiskit.pulse.pulse_library as pulse_lib (I don't know if this is this should be changed to the one below?)
    import qiskit.pulse.library as pulse_lib
    from qiskit import IBMQ
    from qiskit.compiler import assemble
    from qiskit.pulse import Acquire, Play, Schedule
    from qiskit.tools.jupyter import *
    from qiskit.tools.monitor import job_monitor

    # IBM credentials and backend selection
    IBMQ.enable_account("API TOKEN")
    provider = IBMQ.get_provider(
        hub="ibm-q", group="open", project="main"
    )
    backend = provider.get_backend("BACKEND")
    backend_defaults = backend.defaults()
    backend_config = backend.configuration()

    # Backend properties
    dt = backend_config.dt
    print(f"Hardware sampling time: {dt/1e-9} ns")

    qubit_freq_est = []
    for qubit in backend_config.meas_map[0]:
        qubit_freq_est.append(backend_defaults.qubit_freq_est[qubit])
        print(f"Qubit [{qubit}] frequency estimate: {qubit_freq_est[qubit]/1e9} GHz")

amplitudes = np.linspace(0.2, 0.7, 11)
frequencies = []
times = np.arange(100, 200, 2)

def cosine_func(x, A, rabi_freq, phi):
    return A * np.cos(2 * np.pi * rabi_freq * x + phi) ** 2

for amplitude in amplitudes:
    controls = []
    signals = []

    for dt in times:
        controls.append({"duration": int(dt), "values": np.array([amplitude])})

    experiment_results = qctrl.functions.calculate_qchack_measurements(
        controls = controls,
        shot_count = 1024
    )

    measurements = experiment_results.measurements

    for measurement_counts in measurements:
        signals.append(np.mean(measurement_counts))

    plt.plot(times, np.array(signals), "o")
    
    fit_parameters, y_fit = fit_function_bounds(
        times,
        np.array(signals),
        cosine_func,
        (
            [0.8, np.abs(amplitude * 8 * 1e7), -4],
            [1, np.abs(amplitude * 11 * 1e7), 4],
        ),
    )

    plt.plot(times, cosine_func(times, fit_parameters[0], fit_parameters[1], fit_parameters[2]), '-')

    plt.savefig("graphs/%lf.png"%(amplitude))
    plt.close()

    frequencies.append(fit_parameters[1])

plt.plot(amplitudes, np.array(frequencies), 'o')
plt.savefig("graphs/amplitude-frequencies.png")
plt.close()