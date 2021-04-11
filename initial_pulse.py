import matplotlib.pyplot as plt
import numpy as np

from qctrlvisualizer import plot_controls

def get_pulse_plot_dict(name="default", duration=1.0, values=np.array([1.0])):
    segments = len(values)
    segment_durations = duration / segments
    pulse_plot_dict = {
        name: [{"duration": segment_durations, "value": v} for v in values]
    }
    return pulse_plot_dict

def get_initial_pulse():
    m = 0.2908882086657218
    b = 0.02521031141769572

    max_rabi_rate = m * 0.7 + b  # rad / ns

    not_duration = np.pi / max_rabi_rate  # ns
    not_values = np.array([max_rabi_rate])
    h_duration = 3 * np.pi / (2 * max_rabi_rate)  # ns
    h_values = np.array([-1j * max_rabi_rate, max_rabi_rate, max_rabi_rate])

    not_pulse = get_pulse_plot_dict(
        name="$\Omega_{NOT}$", duration=not_duration, values=not_values
    )
    h_pulse = get_pulse_plot_dict(name="$\Omega_{H}$", duration=h_duration, values=h_values)
    both_pulses = {**not_pulse, **h_pulse}

    print(both_pulses)

    fig = plt.figure()
    plot_controls(fig, both_pulses, polar=False)
    plt.show()