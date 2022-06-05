import numpy as np
import matplotlib.pyplot as plt
from ctrnn import CTRNN
import scipy.signal as signal

# Bokeh Libraries
from.io import output_file, output_notebook, curdoc
from bokeh.layouts import column, row
from bokeh.models import ColumnDataSource, Slider, TextInput, Div, Select, Button
from bokeh.plotting import figure, show, reset_output

# Get a CTRNN solution for specified input and parameters
def generate_solution(inpt, dt, Wi, Wf, tau, b, st):
    N = 1
    pars = {
        "number of neurons": N,
        "number of inputs": N,
        "decay constant": tau,
        "weight matrix": [[Wf]],
        "input weights": [[Wi]],

        "integration mode": "rk4",
        
        "initial state": [st]*N,
        "biases": [b]*N,
    }

    net = CTRNN(**pars)
    def sigmoid(x):
        """
        Sigmoid Function.

        :param x: input vector
        :return: output of 1/(1+exp(-x))
        """
        a = 7.0
        x = np.atleast_1d(x)
        return 1/(1+np.exp(-a*x))
    net.set_activation_function(sigmoid, "sigmoid with param") #TODO: make this function more intuitive and less weird
    #TODO: add activation function selection/input
    net.simulate([inpt], dt)
    return net.output_sequence[0], net.neuron_history[0]


# Input variable widgets
wavetype_map = {
    "Square": signal.square,
    "Sawtooth": signal.sawtooth,
    "Sinusoid": np.sin,
    # key here is that all three share offset, amplitude, phase, freq so can be drawn together
    #TODO: add "Gaussian Pulse" -> requires adding options to menu
    #TODO: add "Constant" -> requires removing options from menu
    #TODO: add duty cycle choice for Saw and Sqr, Sin doesn't need it
}
wave_type = Select(title="Waveform", options=sorted(wavetype_map.keys()), value="Sinusoid")
offset = Slider(title="Offset", value=1.0, start=-1.0, end=2.0, step=0.1)
amplitude = Slider(title="Amplitude", value=1.0, start=-5.0, end=5.0, step=0.1)
phase = Slider(title="Phase", value=0.0, start=0.0, end=2*np.pi, step=0.1)
freq = Slider(title="Frequency", value=1.0, start=0.1, end=5.1, step=0.1)
duty = Slider(title="Duty Cycle", value=0.5, start=0.0, end=1.0, step=0.1)

# Integration parameter widgets
number_of_points = Slider(title="Number Simulation Points", value=10000, start=5000, end=100000, step=5000)
end_time = Slider(title="Simulation end time", value=15.0, start=5.0, end=50.0, step=0.5)

# System parameter widgets
input_weight = Slider(title="Input Weight", value=1.0, start=-1.0, end=1.0, step=0.01)
self_weight = Slider(title="Feedback Weight", value=1.0, start=-1.0, end=1.0, step=0.01)
bias = Slider(title="State Bias", value=0.0, start=-2.0, end=2.0, step=0.1)
initial_state = Slider(title="Initial State", value=0.0, start=-2.0, end=2.0, step=0.1)
decay_constant = Slider(title="Tau", value=0.5, start=0.1, end=1.0, step=0.1)

controls = [wave_type, amplitude, freq, phase, offset, duty, input_weight, self_weight, initial_state, bias, decay_constant, number_of_points, end_time]

# Function callbacks
def update_data():
    # Get the current input selection values
    a = amplitude.value
    b = offset.value
    w = phase.value
    k = freq.value
    input_function = wavetype_map[wave_type.value]

    # Get the current simulation selection values
    t0 = 0.0
    t1 = end_time.value
    N_pts = int(number_of_points.value)

    # Generate the input data
    time = np.linspace(t0, t1, N_pts, endpoint=False)
    if wave_type.value == "Sawtooth" or wave_type.value == "Square":
        inpt = a*input_function(k*time + w, duty.value) + b
    else:
        inpt = a*input_function(k*time + w) + b

    # Generate the CTRNN sim and get output data
    dt = t1/N_pts
    Wf = self_weight.value
    Wi = input_weight.value
    tau = decay_constant.value
    b = bias.value
    st = initial_state.value
    outpt, state = generate_solution(inpt, dt, Wi, Wf, tau, b, st)

    # Send new data to data source
    source.data = dict(time=time, input=inpt, output=outpt, state=state)

# for control in controls:
#     control.on_change('value', update_data) #TODO: add button? lightens computational load so not running an RK4 on every change
button = Button(label="Update Figure", button_type="success")
button.on_click(update_data)

# Set up data
x = np.linspace(0.0, end_time.value, int(number_of_points.value), endpoint=False)
y1 = amplitude.value*wavetype_map[wave_type.value](freq.value*x + phase.value) + offset.value
y2, y3 = generate_solution(y1, end_time.value/int(number_of_points.value), input_weight.value, self_weight.value, decay_constant.value, bias.value, initial_state.value)
source = ColumnDataSource(data=dict(time=x, input=y1, output=y2, state=y3))

# Set up time-domain plot
plot1 = figure(height=500, width=800, title="Time Series",
              tools="crosshair,pan,reset,save,wheel_zoom",
              x_range=[0, end_time.value], y_range=[-1.5, 1.5], name="time_domain_plot")
plot1.line('time', 'input', source=source, line_color='black', line_width=2.0, line_alpha=1.0, legend_label='x(t)')
plot1.line('time', 'output', source=source, line_color='seagreen', line_width=2.0, line_alpha=1.0, legend_label='y(t)')
plot1.line('time', 'state', source=source, line_color='dodgerblue', line_width=2.0, line_alpha=1.0, legend_label='s(t)')

# Set up phase-space plot
plot2 = figure(height=500, width=500, title="Phase Space",
              tools="crosshair,pan,reset,save,wheel_zoom",
              #x_range=[y1.min(), y1.max()], y_range=[y2.min(), y2.max()])
              x_range=[-1.5, 1.5], y_range=[-1.5, 1.5], name="phase_space_plot")
plot2.line('input', 'output', source=source, line_color='seagreen', line_width=1.0, line_alpha=1.0)
plot2.line('input', 'state', source=source, line_color='dodgerblue', line_width=1.0, line_alpha=1.0)

# Add interactivity to the legend
plot1.legend.click_policy = 'hide'

# Set up layouts and add to document
inputs = column(*controls, button)
main_row = row(inputs, plot1, plot2)#, sizing_mode="scale_both")
curdoc().add_root(main_row)
curdoc().title = "CTRNN Plots"

