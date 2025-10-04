import subprocess
import json
import numpy as np
from itertools import product
import random
import math
import matplotlib.pyplot as plt
import time
from plot_3d_segments import generate_plot

# Select a fixed formula for testing, or set to None to generate new ones indefinitely
FIXED_FORMULA = """
def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
    dx = params[0] * (prev_dy - prev_dx)
    dy = prev_dx * (params[1] - prev_dz) - prev_dy
    dz = prev_dx * prev_dy - params[2] * prev_dz
    return np.array([dx, dy, dz])"""
USE_FIXED_FORMULA = True  # Set to True to use the fixed formula
ATTRACTOR_STEPS = 1000  # Number of steps to simulate for each parameter set
ATTRACTOR_DT = 0.01  # Time step for the simulation
ATTRACTOR_SPACE_BAILOUT = 1e4  # Bailout threshold for attractor values

INITIAL_POS = [1.0, 1.0, 1.0]  # Initial position for the attractor
PARAM_MAX_ABS = 30  # Add +- this value around the center
PARAM_COUNT = 3  # How many values per axis
PARAM_AMT = 3  # Number of params
LLM_MODEL = "gemma3:12b"

# ---- PROMPT TEMPLATE ----
prompt_template = """
You are an expert in nonlinear dynamical systems.
Generate the formula definition string for a 3D dynamical system.
The variables are dx, dy, dz, the previous state variables are prev_dx, prev_dy, prev_dz, and the parameters are in the list params[0], params[1], params[2], etc.

Don't be too creative, just try mixing parameters, adding new terms, or changing the order of the params, you can add up to 10 more parameters if needed.
You can use basic arithmetic operations (+, -, *, /) and simple functions like math.sin, math.cos, math.exp, etc.
Do NOT include any explanations or imports — just the code.

Here are is a valid example:

dx = params[0] * (prev_dy - prev_dx)
dy = prev_dx * (params[1] - prev_dz) - prev_dy
dz = prev_dx * prev_dy - params[2] * prev_dz
"""


def generate_formula_from_ollama(model, prompt):
    """
    Calls a local Ollama model and returns the generated text.
    """
    result = subprocess.run(
        ["ollama", "run", model], input=prompt.encode(), stdout=subprocess.PIPE
    )
    return result.stdout.decode()


# ---- DYNAMICAL SYSTEM EVALUATION ----
def test_generate(params):
    state = np.array(INITIAL_POS)  # initial conditions
    points = []
    for i in range(ATTRACTOR_STEPS):
        state = state + formula(state, params) * ATTRACTOR_DT

        # Bail out if values explode
        if np.any(np.abs(state) > ATTRACTOR_SPACE_BAILOUT):
            raise ValueError("Diverged")

        points.append(state.copy())
    return np.array(points)


# ---- MAIN LOOP ----
while True:
    if USE_FIXED_FORMULA:
        # Use the fixed formula for testing
        formula_to_test = FIXED_FORMULA
        print("Using fixed formula:\n", formula_to_test)
    else:
        model_output = generate_formula_from_ollama(LLM_MODEL, prompt_template).strip()
        idented_output = "\n".join("    " + line for line in model_output.splitlines())

        formula_to_test = f"""
def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
{idented_output}
    return np.array([dx, dy, dz])"""

        print("Generated formula:\n", formula_to_test)

    # Test defining the formula
    try:
        exec(formula_to_test, globals())
    except Exception as e:
        print("Error executing formula:", e)
        continue

    # Param space exploration
    params_space = [
        list(p)
        for p in product(
            np.linspace(-PARAM_MAX_ABS, PARAM_MAX_ABS, PARAM_COUNT), repeat=PARAM_AMT
        )
    ]

    # params_space = [
    #     [10.0 + -10, 28.0 + -10, -10 + 8.0 / 3.0],
    #     [10.0, 28.0, 8.0 / 3.0],
    #     [10.0 + 10, 28.0 + 10, 10 + 8.0 / 3.0],
    # ]

    attractor_lines = []
    for params_list in params_space:
        # Generate random parameters
        print(f"Iterating: {params_list}")

        try:
            points = test_generate(params_list)  # Try generating the attractor
            # Shift by all three params to separate in space
            attractor_lines.append(points + np.array(params_list) * 10)
            # attractor_lines.append(points) # Add without shift

        except Exception as e:
            print("Error running formula:", e)

    # Plot the points using the imported function
    generate_plot(attractor_lines)

    if USE_FIXED_FORMULA:
        break  # Exit after one iteration if using fixed formula

    time.sleep(60)
