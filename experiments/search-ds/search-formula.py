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
ATTRACTOR_SPACE_BAILOUT = 50  # Bailout threshold for attractor values
PARAM_MAX_ABS = 10  # Maximum absolute value for parameters
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
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
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
    param_space = [
        list(p)
        for p in product(
            np.linspace(-PARAM_MAX_ABS, PARAM_MAX_ABS, PARAM_COUNT), repeat=PARAM_AMT
        )
    ]

    attractor_space = []
    for param_list in param_space:
        # Generate random parameters
        print(f"Iterating: {param_list}")

        try:
            points = test_generate(param_list)  # Try generating the attractor

            # Push points to attractor space
            attractor_space.append(points)

        except Exception as e:
            print("Error running formula:", e)

    # Plot the points using the imported function
    generate_plot(
        attractor_space, ATTRACTOR_SPACE_BAILOUT // 2, filename=f"3d_plot.png"
    )

    if USE_FIXED_FORMULA:
        break  # Exit after one iteration if using fixed formula

    time.sleep(60)
