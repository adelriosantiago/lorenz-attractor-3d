import subprocess
import json
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

# Select a fixed formula for testing, or set to None to generate new ones indefinitely
FIXED_FORMULA = """
def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
    dx = params[0] * prev_dy + params[1] * prev_dz
    dy = -params[2] * prev_dx + params[3] * prev_dy
    dz = params[4] * prev_dx * prev_dy - params[5] * prev_dz + params[6] * prev_dx
    return np.array([dx, dy, dz])"""
USE_FIXED_FORMULA = True  # Set to True to use the fixed formula
PARAM_ITERATIONS = 300  # Number of parameters to generate
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


# ---- FUNCTION TO CALL OLLAMA LOCALLY ----
def generate_formula_from_ollama(model, prompt):
    """
    Calls a local Ollama model and returns the generated text.
    """
    result = subprocess.run(
        ["ollama", "run", model], input=prompt.encode(), stdout=subprocess.PIPE
    )
    return result.stdout.decode()


# ---- DYNAMICAL SYSTEM EVALUATION ----
def test_generate(params, steps=3000, dt=0.01):
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + formula(state, params) * dt

        # Bail out if values explode
        if np.any(np.abs(state) > 1e6):
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
        print(f"\n=== Generating formula {i+1} ===")

        model_output = generate_formula_from_ollama(LLM_MODEL, prompt_template).strip()
        idented_output = "\n".join("    " + line for line in model_output.splitlines())
        print("Raw model output:", model_output)

        formula_to_test = f"""
def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
{idented_output}
    return np.array([dx, dy, dz])"""

        print("Generated formula:\n", formula_to_test)

    # Exec the formula safely
    try:
        exec(formula_to_test, globals())
    except Exception as e:
        print("Error executing formula:", e)
        continue

    for i in range(PARAM_ITERATIONS):
        # Generate random parameters
        params = [random.uniform(-100, 100) for _ in range(30)]
        print(f"Iterating: {i}")

        # Try generating the attractor
        try:
            random_params = [random.uniform(-100, 100) for _ in range(30)]
            points = test_generate(random_params)

            if len(points) < 1000:
                print("Too few points produced.")
                continue

            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(points[:, 0], points[:, 1], points[:, 2])
            ax.set_title(f"Formula {i+1}")
            plt.show()

        except Exception as e:
            print("Error running formula:", e)

    if USE_FIXED_FORMULA:
        break  # Exit after one iteration if using fixed formula

    time.sleep(60)
