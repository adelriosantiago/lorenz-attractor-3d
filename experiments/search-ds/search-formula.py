import subprocess
import json
import numpy as np
import random
import matplotlib.pyplot as plt

# ---- CONFIG ----
MODEL = "gemma3:12b"
N_FORMULAS = 1

# ---- PROMPT TEMPLATE ----
prompt_template = """
You are an expert in nonlinear dynamical systems.
Generate the formula definition string for a 3D dynamical system.
The variables are dx, dy, dz, the previous state variables are prev_dx, prev_dy, prev_dz, and the parameters are in the list params[0], params[1], params[2], etc.

Don't be too creative, just try mixing parameters, adding new terms, or changing the order of the params, you can add up to 10 more parameters if needed.
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
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE
    )
    return result.stdout.decode()

# ---- DYNAMICAL SYSTEM EVALUATION ----
def tryGenerate(params, steps=3000, dt=0.01):
    state = np.array([1.0, 1.0, 1.0])  # initial conditions
    points = []
    for i in range(steps):
        state = state + formula(state, params) * dt
        if i > steps // 10:  # skip transient
            points.append(state.copy())
    return np.array(points)

# ---- MAIN LOOP ----
for i in range(N_FORMULAS):
    print(f"\n=== Generating formula {i+1} ===")

    model_output = generate_formula_from_ollama(MODEL, prompt_template).strip()
    idented_output = "\n".join("    " + line for line in model_output.splitlines())
    print("Raw model output:", model_output)

    new_formula_str = f"""
def formula(state, params):
    prev_dx, prev_dy, prev_dz = state
{idented_output}
    return np.array([dx, dy, dz])
    """

    print("Generated formula:\n", new_formula_str)

    # Exec the formula safely
    try:
        exec(new_formula_str, globals())
    except Exception as e:
        print("Error executing formula:", e)
        continue

    # Try generating the attractor
    try:
        points = tryGenerate([random.uniform(-100, 100) for _ in range(30)])

        if len(points) < 10:
            print("Too few points produced.")
            continue

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(points[:,0], points[:,1], points[:,2])
        ax.set_title(f"Formula {i+1}")
        plt.show()

    except Exception as e:
        print("Error running formula:", e)
