import os
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
import requests
import json
import re

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {MISTRAL_API_KEY}",
    "Content-Type": "application/json",
}


def get_llm_response(prompt, model="mistral-tiny"):
    """Sends the prompt to the LLM and returns the response."""
    data = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()

        if 'choices' in response_json and response_json['choices']:
          return response_json["choices"][0]["message"]["content"].strip()
        else:
          print(f"Error: No choices in the Mistral API response: {response_json}")
          return None
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Mistral API: {e}")
        return None


def parse_poly(poly_str):
    """Parses a polynomial string and returns coefficients."""
    poly_str = poly_str.replace(" ", "")
    matches = re.findall(r"([-+]?\d*\.?\d*)(x(?:\^(\d+))?)?", poly_str)
    coefs = {}

    for match in matches:
        coef_str, _, power_str = match
        if coef_str in ["-", "+", ""]:
            coef = 1 if coef_str != "-" else -1
        else:
            coef = float(coef_str)
        power = int(power_str) if power_str else 1 if _ else 0
        coefs[power] = coef

    # Fill missing powers with 0
    max_power = max(coefs.keys(), default=0)
    all_coefs = [coefs.get(i, 0) for i in range(max_power, -1, -1)]
    return all_coefs

def plot_graph(function_name, params):
    """Plots the graph of the specified function."""
    x_min, x_max = params["x_min"], params["x_max"]
    x = np.linspace(x_min, x_max, 400)

    if function_name in plot_functions:
        y = plot_functions[function_name](x, params)
        plt.plot(x, y)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Plot of {function_name}")
        plt.grid(True)
        plt.show()
    else:
        print("Function not recognized")
        return


def parse_llm_output(llm_output):
    """Parses the LLM output into function name and params."""
    try:
        parts = llm_output.strip().split(",")
        function_name = parts[0]
        if function_name == "poly":
            coeffs = parse_poly(parts[1])
            x_min, x_max = float(parts[2]), float(parts[3])
            return function_name, {"coeffs": coeffs, "x_min": x_min, "x_max": x_max}
        elif function_name in ["sin", "cos"]:
            k, x_min, x_max = float(parts[1]), float(parts[2]), float(parts[3])
            return function_name, {"k": k, "x_min": x_min, "x_max": x_max}
        else:
            x_min, x_max = float(parts[1]), float(parts[2])
            return function_name, {"x_min": x_min, "x_max": x_max}
    except (ValueError, IndexError) as e:
        print(f"Error parsing LLM output: {e}")
        return None, None



def main():
    """Main function to run the plotting application."""
    print("Welcome to the Plotter App!")

    while True:
        user_input = input("Enter your plot request: ")
        prompt = f"""
        You are a function and interval extractor. Extract function name and interval for the requested plot from the user request.
        Available functions are: y=x, y=x^2, y=sin(k*x) and y=cos(k*x), and any polynomial up to power 4.
        Return answer as comma separated string of function_name, parameters. For the polynomial function, return poly,coefficients as string,x_min,x_max; for sin or cos return sin or cos,k,x_min,x_max;  for the rest return function_name,x_min,x_max.
        If the user asks to end the session return 'exit'.
        User request: {user_input}
        """
        llm_output = get_llm_response(prompt)

        if llm_output is None:
            continue

        if llm_output.strip() == "exit":
            print("Goodbye!")
            break
        if "," not in llm_output:
          print(f"Error processing the request, incorrect response from LLM: {llm_output}")
          continue
        function_name, params = parse_llm_output(llm_output)
        if function_name:
          plot_graph(function_name, params)

plot_functions = {
    "x": lambda x, params: x,
    "x^2": lambda x, params: x**2,
    "sin": lambda x, params: np.sin(params["k"] * x),
    "cos": lambda x, params: np.cos(params["k"] * x),
    "poly": lambda x, params: np.polyval(params["coeffs"], x),
}


if __name__ == "__main__":
    main()