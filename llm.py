import os
from typing import Any

from mistralai import Mistral
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import re

load_dotenv()

api_key = os.environ["MISTRAL_API_KEY"]
model = "mistral-large-latest"

client = Mistral(api_key=api_key)


def get_llm_response(user_message):
    prompt = f"""
    You are a helpful assistant that extracts function and interval for graph plotting.
    You can plot:
    - Polynomials of power 1 to 4 (e.g., "x^3 - 2x + 1").
    - Scaled sine and cosine functions (e.g., "sin(3x)", "cos(0.5x)").

    For polynomials, extract the coefficients as a list, starting from the highest power.
    For scaled trig functions, extract the function name (sin or cos) and the scale factor 'k'.

    Allowed interval format is "from x_min to x_max" or "between x_min and x_max".

    If the user asks to plot a graph, extract:
    - function_type (polynomial, sin, cos)
    - function_parameters (list of coefficients for polynomial, scale factor 'k' for trig)
    - x_min (numeric value from the interval)
    - x_max (numeric value from the interval)

    **IMPORTANT INSTRUCTIONS: Return ONLY the result in the format: "function_type,function_parameters,x_min,x_max".**
    **For polynomials, function_parameters should be a comma-separated list of coefficients INSIDE SQUARE BRACKETS (e.g., [1,-2,1]).**
    **For sin(kx) and cos(kx), function_parameters should be just the value of k (a single number).**
    **Do not include any extra text, explanations, or preambles. Just the comma-separated output.**

    If the user wants to end the session, return "exit".
    If you cannot understand the request, return "unknown".

    Example 1:
    User: Plot x^2 - x + 2 from -2 to 3
    Assistant: polynomial,[1,-1,2],-2,3

    Example 2:
    User: Graph of sin(2x) between 0 and 2pi
    Assistant: sin,2,0,6.28

    Example 3:
    User: plot cos(0.5x) from -pi to pi
    Assistant: cos,0.5,-3.14,3.14

    Example 4:
    User: Plot x^4 + 1
    Assistant: polynomial,[1,0,0,0,1],-10,10

    Example 5:
    User: bye bye
    Assistant: exit

    Example 6:
    User: draw something else
    Assistant: unknown

    User message: {user_message}
    Assistant:
    """
    try:
        chat_response = client.chat.complete(
            model=model,
            messages=[
                {"role": "system", "content": "You are a graph plotting assistant."},
                {"role": "user", "content": prompt}
            ],
        )
        return chat_response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error communicating with Mistral LLM: {e}")
        return "error"

def parse_llm_output(llm_response):
    # I'm very confused right now
    if llm_response == "exit":
        return "exit", None, None, None, None
    if llm_response == "unknown" or llm_response == "error":
        return "unknown", None, None, None, None

    try:
        parts = llm_response.split(',', 1)  # Split only at the first comma
        if len(parts) < 2:
            print("Warning: Incomplete response format (missing comma after function type).")
            return "unknown", None, None, None, None

        function_type = parts[0].strip()
        rest_of_response = parts[1]

        print(f"Debug: function_type = '{function_type}'")
        print(f"Debug: rest_of_response = '{rest_of_response}'")

        if function_type == 'polynomial':
            start_bracket = rest_of_response.find('[')
            end_bracket = rest_of_response.find(']')
            if start_bracket != -1 and end_bracket != -1 and start_bracket < end_bracket:
                function_params_str = rest_of_response[start_bracket+1:end_bracket].strip()
                interval_start_index = end_bracket + 1 + 1  # +1 to skip ']' and +1 to skip next comma (if any)
                interval_part = rest_of_response[interval_start_index:].strip() if interval_start_index < len(rest_of_response) else ""

                print(f"Debug: polynomial function_params_str = '{function_params_str}'")
                print(f"Debug: polynomial interval_part = '{interval_part}'")

                function_params = []
                try:
                    coeff_str_list = function_params_str.split(',')
                    function_params = [float(coeff.strip()) for coeff in coeff_str_list]
                except:
                    print("Warning: Could not parse polynomial coefficients correctly.")
                    return "unknown", None, None, None, None
            else:
                print("Warning: Malformed polynomial parameters (missing brackets).")
                return "unknown", None, None, None, None

        elif function_type in ['sin', 'cos']:
            parts_after_type = rest_of_response.split(',', 1) # Split rest by comma again
            if len(parts_after_type) < 2:
                 print("Warning: Incomplete response format for sin/cos (missing comma after function param).")
                 return "unknown", None, None, None, None

            function_params_str = parts_after_type[0].strip()
            interval_part = parts_after_type[1].strip()

            print(f"Debug: trig function_params_str = '{function_params_str}'")
            print(f"Debug: trig interval_part = '{interval_part}'")

            try:
                function_params = float(function_params_str)  # k value
            except:
                print("Warning: Could not parse scale factor correctly.")
                return "unknown", None, None, None, None
        else:
            print("Warning: Unknown function type from LLM: " + function_type)
            return "unknown", None, None, None, None

        # Parse interval part (common for all function types)
        interval_match = re.search(r"\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*", interval_part)
        if interval_match:
            x_min = float(interval_match.group(1))
            x_max = float(interval_match.group(2))
            print(f"Debug: x_min = {x_min}, x_max = {x_max}")  # Debug print for parsed x_min and x_max
        else:
            print("Warning: Could not parse interval correctly from LLM response.")
            return "unknown", None, None, None, None

        return "plot", function_type, function_params, x_min, x_max

    except Exception as e:
        print(f"Parsing error: {e}")
        return "unknown", None, None, None, None

def plot_graph(function_type: str, function_parameters: Any, x_min: float, x_max: float) -> bool:
    x = np.linspace(x_min, x_max, 400)
    y = np.zeros_like(x)
    if function_type == 'polynomial':
        coefficients = function_parameters
        degree = len(coefficients) - 1
        for i, coeff in enumerate(coefficients):
            power = degree - i
            y += coeff * (x ** power)

        poly_str = ""
        for i, coeff in enumerate(coefficients):
            power = degree - i
            if coeff != 0:
                sign = "+" if coeff > 0 and i > 0 else ""
                if power > 1:
                    poly_str += f"{sign}{coeff}x^{power}"
                elif power == 1:
                    poly_str += f"{sign}{coeff}x"
                else:
                    poly_str += f"{sign}{coeff}"
        if poly_str.startswith("+"):
            poly_str = poly_str[1:]

        title_func = f"y = {poly_str}"

    elif function_type == 'sin':
        k = function_parameters
        y = np.sin(k * x)
        title_func = f"y = sin({k}x)"
    elif function_type == 'cos':
        k = function_parameters
        y = np.cos(k * x)
        title_func = f"y = cos({k}x)"
    else:
        print(f"Error: Unknown function type '{function_type}' (internal error).")
        return False

    plt.figure(figsize=(8, 6))
    plt.plot(x, y)
    plt.title(f'Plot of {title_func} from {x_min} to {x_max}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()
    return True


if __name__ == "__main__":
    while True:
        user_input = input("What graph do you want to plot? (or say 'bye' to exit): ")
        llm_response = get_llm_response(user_input)
        print(f"LLM response: {llm_response}")
        action, function_type, function_params, x_min, x_max = parse_llm_output(llm_response)

        if action == "exit":
            print("Thank you, goodbye!")
            break
        elif action == "plot":
            if plot_graph(function_type, function_params, x_min, x_max):
                print("Graph plotted successfully.")
            else:
                print("Sorry, there was an error plotting the graph.")
        elif action == "unknown":
            print(
                "Sorry, I didn't understand your request. Please ask again using polynomials (up to power 4), sin(kx), or cos(kx).")
        elif action == "error":
            print("Sorry, there was an error communicating with the LLM. Please try again later.")