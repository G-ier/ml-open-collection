import tqdm
import argparse

def linear_regression(x: list, y: list):
    pass

def polynomial_regression(x: list, y: list, degree: int):
    pass

def logistic_regression(x: list, y: list):
    pass

# Define the regression factory
def regression_factory(type: str, args: dict):

    # Define the model
    if type == "linear":
        return linear_regression(args["x"], args["y"])
    elif type == "polynomial":
        return polynomial_regression(args["x"], args["y"], args["degree"])
    elif type == "logistic":
        return logistic_regression(args["x"], args["y"])
    else:
        raise ValueError(f"Model {type} not found")

if __name__ == "__main__":
    
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, required=True)
    parser.add_argument("--x", type=list, required=True)
    parser.add_argument("--y", type=list, required=True)

    # Define the degree for the polynomial regression
    degree_required = False
    if type == "polynomial":
        degree_required = True
    parser.add_argument("--degree", type=int, required=degree_required)
    args = parser.parse_args()
    
    # Run the regression
    regression_factory(args.type, args)