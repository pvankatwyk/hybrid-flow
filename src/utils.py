import yaml


def get_configs():
    # Loads configuration file in the repo and formats it as a dictionary.
    try:
        with open('../config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    except FileNotFoundError:     # depends on where you're calling it from...
        with open('./config.yaml') as c:
            data = yaml.load(c, Loader=yaml.FullLoader)
    return data


def check_input(input, options):
    # simple assert that input is in the designated options (readability purposes only)
    input = input.lower()
    assert input in options, f"input must be in {options}, received {input}"
