import math


def build_settings_from_config(config, exp, args):
    for param, value in config.items():
        if isinstance(value, float):  # ensure no negative values
            if value < 0.0:  # ensure no negative values
                value = 0.0
            if param in vars(exp):  # cap probability values
                if vars(exp)[param] <= 1.0 and value >= 1.0:
                    value = 1.0

        if param in vars(exp):
            setattr(exp, param, value)
            continue
        if param in vars(args):
            setattr(args, param, value)
            continue
        raise Exception(f"Parameter {param} : {value} could not be set")

    return exp, args


def build_config(exp, args, perform_hpo, HPO_type="bayes"):
    # TODO work out details for this
    hyperparameters = exp.hyperparameters
    exp_params = vars(exp)
    arg_params = vars(args)

    config = {}

    if not perform_hpo:  # just save hyperparm details
        for key in hyperparameters:
            if key in exp_params:
                config[key] = exp_params[key]
            if key in arg_params:
                config[key] = arg_params[key]
        return config

    custom_params = exp.get_hpo_custom_params()

    for key in hyperparameters:
        if key in exp_params:
            value = exp_params[key]
        if key in arg_params:
            value = arg_params[key]
        # parameters need to be saved differently for HPO
        # how params are represented depend on hpo type and nature of param.

        if key in custom_params:
            config[key] = custom_params[key]
            continue

        # defaults:
        # assumption: there are no values that should be < 0
        if isinstance(value, int):
            config[key] = {
                "distribution": "q_normal",
                "mu": value,
                "sigma": value // 2,
                "q": 1,
            }
        elif isinstance(value, float):
            config[key] = {
                "distribution": "normal",
                "mu": value,  # value used as good guess
                "sigma": math.sqrt(value) if value > 0.0 else 1.0,
            }
        elif isinstance(value, bool):
            config[key] = {"distribution": "categorical", "values": [True, False]}
        else:
            raise Exception(
                f"Could not find appropriate config settings for {key} with value {value} | {type(value)}"
            )

    return config
