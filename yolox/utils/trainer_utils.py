import os
import math


def create_output_file_name(output_dir, experiment_name):
    file_name = os.path.join(output_dir, experiment_name)
    suffix_number = 1
    while os.path.exists(file_name):  # check for existing experiment with same name
        file_name = os.path.join(output_dir, experiment_name + str(suffix_number))
        suffix_number += 1
    return file_name


def setup_wandb_logger(logger, experiment_name):
    if logger is not None:
        return logger
    import wandb

    wandb.init(project=experiment_name)  # in this method, unable to log more configs
    return wandb

