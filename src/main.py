import os
import sys
import time
import random
import collections

import yaml
import numpy as np
import torch as th

from copy import deepcopy
from os.path import dirname, abspath

from ray import tune
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
from ray.tune.schedulers import PopulationBasedTraining

from run import run
from utils.logging import get_logger

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
th.set_num_threads(1)
th.set_num_interop_threads(1)

SETTINGS['CAPTURE_MODE'] = "fd" # set to "no" if you want to see stdout/stderr in console
logger = get_logger()

ex = Experiment("pymarl")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"]

    # save pid
    config['pid'] = os.getpid()

    # run the framework
    run(_run, config, _log)


def _get_basic_config(params, other_params=None, arg_name='env_args.map_name'):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            if other_params is not None:
                other_params.pop(other_params.index(_v))
            return _v.split("=")[1]
    else:
        raise ValueError('arg_name: {} should be set in command arguments, '
                         'it is not found in {}'.format(arg_name, params))


def _get_config(params, arg_name, subfolder):
    config_name = None
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            config_name = _v.split("=")[1]
            del params[_i]
            break

    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict


def recursive_dict_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    params = deepcopy(sys.argv)
    params_bak = deepcopy(sys.argv)

    # Get the defaults from default.yaml
    with open(os.path.join(os.path.dirname(__file__), "config", "default.yaml"), "r") as f:
        try:
            config_dict = yaml.load(f)
        except yaml.YAMLError as exc:
            assert False, "default.yaml error: {}".format(exc)

    # Load algorithm and env base configs
    env_config = _get_config(params, "--env-config", "envs")
    alg_config = _get_config(params, "--config", "algs")
    # config_dict = {**config_dict, **env_config, **alg_config}
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    results_path = _get_basic_config(params_bak, other_params=params, arg_name='--results-dir')
    env_name = _get_basic_config(params_bak, arg_name='env_args.map_name')
    algo_name = _get_basic_config(params_bak, arg_name='--config')
    # Save to disk by default for sacred
    save_path = os.path.join(results_path, env_name, algo_name + '_pbt' if config_dict['use_pbt'] else '')

    logger.info(f"Saving to FileStorageObserver in {save_path}/sacred.")
    file_obs_path = os.path.join(save_path, "sacred")

    # save the results to config_dict and can be used in args
    config_dict['local_results_path'] = file_obs_path

    # now add all the config to sacred
    ex.add_config(config_dict)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
