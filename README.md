# The Code of RMIX

RMIX: Learning Risk-Sensitive Policies for Cooperative Reinforcement Learning Agents (NeurIPS 2021)

Note that our code is built on top of [PyMARL](https://github.com/oxwhirl/pymarl). You can read the code of PyMARL to start your MARL research.

## Installation instructions

Build the Dockerfile using 
```shell
cd docker
bash build.sh
```

Set up StarCraft II and SMAC:
```shell
bash install_sc2.sh
```

This will download SC2 into the 3rdparty folder and copy the maps necessary to run over.

The requirements.txt file can be used to install the necessary packages into a virtual environment (not recomended).


## Run an experiment 

```shell
python3 src/main.py --config=rmix --results-dir=/path/to/results --env-config=sc2 with env_args.map_name=1c3s5z t_max=1050000
```

The config files act as defaults for an algorithm or environment. 

They are all located in `src/config`.
`--config` refers to the config files in `src/config/algs`
`--env-config` refers to the config files in `src/config/envs`

To run experiments using the Docker container:
```shell
bash run.sh $GPU python3 src/main.py --config=rmix --results-dir=/path/to/results --env-config=sc2 with env_args.map_name=1c3s5z t_max=1050000
```

Results are stored in `/path/to/results` and you can set `save_model=True` to save the model and use the trained models for evaluation.

If you want to use RMIX on new scenarios, you can tune the `start_to_update_qr` and the `qr_update_interval` in your experiments. When `start_to_update_qr=max_time_step`, QR will not be used to update the distribution and CVaR is reduced to k-minimum `δ` values and will be trained in an implicit manner.
