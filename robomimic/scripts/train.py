"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

import argparse
import json
import numpy as np
import time
import os
import psutil
import sys
import traceback
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.lang_utils as LangUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings
from robomimic.utils.dataset import MetaDataset


def scan_datasets(folder, postfix=".hdf5"):
    """
    Recursively scan a folder for HDF5 files.

    Args:
        folder (str): path to folder to scan
        postfix (str): file extension to look for

    Returns:
        list: list of paths to HDF5 files
    """
    dataset_paths = []
    for root, dirs, files in os.walk(os.path.expanduser(folder)):
        for f in files:
            if f.endswith(postfix):
                dataset_paths.append(os.path.join(root, f))
    return dataset_paths


def train(config, device, eval_only=False):
    """
    Train a model using the algorithm.
    """

    # Check action normalization requirements
    assert config.train.action_config["actions"]["normalization"] == "min_max", "Actions must be normalized to [-1, 1] for tanh output"
    # Check frame stacking requirements
    if config.algo_name == "bc":
        assert config.train.frame_stack == 1, "BC does not support frame stacking"

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    # set num workers
    torch.set_num_threads(1)

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir, vis_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists and handle directories
    if isinstance(config.train.data, str):
        dataset_path = os.path.expandvars(os.path.expanduser(config.train.data))
        if os.path.isdir(dataset_path):
            # If it's a directory, scan for HDF5 files
            dataset_paths = scan_datasets(dataset_path)
            if not dataset_paths:
                raise Exception("No HDF5 files found in directory: {}".format(dataset_path))
            print("Found {} HDF5 files in directory: {}".format(len(dataset_paths), dataset_path))
        else:
            dataset_paths = [dataset_path]
    else:
        dataset_paths = []
        for dataset_cfg in config.train.data:
            path = os.path.expandvars(os.path.expanduser(dataset_cfg["path"]))
            if os.path.isdir(path):
                # If it's a directory, scan for HDF5 files
                dir_paths = scan_datasets(path)
                if not dir_paths:
                    raise Exception("No HDF5 files found in directory: {}".format(path))
                print("Found {} HDF5 files in directory: {}".format(len(dir_paths), path))
                dataset_paths.extend(dir_paths)
            else:
                dataset_paths.append(path)

    ds_format = config.train.data_format
    for dataset_path in dataset_paths:
        if not os.path.exists(dataset_path):
            raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from first training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path=dataset_paths[0],
        ds_format=ds_format
    )

    # update env meta if applicable
    from robomimic.utils.script_utils import deep_update
    deep_update(env_meta, config.experiment.env_meta_update_dict)

    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_paths[0],
        action_keys=config.train.action_keys,
        all_obs_keys=config.all_obs_keys,
        ds_format=ds_format,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    print("")

    # setup for a new training run
    data_logger = DataLogger(
        log_dir,
        config,
        log_tb=config.experiment.logging.log_tb,
        log_wandb=config.experiment.logging.log_wandb,
    )
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )

    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    ckpt_path = config.experiment.ckpt_path
    if ckpt_path is not None and os.path.isfile(os.path.expanduser(ckpt_path)):
        print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
        from robomimic.utils.file_utils import maybe_dict_from_checkpoint
        ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
        model.deserialize(ckpt_dict["model"])

    print("\n============= Model Summary =============")
    print(model)  # print model summary
    print("")

    # load training data
    lang_encoder = LangUtils.LangEncoder(
        device=device,
    )

    trainsets = []
    validsets = []
    for dataset_path in dataset_paths:
        # Create a temporary config for this dataset
        temp_config = deepcopy(config)
        temp_config.train.data = dataset_path

        # Load dataset using dataset_factory
        train_dataset, valid_dataset = TrainUtils.load_data_for_training(
            config=temp_config,
            obs_keys=shape_meta["all_obs_keys"],
            lang_encoder=lang_encoder
        )
        trainsets.append(train_dataset)

        if valid_dataset is not None:
            validsets.append(valid_dataset)

    # Combine datasets if multiple are provided
    if len(trainsets) > 1:
        trainset = MetaDataset(trainsets, ds_weights=[1.0]*len(trainsets), normalize_weights_by_ds_size=True)
    else:
        trainset = trainsets[0]

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # maybe retreve statistics for normalizing actions
    action_normalization_stats = trainset.get_action_normalization_stats()

    if validsets:
        if len(validsets) > 1:
            validset = MetaDataset(validsets, ds_weights=[1.0]*len(validsets), normalize_weights_by_ds_size=True)
        else:
            validset = validsets[0]
        validset.set_action_normalization_stats(action_normalization_stats)
    else:
        validset = None

    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # main training loop
    best_valid_loss = None
    last_ckpt_time = time.time()

    # number of learning steps per epoch (defaults to a full dataset pass)
    train_num_steps = config.experiment.epoch_every_n_steps
    valid_num_steps = config.experiment.validation_epoch_every_n_steps
    
    for epoch in range(0, config.train.num_epochs + 1):  # epoch numbers start at 1        
        # if checkpoint directory is specified, load in new ckpt if exists
        ckpt_path = config.experiment.ckpt_path
        if ckpt_path is not None and os.path.isdir(os.path.expanduser(ckpt_path)):
            ckpt_path = os.path.join(ckpt_path, "models", f"model_epoch_{epoch}.pth")
            if os.path.exists(ckpt_path):
                print("LOADING MODEL WEIGHTS FROM " + ckpt_path)
                from robomimic.utils.file_utils import maybe_dict_from_checkpoint
                ckpt_dict = maybe_dict_from_checkpoint(ckpt_path=ckpt_path)
                model.deserialize(ckpt_dict["model"])
        
        if epoch > 0 and not eval_only:
            step_log = TrainUtils.run_epoch(
                model=model,
                data_loader=train_loader,
                epoch=epoch,
                num_steps=train_num_steps,
                obs_normalization_stats=obs_normalization_stats
            )
            model.on_epoch_end(epoch)

            # setup checkpoint path
            epoch_ckpt_name = "model_epoch_{}".format(epoch)

            # check for recurring checkpoint saving conditions
            should_save_ckpt = False
            if config.experiment.save.enabled:
                time_check = (config.experiment.save.every_n_seconds is not None) and \
                    (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
                epoch_check = (config.experiment.save.every_n_epochs is not None) and \
                    (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
                epoch_list_check = (epoch in config.experiment.save.epochs)
                should_save_ckpt = (time_check or epoch_check or epoch_list_check)
            ckpt_reason = None
            if should_save_ckpt:
                last_ckpt_time = time.time()
                ckpt_reason = "time"

            print("Train Epoch {}".format(epoch))
            print(json.dumps(step_log, sort_keys=True, indent=4))
            for k, v in step_log.items():
                if k.startswith("Time_"):
                    data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
                else:
                    data_logger.record("Train/{}".format(k), v, epoch)

            # Evaluate the model on validation set
            if config.experiment.validate:
                with torch.no_grad():
                    step_log = TrainUtils.run_epoch(
                        model=model,
                        data_loader=valid_loader,
                        epoch=epoch,
                        validate=True,
                        num_steps=valid_num_steps,
                        obs_normalization_stats=obs_normalization_stats
                    )
                for k, v in step_log.items():
                    if k.startswith("Time_"):
                        data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
                    else:
                        data_logger.record("Valid/{}".format(k), v, epoch)

                print("Validation Epoch {}".format(epoch))
                print(json.dumps(step_log, sort_keys=True, indent=4))

                # save checkpoint if achieve new best validation loss
                valid_check = "Loss" in step_log
                if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
                    best_valid_loss = step_log["Loss"]
                    if config.experiment.save.enabled and config.experiment.save.on_best_validation:
                        epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
                        should_save_ckpt = True
                        ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason
        else:
            should_save_ckpt = False
            epoch_ckpt_name = "model_epoch_{}".format(epoch)
            ckpt_reason = None

        # Save model checkpoints based on conditions (validation loss)
        if should_save_ckpt:
            TrainUtils.save_model(
                model=model,
                config=config,
                env_meta=env_meta,
                shape_meta=shape_meta,
                ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
                obs_normalization_stats=obs_normalization_stats,
                action_normalization_stats=action_normalization_stats,
            )

        # Finally, log memory usage in MB
        process = psutil.Process(os.getpid())
        mem_usage = int(process.memory_info().rss / 1000000)
        data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
        print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # terminate logging
    data_logger.close()


def main(args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        # config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2
        config.train.batch_size = 4

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    # config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, device=device, eval_only=args.eval_only)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


def get_parser():
    parser = argparse.ArgumentParser()
    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    # debug mode
    parser.add_argument(
        "--eval_only",
        action='store_true',
        help="disables training and only runs policy evaluation. config must include ckpt_path"
    )

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
