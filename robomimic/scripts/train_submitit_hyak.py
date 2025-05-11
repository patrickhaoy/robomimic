import time
from dataclasses import dataclass

from submitit_tools.jobs import SubmititState, BaseJob, create_function_job, grid_search_job_configs
from submitit_tools.configs import SubmititExecutorConfig, BaseJobConfig, WandbConfig

#  This is the "TopLevel" Executor Config that submitit uses to execute all jobs.
@dataclass
class ExampleExecutorConfig(SubmititExecutorConfig):
    timeout_min: int = 48 * 60
    slurm_partition: str = "gpu-l40s"
    root_folder: str = "logging_dir"
    cpus_per_task: int = 16
    mem_gb: int = 200
    slurm_gpus_per_node: str = "1" # this is saying we want 1 gpu per node
    # slurm_constraint:str = "l40s" # This is saying we need a node with these gpus

@dataclass
class ArgumentsConfig(BaseJobConfig):
    config: str = "dp_rgb"

job_configs = [ArgumentsConfig(config=config) for config in ["mlp_rgb", "mlp_rgb_cnn", "mlp_rgb_r3m", "mlp_rgb_dinov2", "mlp_rgb_r3m_finetune", "dp_rgb"]]

# # Since we do not need any checkpointing functionality, and the jobs will 
# # not use the checkpoint path at all, we can use the base config.
# job_configs = [BaseJobConfig for _ in range(10)]

# No wandb
wandb_configs = None

# This defines the function that our job should execute. This is used to create
# a function job. It takes in the job_cfg as a parameter (but doesn't use it in this case)
def job_fn(job_cfg: ArgumentsConfig):
    import sys
    import argparse
    import os
    from robomimic.scripts.train import main, get_parser

    # Step 1: Inject defaults via sys.argv (simulate CLI input)
    sys.argv = [
        "script.py",  # dummy placeholder for program name
        "--config", f"robomimic/exps/templates/{job_cfg.config}.json",
        "--name", f"{job_cfg.config}_v3",
        "--dataset", f"/tmp/datasets_{job_cfg.config}"
    ]

    # Step 2: Copy datasets to /tmp
    os.system(f"time cp -a ../OctiLab/datasets /tmp/datasets_{job_cfg.config}")

    # Step 3: Parse arguments and call main
    parser = get_parser()
    args = parser.parse_args()
    main(args)

# This will return a job class that all it does is call and return the job_fn. We can
# pass this to the executor
GetNodeGPUJob = create_function_job(job_fn)

# 4. Create a Submitit state manager.
state = SubmititState(
    job_cls=GetNodeGPUJob,
    executor_config=ExampleExecutorConfig(),
    job_run_configs=job_configs,
    job_wandb_configs=wandb_configs,
    max_retries=10,
)

# 5. Wait and then get the results of the submission
results = state.run_all_jobs()

# 6. Output the results. In this case, it is the gpu type
for result in results:
    print(result)
