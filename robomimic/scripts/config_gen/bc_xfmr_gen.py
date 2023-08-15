from robomimic.scripts.config_gen.helper import *

def make_generator_helper(args):
    algo_name_short = "bc_xfmr"

    generator = get_generator(
        algo_name="bc",
        config_file=os.path.join(base_path, 'robomimic/exps/templates/bc_transformer.json'),
        args=args,
        algo_name_short=algo_name_short,
        pt=True,
    )
    if args.ckpt_mode is None:
        args.ckpt_mode = "off"

    if args.env == "r2d2":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": p} for p in scan_datasets("~/Downloads/example_pen_in_cup", postfix="trajectory_im128.h5")],
            ],
            value_names=[
                "pen-in-cup",
            ],
        )
    elif args.env == "kitchen":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [{"path": "~/datasets/kitchen/prior/human_demos/pnp_table_to_cab/all/20230806_im84.hdf5", "filter_key": "100_demos"}],
                [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "100_demos"}],
                [{"path": "~/datasets/kitchen/prior/mimicgen/pnp_table_to_cab/viraj_mg_2023-08-10-20-31-14/demo_im84.hdf5", "filter_key": "1000_demos"}],
            ],
            value_names=[
                "human-100",
                "mg-100",
                "mg-1000",
            ],
        )
    elif args.env == "square":
        generator.add_param(
            key="train.data",
            name="ds",
            group=2,
            values=[
                [
                    {"path": "~/datasets/square/ph/square_ph_abs_tmp.hdf5"}, # replace with your own path
                ],
            ],
            value_names=[
                "square",
            ],
        )
    else:
        raise ValueError

    # change default settings: predict 10 steps into future
    generator.add_param(
        key="algo.transformer.pred_future_acs",
        name="predfuture",
        group=1,
        values=[True],
        hidename=True,
    )
    generator.add_param(
        key="algo.transformer.supervise_all_steps",
        name="supallsteps",
        group=1,
        values=[True],
        hidename=True,
    )
    generator.add_param(
        key="algo.transformer.causal",
        name="causal",
        group=1,
        values=[False],
        hidename=True,
    )
    generator.add_param(
        key="train.seq_length",
        name="",
        group=1,
        values=[10],
        hidename=True,
    )
    
    generator.add_param(
        key="train.output_dir",
        name="",
        group=-1,
        values=[
            "~/expdata/{env}/{mod}/{algo_name_short}".format(
                env=args.env,
                mod=args.mod,
                algo_name_short=algo_name_short,
            )
        ],
    )

    return generator

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    make_generator(args, make_generator_helper)
