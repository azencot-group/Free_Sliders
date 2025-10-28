import argparse
import importlib as importlib
from typing import List

from methods.GuidedDiffusion import GuidedDiffusion

from utils.config_utils import load_config_from_yaml
from utils.prompt_utils import load_prompts_from_yaml
from utils.utils import flush

from loggers import create_default_logger, CompositeLogger, NeptuneLogger, PrintLogger, BaseLogger

import warnings
warnings.filterwarnings(
    "ignore",
    message="Accessing config attribute `__len__` directly via 'CogVideoXTransformer3DModel' object attribute is deprecated.*",
    category=FutureWarning,
)

def setup_logger(args, rank=0):
    """Set up the experiment logger based on args."""
    loggers : List[BaseLogger] = [create_default_logger(rank=rank)]

    if args.neptune:
        try:
            neptune_logger = NeptuneLogger(
                rank=rank
            )
            loggers.append(neptune_logger)
        except FileNotFoundError:
            print("Neptune credentials not found, falling back to PrintLogger only")

    return CompositeLogger(loggers) if len(loggers) > 1 else loggers[0]


def main(base_model, args, prompts):

    # Save the prompts and prompts embeddings to be evaluated
    text_embeddings = {}
    if args.BaseModel == "T2I" or args.static:
        static_prompt_embeddings = base_model.encode_prompts(prompts['static'])
        text_embeddings["static"] = static_prompt_embeddings

    base_model.delete_and_flush()

    if args.Guided:
        guided_model = GuidedDiffusion(base_model)
        guided_model(
            args=args,
            text_embeddings=text_embeddings,
            prompts=prompts,
            logger=logger,
        )

    logger.stop()

    del (
        base_model.network,
        base_model.noise_scheduler,
    )

    flush()



if __name__ == '__main__':
    flush()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", default="configs/config.yaml", help="Config file for running.")
    parser.add_argument("--BaseModel", type=str, default="T2I", choices=["T2I", "T2V"], required=True, help="Base model to use for the experiment "
                                                                                    "(T2I: Stable Diffusion, T2V: CogVideoX)")
    parser.add_argument("--Guided", action='store_const', const=True, default=False)
    parser.add_argument("--neptune", action='store_const', const=True, default=False)
    parser.add_argument("--num_images", type=int, default=10, help="Number of sliders to generate for each concept")
    parser.add_argument("--prompt_index", type=int, default=-1, help="Use a specific prompt index from the config file, -1 for all prompts")

    args = parser.parse_known_args()

    if args[0].Guided:
        parser.add_argument('--t', type=int, help="T values", default=15)
        parser.add_argument('--scales', type=float, nargs='+', help="Scales for the model",)
        parser.add_argument('--guidance', type=float, help="Guidance scales", default=6.0)
        parser.add_argument('--guidance2', type=float, default=1.0, help="Guidance scales")

    args = parser.parse_args()

    # For path names to prevent concurrent reading errors
    from datetime import datetime
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    logger = setup_logger(args, rank=0)
    logger.log_hparams(vars(args))

    # ------------------------ load config ------------------------
    config_file = args.config_file
    config = load_config_from_yaml(config_file)
    logger.add_tags(args.BaseModel)

    # ------------------------ load prompts and base model ------------------------
    prompts = {}

    if args.BaseModel == "T2I":
        module_name = "base_models.BaseModel_StableDiffusion"

        image_prompts = load_prompts_from_yaml(config.prompts_file, [])
        prompts["static"] = image_prompts

    if args.prompt_index != -1 and args.static:
        if 0 <= args.prompt_index < len(prompts["static"]):
            prompts["static"] = [prompts["static"][args.prompt_index]]
        else:
            print(f"Prompt index {args.prompt_index} is out of range. Using all prompts.")

    if args.prompt_index != -1 and args.dynamic:
        if 0 <= args.prompt_index < len(prompts["dynamic"]):
            prompts["dynamic"] = [prompts["dynamic"][args.prompt_index]]
        else:
            print(f"Prompt index {args.prompt_index} is out of range. Using all prompts.")

    BaseModelClass = importlib.import_module(module_name).BaseModel
    baseModel = BaseModelClass(config, run_id)

    main(baseModel, args, prompts)

