from recstudio.utils import *
from recstudio import GRE
import torch


if __name__ == '__main__':
    parser = get_default_parser()
    args, command_line_args = parser.parse_known_args()
    parser = add_model_arguments(parser, args.model)
    command_line_conf = parser2nested_dict(parser, command_line_args)
    command_line_conf['main'] = {}
    command_line_conf['main']['dataset_path'] = args.dataset_path
    command_line_conf['main']['use_dataset_config'] = args.use_dataset_config

    model_class, model_conf = get_model(args.model)
    model_conf = deep_update(model_conf, command_line_conf)

    GRE.run(args.model, 
            dataset_path=args.dataset_path,
            model_config=model_conf, 
            run_mode=args.mode,
        )
