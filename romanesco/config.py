#!/usr/bin/env python3

import json
import argparse
import logging

def save_config(args: argparse.Namespace,
                config_path: str):

    logging.info("Saving model config to '%s'." % config_path)

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)


def update_namespace_from_config(args: argparse.Namespace,
                                 config_path: str):

    logging.info("Loading model config from '%s'." % config_path)

    args_dict = vars(args)

    updateable_args = ["hidden_size", "embedding_size", "vocab_max_size", "num_steps"]

    with open(config_path, "r") as f:
        for key, value in json.load(f).items():
            if key in updateable_args:
                args_dict[key] = value

    return args
