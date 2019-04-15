#!/usr/bin/env python3

import json
import argparse


def save_config(args: argparse.Namespace,
                config_path: str):

    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)


def update_namespace_from_config(args: argparse.Namespace,
                                 config_path: str):

    with open(config_path, "r") as f:
        for key, value in json.load(f).items():
            args[key] = value

    return args