import os
import attr
import warnings
from typing import (
    Dict,
    Any,
)
from enum import Enum
import argparse
import numpy as np

from cli_util import StoreConfigFile, load_config, parser
from exception import ConfigWarning

class CheckpointSettings:
    def __init__(self):
        self.dict: Dict[str, Any] = {
            "run_id": parser.get_default("run_id"),
            "resume": parser.get_default("resume"),
            "force": parser.get_default("force"),
            "results_dir": parser.get_default("results_dir"),
            "interval": 1000,
        }
        self.update()        

    def update(self):
        self.run_id: str = self.dict["run_id"]
        self.resume: bool = self.dict["resume"]
        self.force: bool = self.dict["force"]
        self.results_dir: str = self.dict["results_dir"]
        self.interval: int = self.dict["interval"]

    @property
    def write_path(self) -> str:
        return os.path.join(self.results_dir, self.run_id)
    
    @property
    def run_logs_dir(self) -> str:
        return os.path.join(self.write_path, "run_logs")


def from_argparse(args: argparse.Namespace) -> Dict[str, Any]:
    argparse_args = vars(args)
    config_path = StoreConfigFile.config_path

    # load yaml file
    config_dict: Dict[str, Any] = {}
    if config_path is not None:
        config_dict.update(load_config(config_path))
    else:
        warnings.warn(
            f"invalid config file path.",
            ConfigWarning,
        )
    
    # add other parser
    checkpoint_settings = CheckpointSettings()
    for key, val in argparse_args.items():
        checkpoint_settings.dict[key] = val
    checkpoint_settings.dict["interval"] = config_dict["checkpoint_settings"]["interval"]
    checkpoint_settings.update()
    
    return config_dict, checkpoint_settings
