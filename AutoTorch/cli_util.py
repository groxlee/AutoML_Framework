import os
import yaml
from typing import Set, Dict, Any, TextIO
import argparse
from exception import ConfigError

class DetectDefault(argparse.Action):
    """
    Internal custom Action to help detect arguments that aren't default.
    """

    non_default_args: Set[str] = set()

    def __call__(self, arg_parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)
        DetectDefault.non_default_args.add(self.dest)


class DetectDefaultStoreTrue(DetectDefault):
    """
    Internal class to help detect arguments that aren't default.
    Used for store_true arguments.
    """

    def __init__(self, nargs=0, **kwargs):
        super().__init__(nargs=nargs, **kwargs)

    def __call__(self, arg_parser, namespace, values, option_string=None):
        super().__call__(arg_parser, namespace, True, option_string)


class StoreConfigFile(argparse.Action):
    """
    Custom Action to store the config file location not as part of the CLI args.
    maintain an equivalence between the config file's contents and the args 
    themselves.
    """

    config_path: str

    def __call__(self, arg_parser, namespace, values, option_string=None):
        delattr(namespace, self.dest)
        StoreConfigFile.config_path = values


def _create_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    argparser.add_argument(
        "config_path", action=StoreConfigFile, nargs="?", default=None
    )

    # add other arg parser here
    argparser.add_argument(
        "--run-id",
        default="conv_test",
        help="The identifier for the training run.",
        dest="run_id",
    )
    argparser.add_argument(
        "--resume",
        default=False,
        dest="resume",
        action=DetectDefaultStoreTrue,
        help="Whether to resume training from a checkpoint. Specify a --run-id to use this option. ",
    )
    argparser.add_argument(
        "--force",
        default=False,
        dest="force",
        action=DetectDefaultStoreTrue,
        help="Whether to force-overwrite this run-id's existing summary and model data. (Without "
        "this flag, attempting to train a model with a run-id that has been used before will throw "
        "an error.",
    )
    argparser.add_argument(
        "--results-dir",
        default="results",
        help="Results base directory",
    )

    return argparser


def load_config(config_path: str) -> Dict[str, Any]:
    try:
        with open(config_path) as data_file:
            return _load_config(data_file)
    except OSError:
        abs_path = os.path.abspath(config_path)
        raise ConfigError(f"Config file could not be found at {abs_path}.")
    except UnicodeDecodeError:
        raise ConfigError(
            f"There was an error decoding Config file from {config_path}. "
            f"Make sure your file is save using UTF-8"
        )

def _load_config(fp: TextIO) -> Dict[str, Any]:
    """
    Load the yaml config from the file-like object.
    """
    try:
        return yaml.safe_load(fp)
    except yaml.parser.ParserError as e:
        raise ConfigError(
            "Error parsing yaml file. Please check for formatting errors. "
            "A tool such as http://www.yamllint.com/ can be helpful with this."
        ) from e


parser = _create_parser()