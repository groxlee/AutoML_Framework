from settings import from_argparse, CheckpointSettings
from directory_utils import validate_existing_directories
from exception import CheckPointException
from cli_util import parser
from typing import Optional, List, Dict, Any
from trainer.conv import ConvNet
from fuel.BaseTorch import BaseTorch
from fuel.TuneTorch import TuneTorch

def parse_command_line(argv: Optional[List[str]] = None):
    args = parser.parse_args(argv)
    config, cs = from_argparse(args)
    return config, cs

def read_test():
    config, cs = parse_command_line()
    print(config)
    print(cs.resume)
    print(cs.run_id)
    print(cs.interval)

def smoke_test(model, config, cs):
    test = BaseTorch(model, config, cs)
    test.start()

def tune_smoke_test(model, config, cs):
    test = TuneTorch(model, config, cs)
    test.start()

def test():
    config, cs = parse_command_line()
    model = ConvNet()
    validate_existing_directories(cs.run_logs_dir, cs.resume, cs.force)
    if config["use_tune"] is True:
        tune_smoke_test(model, config, cs)
    else:
        smoke_test(model, config, cs)

def main():
    # read_test()
    test()


if __name__ == "__main__":
    main()