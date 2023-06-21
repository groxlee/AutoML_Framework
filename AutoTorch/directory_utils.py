import os
import shutil
from exception import CheckPointException

def validate_existing_directories(
    output_path: str, resume: bool, force: bool
) -> None:
    """
    Validates that if the run_id model exists, we do not overwrite it unless --force is specified.
    Throws an exception if resume isn't specified and run_id exists. Throws an exception
    if --resume is specified and run-id was not found.
    """

    output_path_exists = os.path.isdir(output_path)

    if output_path_exists:
        if not resume and not force:
            raise CheckPointException(
                "Previous data from this run ID was found. "
                "Either specify a new run ID, use --resume to resume this run, "
                "or use the --force parameter to overwrite existing data."
            )
        if force:
            shutil.rmtree(output_path)
    else:
        if resume:
            raise CheckPointException(
                "Previous data from this run ID was not found. "
                "Train a new run by removing the --resume flag."
            )