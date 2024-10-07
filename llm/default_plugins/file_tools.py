import json
import glob
from typing import Annotated
import llm


@llm.hookimpl
def register_tools(register):
    register(read_file)


@llm.Tool
def read_file(
    filename: Annotated[
        str, "The path to the file to read. Can be a Python `glob.glob()` pattern."
    ]
) -> str:
    """Read the given filename and return the contents."""
    result = []
    for filename in glob.glob(filename):
        with open(filename, "r") as f:
            result.append({"filename": filename, "contents": f.read()})
    return json.dumps(result)
