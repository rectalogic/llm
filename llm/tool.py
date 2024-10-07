import enum
import inspect
import json
import types
from typing import Any, Annotated, Generic, Union, ParamSpec, get_origin
from collections.abc import Callable
from functools import update_wrapper

# __origin__ could be types.UnionType instance (for optional parameters that have a None default) or a class
TYPEMAP = {
    int: "integer",
    Union[int | None]: "integer",
    float: "number",
    Union[float | None]: "number",
    list: "array",
    Union[list | None]: "array",
    bool: "boolean",
    Union[bool | None]: "boolean",
    str: "string",
    Union[str | None]: "string",
    types.NoneType: "null",
}


def convert_parameter(param: inspect.Parameter) -> dict[str, Any]:
    """Convert a function parameter to a JSON schema parameter."""
    annotation = param.annotation
    # This will return Annotated, or None for inspect.Parameter.empty or other types
    unsubscriped_type = get_origin(annotation)
    if not (
        unsubscriped_type is Annotated
        and len(annotation.__metadata__) == 1
        and isinstance(annotation.__metadata__[0], str)
    ):
        raise ValueError(
            "Function parameters must be annotated with typing.Annotated[<type>, 'description']"
        )

    schema: dict[str, Any] = {
        "description": annotation.__metadata__[0],
    }

    origin = annotation.__origin__
    type_ = TYPEMAP.get(origin)
    if type_:
        schema["type"] = type_
    elif issubclass(origin, enum.StrEnum):
        schema["type"] = "string"
        schema["enum"] = [m.value for m in origin]
    else:
        raise TypeError(f"Annotated parameter type {origin} not supported")

    return schema


def format_exception(e: Exception) -> str:
    return json.dumps({"is_error": True, "exception": repr(e)})


def format_error(message: str) -> str:
    return json.dumps({"is_error": True, "error": message})


P = ParamSpec("P")


class Tool(Generic[P]):
    schema: dict[str, Any]

    def __init__(self, function: Callable[P, str]) -> None:
        update_wrapper(self, function)
        self.function = function
        signature = inspect.signature(function)
        if not function.__doc__:
            raise ValueError("Tool functions must have a doc comment description")
        if signature.return_annotation is not str:
            raise ValueError("Tool functions must return a string")

        self.schema = {
            "type": "function",
            "function": {
                "name": function.__name__,
                "description": function.__doc__,
            },
        }
        if signature.parameters:
            self.schema["function"]["parameters"] = {
                "type": "object",
                "properties": {
                    name: convert_parameter(param)
                    for name, param in signature.parameters.items()
                },
                "required": [
                    name
                    for name, param in signature.parameters.items()
                    if param.default is inspect.Parameter.empty
                ],
            }

    def __call__(self, /, *args: P.args, **kwargs: P.kwargs) -> str:
        return self.function(*args, **kwargs)

    def safe_call(self, json_args: str) -> str:
        try:
            return self.function(**json.loads(json_args))
        except Exception as e:
            return format_exception(e)
