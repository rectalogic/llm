import enum
from typing import Annotated
import json
import os

import pytest
from pytest_httpx import IteratorStream
from click.testing import CliRunner

from llm.tool import Tool
from llm.cli import cli
from llm.default_plugins.file_tools import read_file


def test_no_parameters():
    @Tool
    def tool() -> str:
        "tool description"
        return "output"

    assert tool.schema == {
        "type": "function",
        "function": {"name": "tool", "description": "tool description"},
    }
    assert tool() == "output"


def test_missing_description():
    with pytest.raises(ValueError, match=" description"):

        @Tool
        def tool() -> str:
            return "output"


def test_missing_return():
    with pytest.raises(ValueError, match=" return"):

        @Tool
        def tool():
            "tool description"


def test_missing_annotated():
    with pytest.raises(ValueError, match=" annotated"):

        @Tool
        def tool(a: int) -> str:
            "tool description"


def test_missing_annotated_description():
    with pytest.raises(TypeError, match=" at least two arguments"):

        @Tool
        def tool(a: Annotated[int]) -> str:
            "tool description"


def test_unsupported_parameters():
    with pytest.raises(TypeError, match=" parameter type"):

        @Tool
        def tool(a: Annotated[object, "a desc"]) -> str:
            "tool description"


def test_safe_call():
    @Tool
    def tool(a: Annotated[int, "a desc"]) -> str:
        "tool description"
        return "output"

    assert tool.safe_call(json.dumps({"a": 1})) == "output"

    assert "exception" in tool.safe_call("{}")
    assert "exception" in tool.safe_call(json.dumps({"a": 1, "b": 2}))


def test_annotated_parameters():
    @Tool
    def tool(
        a: Annotated[bool, "a desc"],
        b: Annotated[int, "b desc"] = 1,
        c: Annotated[str | None, "c desc"] = "2",
    ) -> str:
        "tool description"
        return "output"

    assert tool.schema == {
        "type": "function",
        "function": {
            "name": "tool",
            "description": "tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"description": "a desc", "type": "boolean"},
                    "b": {"description": "b desc", "type": "integer"},
                    "c": {"description": "c desc", "type": "string"},
                },
                "required": ["a"],
            },
        },
    }
    assert tool(True) == "output"


def test_enum_parameters():
    class MyEnum(enum.StrEnum):
        A = "a"
        B = "b"

    @Tool
    def tool(
        a: Annotated[MyEnum, "a enum desc"],
        b: Annotated[int, "b desc"] = 1,
    ) -> str:
        "tool description"
        return "output"

    assert tool.schema == {
        "type": "function",
        "function": {
            "name": "tool",
            "description": "tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {
                        "description": "a enum desc",
                        "type": "string",
                        "enum": ["a", "b"],
                    },
                    "b": {"description": "b desc", "type": "integer"},
                },
                "required": ["a"],
            },
        },
    }
    assert tool(MyEnum.A) == "output"


def test_object_tool():
    class MyTool:
        "tool description"

        __name__ = "tool"

        def __call__(
            self,
            a: Annotated[bool, "a desc"],
            b: Annotated[int, "b desc"] = 1,
        ) -> str:
            return "output"

    tool = Tool(MyTool())

    assert tool.schema == {
        "type": "function",
        "function": {
            "name": "tool",
            "description": "tool description",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"description": "a desc", "type": "boolean"},
                    "b": {"description": "b desc", "type": "integer"},
                },
                "required": ["a"],
            },
        },
    }
    assert tool(True, 3) == "output"


def stream_tool_call(datafile):
    with open(datafile) as f:
        for line in f:
            yield f"{line}\n\n".encode("utf-8")


@pytest.fixture
def read_file_mock(monkeypatch):
    def mock_read_file(filename):
        return "some license text"

    monkeypatch.setattr(read_file, "function", mock_read_file)


def test_tool_completion_stream(httpx_mock, read_file_mock, logs_db):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        stream=IteratorStream(
            stream_tool_call(
                os.path.join(os.path.dirname(__file__), "fixtures/stream_tool_call.txt")
            )
        ),
        headers={"Content-Type": "text/event-stream"},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        stream=IteratorStream(
            stream_tool_call(
                os.path.join(
                    os.path.dirname(__file__), "fixtures/stream_tool_call_result.txt"
                )
            )
        ),
        headers={"Content-Type": "text/event-stream"},
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--enable-tools",
            "-m",
            "4o",
            "--key",
            "x",
            "Summarize this file /tmp/LICENSE",
        ],
    )
    assert result.exit_code == 0
    assert result.output == (
        "The file `/tmp/LICENSE` contains text indicating that the software is distributed under a "
        'certain license on an "AS IS" basis, without any warranties or conditions of any kind, either express '
        "or implied. It advises the reader to see the license for specific terms governing permissions and limitations.\n"
    )
    rows = list(logs_db["responses"].rows_where(select="response_json"))
    assert (
        len(json.loads(rows[0]["response_json"])) == 2
    )  # two response_jsons for tools


def test_tool_completion_nostream(httpx_mock, read_file_mock, logs_db):
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-AGA48pr2cMfyvgJC3Z476OsW7Jsus",
            "object": "chat.completion",
            "created": 1728415304,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_Q1DBTR3thWR0Iz7ZF193QF5u",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"filename":"/tmp/LICENSE"}',
                                },
                            }
                        ],
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 73,
                "completion_tokens": 16,
                "total_tokens": 89,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "system_fingerprint": "fp_e5e4913e83",
        },
        headers={"Content-Type": "application/json"},
    )
    httpx_mock.add_response(
        method="POST",
        url="https://api.openai.com/v1/chat/completions",
        json={
            "id": "chatcmpl-AGA49bKBWs0KcqHvXRGJLXjmfkF1v",
            "object": "chat.completion",
            "created": 1728415305,
            "model": "gpt-4o-2024-08-06",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": 'The license in the file states that the software is distributed on an "AS IS" basis, without warranties or conditions of any kind, either express or implied. It also mentions that the specific language governing permissions and limitations is found in the License itself.',
                        "refusal": None,
                    },
                    "logprobs": None,
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 172,
                "completion_tokens": 51,
                "total_tokens": 223,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 0},
            },
            "system_fingerprint": "fp_2f406b9113",
        },
        headers={"Content-Type": "application/json"},
    )
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "--no-stream",
            "--enable-tools",
            "-m",
            "4o",
            "--key",
            "x",
            "Summarize this file /tmp/LICENSE",
        ],
    )
    assert result.exit_code == 0
    assert result.output == (
        'The license in the file states that the software is distributed on an "AS IS" basis, '
        "without warranties or conditions of any kind, either express or implied. "
        "It also mentions that the specific language governing permissions and limitations is found in the License itself.\n"
    )
    rows = list(logs_db["responses"].rows_where(select="response_json"))
    assert (
        len(json.loads(rows[0]["response_json"])) == 2
    )  # two response_jsons for tools
