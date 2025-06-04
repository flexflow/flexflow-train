import json
import os

import torch
import torch.nn as nn


def serialize_to_json_file(tensor: torch.Tensor, filepath: str) -> None:
    """
    Serialize a PyTorch tensor to a JSON file.

    Arg:
       tensor (torch.Tensor): The tensor to serialize.
       filepath (str): The path to the output JSON file.
    """
    with open(filepath, "w") as f:
        json.dump(tensor.tolist(), f, indent=4)


def get_output_file_path(filename: str) -> str:
    """
    Get the path to the output file where the serialized tensor will be saved.

    Args:
        filename (str): The name of the output file.

    Returns:
        str: The path to the output file.
    """
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return os.path.join(current_dir, filename)


def main() -> None:

    output_file_path = get_output_file_path("linear.json")

    linear = nn.Linear(3, 2)
    input = torch.randn(5, 3)
    output = linear(input)

    serialize_to_json_file(
        tensor=output,
        filepath=output_file_path,
    )


if __name__ == "__main__":
    main()
