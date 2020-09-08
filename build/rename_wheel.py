import argparse
from pathlib import Path

wheel_path = list(Path("dist").glob("bionic_model-*-none-any.whl"))
assert len(wheel_path) == 1
wheel_path = wheel_path[0]


def rename_wheel(os, cuda):
    wheel_name_components = wheel_path.stem.split("-")
    module_name, version = wheel_name_components[0], wheel_name_components[1]
    os_name = os.replace("-latest", "")
    target = Path(f"dist/{module_name}-{version}-py38-{cuda}-{os_name}.whl")
    wheel_path.rename(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--os", type=str)
    parser.add_argument("--cuda", type=str)

    args = parser.parse_args()
    rename_wheel(args.os, args.cuda)
