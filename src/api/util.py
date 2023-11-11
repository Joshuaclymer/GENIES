import importlib.util
import pandas as pd
import json
import os
import subprocess
import time

IGNORE_INDEX = -100
current_file_path = os.path.abspath(__file__)
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
import shlex
import subprocess


def execute(cmd):
    cmd = shlex.split(cmd)
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)


def execute_command(cmd_str):
    """
    Execute a command and print its output in real-time using streams.

    Args:
    - cmd_str (str): The command as a string to execute.

    Returns:
    - None
    """

    # Example
    for path in execute(cmd_str):
        print(path, end="")


def execute_command_old(command):
    result = subprocess.run(command, shell=True, check=True, text=True)
    if result.stdout != None:
        print("Bash script output:", result.stdout)


def execute_script(path_to_script):
    result = subprocess.run(
        path_to_script,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        print("Error:", result.stderr)
    else:
        print("Bash script output:", result.stdout)


def save_json(data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    if not os.path.exists(file_path):
        raise ValueError(f"File {file_path} does not exist")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def is_main():
    if not "LOCAL_RANK" in os.environ:
        return True
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        return True
    return False


def print_once(string):
    if not "LOCAL_RANK" in os.environ:
        print(string)
        return
    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print(string)


def openai_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    return credentials["openai_api_key"]


def import_class(class_name, path_to_class_def):
    spec = importlib.util.spec_from_file_location("class_name", path_to_class_def)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def wandb_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    return credentials["wandb_api_key"]


def retry_on_failure(function, max_retries=5, silent=False, delay=0):
    for i in range(max_retries):
        try:
            return function()
        except Exception as e:
            if not silent:
                print(f"Failed with exception {e}. Retrying...")
            time.sleep(delay)
    raise Exception(f"Failed after {max_retries} retries.")


def load_jsonl(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            try:
                obj = json.loads(line.strip())
                data.append(obj)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON at line {len(data) + 1}: {str(e)}")
    return data

def save_as_csv(data, path):
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)