import os
import importlib.util
import json
import subprocess

IGNORE_INDEX = -100
current_file_path = os.path.abspath(__file__)
repo_path = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
accelerator = None


def execute_command(command):
    # Run the Bash script with arguments using subprocess
    try:
        # Use the shell=True argument to run the script in a shell
        result = subprocess.run(command, shell=True, check=True, text=True)
        print("Bash script output:", result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error running Bash script:", e)


def save_json(data, file_path):
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def import_module_from_path(path):
    spec = importlib.util.spec_from_file_location("module.name", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def print_once(string):
    if accelerator != None:
        if accelerator.is_main_process:
            print(string)
    else:
        print(string)


def openai_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    # return credentials["openai_api_key"]
    return credentials["openai_api_key2"]


def wandb_api_key():
    credentials = load_json(f"{repo_path}/configs/credentials.json")
    return credentials["wandb_api_key"]
