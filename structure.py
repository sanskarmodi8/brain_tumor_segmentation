import logging
import os
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

project_name = "BrainTumorSegmentation"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_preprocessing.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/prediction.py",
    f"src/{project_name}/pipeline/stage_01_data_preprocessing.py",
    f"src/{project_name}/pipeline/stage_02_model_training.py",
    f"src/{project_name}/pipeline/stage_03_model_evaluation.py",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",
    f"src/{project_name}/constants/__init__.py",
    "config/config.yaml",
    "main.py",
    "params.yaml",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "dvc.yaml",
    "trials/notebook.ipynb",
    "app.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    # create the directory if not exists
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir} for the file: {filename}")

    # create the file if not exists
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            logging.info(f"Created empty file: {filepath}")

    else:
        logging.info(f"File {filename} already exists")
