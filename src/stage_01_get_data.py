import argparse
import os
import shutil
import torch
from torchvision import datasets, transforms
import mlflow
import mlflow.pytorch
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model_utils import save_binary
import random


STAGE = "GET_DATA" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    logging.info("define training kwargs")
    train_kwargs = {"batch_size": config["params"]["BATCH_SIZE"]}
    test_kwargs = {"batch_size": config["params"]["TEST_BATCH_SIZE"]}

    logging.info("updating device configuration as per cuda availablity")
    device_config = {"DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu'}
    config.update(device_config)


    if config["DEVICE"] == "cuda":
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    
    logging.info("transforms added")
    transform = transforms.Compose(
    [transforms.ToTensor()]
    )
    
    logging.info("downloading training and testing data")
    train = datasets.MNIST(config["source_data_dirs"]["data"], train=True, download=True, transform=transform)
    test = datasets.MNIST(config["source_data_dirs"]["data"], train=False, download=True, transform=transform)

    logging.info("defining training and testing data loader")
    train_loader = torch.utils.data.DataLoader(train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test, **test_kwargs)
    
    artifacts = config["artifacts"]
    model_config_dir = os.path.join(artifacts["artifacts_dir"], artifacts["model_config_dir"])
    create_directories([model_config_dir])
    train_loader_bin_file = artifacts["train_loader_bin"]
    train_loader_bin_filepath = os.path.join(model_config_dir, train_loader_bin_file)
    test_loader_bin_file = artifacts["test_loader_bin"]
    test_loader_bin_filepath = os.path.join(model_config_dir, test_loader_bin_file)
    
    save_binary(train_loader, train_loader_bin_filepath)
    save_binary(test_loader, test_loader_bin_filepath)




if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e