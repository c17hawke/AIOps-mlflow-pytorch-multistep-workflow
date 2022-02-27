import argparse
import os
import shutil
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model_utils import ConvNet, load_binary
import random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import mlflow
import mlflow.pytorch

STAGE = "TRAINING" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )

def train_(config, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pred = model.forward(data)
        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        if batch_idx % config["params"]["LOG_INTERVAL"] == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100.0 * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}")


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    device_config = {"DEVICE": 'cuda' if torch.cuda.is_available() else 'cpu'}
    config.update(device_config)

    model = ConvNet().to(config["DEVICE"])
    scripted_model = torch.jit.script(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["params"]["LR"])

    scheduler = StepLR(optimizer, step_size=1, gamma=config["params"]["GAMMA"])



    artifacts = config["artifacts"]
    model_config_dir = os.path.join(artifacts["artifacts_dir"], artifacts["model_config_dir"])
    train_loader_bin_file = artifacts["train_loader_bin"]
    train_loader_bin_filepath = os.path.join(model_config_dir, train_loader_bin_file)
    train_loader = load_binary(train_loader_bin_filepath)


    for epoch in range(1, config["params"]["EPOCHS"] + 1):
        train_(config, scripted_model, config["DEVICE"], train_loader, optimizer, epoch)
        scheduler.step()
    
    with mlflow.start_run(run_name="training") as run:
        mlflow.log_params(config["params"])
        mlflow.pytorch.log_model(model, "model")

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e