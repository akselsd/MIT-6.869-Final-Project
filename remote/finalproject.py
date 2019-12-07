import matplotlib, numpy as np, matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import torchvision
from torchvision import datasets, models, transforms
import os   
from PIL import Image
from datetime import datetime

import optuna

from dataset import GazeboDataset, get_dataloaders

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
result_path = os.path.join(os.getcwd(), "Results", dt_string)
weights_path = os.path.join(result_path, "CVWeights")
os.mkdir(result_path)
os.mkdir(weights_path)


data_path = os.path.join(os.getcwd(), "CVData")

def get_model(num_out):
    model = models.resnet18(pretrained=True)
    input_size = 224
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_out)
    return model


def get_loss_func(alpha):

    def l(output, target):

        assert target.shape[1] == 3
        # Pos
        loss1 = F.mse_loss(output.view(target.shape)[:,0:2], target[:, 0:2])
        # Angle
        loss2 = F.mse_loss(output.view(target.shape)[:, 2], target[: , 2])

        return loss1 + alpha*loss2
    
    return l

def train(model, device, train_loader, optimizer, epoch, batch_size, loss_func):
    model.train()
    sum_loss = 0

    batches = tqdm(enumerate(train_loader), total=len(train_loader))
    batches.set_description("Epoch NA: Loss (NA) Accuracy (NA %)")

    for batch_idx, (data, target) in batches:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, target)
        sum_loss += loss.item() * batch_size
        loss.backward()
        optimizer.step()

        batches.set_description(
            "Epoch {:d}: Loss ({:.2e})".format(
            epoch, loss.item()))
        
    avg_loss = sum_loss / len(train_loader.dataset)


    
    return avg_loss

def test(model, device, test_loader, loss_func, batch_size):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            test_loss += loss_func(output, target).item() * batch_size # sum up batch loss
            
    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.2e}'.format(test_loss))
    return test_loss

def train_model(hypers, args, train_loader, val_loader, test_loader):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = get_model(args["out_features"])

    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=hypers["lr"], momentum=hypers["momentum"])

    train_loss_func = get_loss_func(hypers["alpha_loss"])
    test_loss_func = get_loss_func(0)

    training_losses = []
    validation_losses = []

    now = datetime.now()
    current_time = now.strftime("%d-%m_%H-%M-%S")
    loc = os.path.join(weights_path, current_time)
    os.mkdir(loc)

    for epoch in range(1, args["epochs"] + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch, args["batch_size"], train_loss_func)
        print("Average train loss:", train_loss)

        val_loss = test(model, device, val_loader, train_loss_func, args["batch_size"])

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        savefile = os.path.join(loc, "w" + str(epoch))
        torch.save(model.state_dict(), savefile)
        savefile = os.path.join(loc, "opt" + str(epoch))
        torch.save(optimizer.state_dict(), savefile)
    
    logger(now, hypers, opts, training_losses, validation_losses)

    return min(validation_losses)


def logger(now, hypers, opts, training_losses, validation_losses):
    l = ""

    l += str(min(validation_losses)) + ","

    for i in hypers.keys():
        l += i + ":" + str(hypers[i]) + ","
    
    for j in opts.keys():
        l += i + ":" + str(opts[j]) + ","
    
    l += "train:" + ";".join(training_losses) + ","
    l += "val:" + ";".join(validation_losses) + ","
    l+= "time:" + now

    with open(os.path.join(result_path, "-results.csv"), "a") as f:
        f.write(l + "\n")
            


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("Cuda available")
    else:
        print("Cant use cuda!")
    input_size = 224

    trans = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        
    opts = {}
    opts["batch_size"] = 32
    opts["epochs"] = 3
    opts["out_features"] = 3
    opts["preload"] = False

    static_dataset = GazeboDataset(data_path, trans, opts["preload"])

    def trainer(trial):
        hyper_grid = {
            "momentum" : trial.suggest_uniform("momentum", 0,1),
            "lr" : trial.suggest_loguniform("lr", 1e-6, 1e-3),
            "alpha_loss": trial.suggest_categorical("alpha_loss", [0])
        }
        print(hyper_grid)
        train, val, test = get_dataloaders(static_dataset, opts["batch_size"])
        
        loss = train_model(hyper_grid, opts, train, val, test)
        print(loss)
    
    study = optuna.create_study(direction="minimize")
    study.optimize(trainer, n_trials=5)



    
def validate_special(model, device, trans):
    model.eval()
    
    with torch.no_grad():
        for setnr in range(1, 2):
            fn = os.path.join(data_path, "test", "set" + str(setnr))
            ds = GazeboDataset(fn, trans, False)
            print("==== SET", setnr, "====")
            for i in range(len(ds)):
                data, device = ds[i]
                data, target = data.to(device), target.to(device)
                output = model(pic)
                print("Out:", output)
                print("Targ:", target)

