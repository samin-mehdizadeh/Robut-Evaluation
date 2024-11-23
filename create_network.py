import torch
import numpy as np
import random
import argparse
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torchvision import models
from setup import *
from shifts import *
import pandas as pd

seed = 42
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def train(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracy = 100 * correct / total
        print(f'End of Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

def test(model, criterion,testloader,device):
  test_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
      for x,y in testloader:
          images, labels = x.to(device),y.to(device)
          outputs = model(images.float())
          loss = criterion(outputs, labels)
          test_loss += loss.item()
          _, predicted = torch.max(outputs, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      avg_test_loss = test_loss / len(testloader)
      accuracy = 100 * correct / total
  return accuracy,avg_test_loss

def get_train_file_name(args):
    if(args.train_file_path != None):
      return args.train_file_path
    output_file = args.train_dir + f"/{args.dataset}_{args.shift}"
    if('feature' in args.shift):
       output_file +=  f"_A{args.Astd}_B{args.Bstd}"
    if('label' in args.shift):
       output_file +=  f"_{args.label_coef}"
    if(args.col):
       output_file += f'col_low{str(args.low_coef_alpha)}_high{str(args.high_coef_alpha)}'
    if(args.res):
       output_file += f'res_low{str(args.low_coef_alpha)}_high{str(args.high_coef_alpha)}'
    output_file += f"_cnn_model.pth"
    return output_file

def get_user_data(args,num_classes,dataset,class_indices,my_model,criterion,num_samples_per_class):
    wass_dist = 0
    if('label' in args.shift):
        num_samples_per_class = determine_labels_per_class(num_classes, 1000, args.label_coef)
    selected_test_dataset = apply_label_shift(dataset,class_indices,num_samples_per_class)
    if('feature' in args.shift):
        selected_test_dataset = apply_feature_shift(selected_test_dataset,args.Astd,args.Bstd)
    if(args.adv):
        wass_dist,selected_test_dataset = apply_wass_adversary(my_model,criterion,selected_test_dataset,args.wass_alpha,args.device)
    if(args.res):
        res_coef = determine_resolution_color_coef(8, 1000, args.high_coef_alpha, args.low_coef_alpha)
        print(res_coef)
        selected_test_dataset = RandomResized(selected_test_dataset,res_coef)
    if(args.col):
        print("c")
        col_coef = determine_resolution_color_coef(8, 1000, args.high_coef_alpha, args.low_coef_alpha)
        print(col_coef)
        selected_test_dataset = RandomColorized(selected_test_dataset,col_coef)
    return selected_test_dataset,wass_dist


def train_network(args):
    total_lables_per_user = 1000
    output_file = get_train_file_name(args)
    trainset,_, input_channels, image_size, num_classes = get_dataset(args.dataset)
    if(args.res or args.col):
      my_model = models.resnet18(pretrained=False)  
      my_model.fc = nn.Linear(my_model.fc.in_features, num_classes) 
    else:
      my_model = CNN(input_channels=input_channels, num_classes=num_classes, image_size=image_size)
    my_model.to(args.device)
    optimizer = optim.SGD(my_model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    num = total_lables_per_user/num_classes
    samples_count = int(num) + (num>int(num))
    num_samples_per_class = [samples_count for i in range(num_classes)]
    dim = np.array(trainset[0][0].shape)
    all_datasets = []
    class_indices = defaultdict(list)
    for idx, (img, label) in enumerate(trainset):
        class_indices[label].append(idx)
    print("generating train dataset....")
    for i in range(0,args.num_users):
        selected_train_dataset,dist = get_user_data(args,num_classes,trainset,class_indices,my_model,criterion,num_samples_per_class)
        all_datasets.append(selected_train_dataset)

    print("concat users...")
    final_dataset = ConcatDataset(all_datasets)
    print(f"generated {len(final_dataset)} samples")
    train_loader = DataLoader(dataset=final_dataset, batch_size=100, shuffle=True)
    print("start training the model...")
    train(my_model, train_loader, criterion, optimizer, 5, args.device)
    torch.save(my_model.state_dict(), output_file)
    print(f'Model saved to {output_file}')
   
def get_test_file_name(args):
    if(args.test_file_path != None):
      return args.test_file_path
    output_file = args.test_dir 
    appendix = f"/{args.dataset}_{args.shift}"
    if('feature' in args.shift):
       appendix +=  f"_A{args.Astd}_B{args.Bstd}"
    if('label' in args.shift):
       appendix +=  f"_{args.label_coef}"
    output_file += appendix
    if(args.adv):
       output_file += "_adv"
    if(args.col):
       output_file += f'col_low{str(args.low_coef_alpha)}_high{str(args.high_coef_alpha)}'
    if(args.res):
       output_file += f'res_low{str(args.low_coef_alpha)}_high{str(args.high_coef_alpha)}'
    output_file += f".csv"
    return output_file

def test_network(args):
    total_lables_per_user = 1000
    output_file = get_test_file_name(args)
    model_path = get_train_file_name(args)
    _,testset, input_channels, image_size, num_classes = get_dataset(args.dataset)
    
    criterion = nn.CrossEntropyLoss()
    if(args.dataset == 'imagenet'):
      my_model = models.resnet18(pretrained=True).to(args.device)
    elif(args.res or args.col):
      my_model = models.resnet18(pretrained=False)  
      my_model.fc = nn.Linear(my_model.fc.in_features, num_classes) 
      print(f"model loaded from {model_path}")
      my_model.load_state_dict(torch.load(model_path, weights_only=True,map_location=args.device))
      my_model.to(args.device)
    else:
      my_model = CNN(input_channels=input_channels, num_classes=num_classes, image_size=image_size).to(args.device)
      print(f"model loaded from {model_path}")
      my_model.load_state_dict(torch.load(model_path, weights_only=True,map_location=args.device))
    my_model.eval() 

    num = total_lables_per_user/num_classes
    samples_count = int(num) + (num>int(num))
    num_samples_per_class = [samples_count for i in range(num_classes)]
    dim = np.array(testset[0][0].shape)
    print(dim)
    class_indices = defaultdict(list)
    print("generate class indices")
    for idx, (img, label) in enumerate(testset):
        class_indices[label].append(idx)
    if(args.adv):
      result = {"loss":[],"accuracy":[],"wass":[]}
    else:
      result = {"loss":[],"accuracy":[]}
    
    print("starting the loop")
    for i in range(0,args.num_users):
            selected_test_dataset,wass_dist = get_user_data(args,num_classes,testset,class_indices,my_model,criterion,num_samples_per_class)
            loader = DataLoader(selected_test_dataset, batch_size=254, shuffle=False,num_workers=2)
            acc,avg_loss = test(my_model,criterion,loader,args.device)
            if(args.adv):
                result["wass"].append(wass_dist)
            result["loss"].append(avg_loss)
            result["accuracy"].append(acc)
            print(f"{i} : {acc,avg_loss}")
            if(i%20 == 0):
                avgdf = pd.DataFrame(result)
                avgdf.to_csv(output_file, index=False)
    avgdf = pd.DataFrame(result)
    avgdf.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--shift",type = str,default="")
    parser.add_argument("--train_dir",type=str,default=f"/content/drive/MyDrive/PyTorch_CIFAR10-master/train/models")
    parser.add_argument("--test_dir",type=str)
    parser.add_argument("--num_users",type = int,default=500)
    parser.add_argument("--wass_alpha",type = float,default=10)
    parser.add_argument("--Astd",type = float,default=0.05)
    parser.add_argument("--Bstd",type=float,default=0.1)
    parser.add_argument('--adv', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument("--label_coef", type=float,default=0.8)
    parser.add_argument("--res",action='store_true')
    parser.add_argument("--col",action='store_true')
    parser.add_argument("--high_coef_alpha", type=tuple, default=(0.7, 1))
    parser.add_argument("--low_coef_alpha", type=tuple, default=(0.4, 0.7))
    parser.add_argument("--train_file_path",type=str,default=None)##optional
    parser.add_argument("--test_file_path",type=str,default=None)##optional
    parser.add_argument("--device", type=torch.device,default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    
    if(args.test):
       test_network(args)
    if(args.train):
       train_network(args)
