import numpy as np
import torch
import random
from setup import CustomDataset
from torch.cuda import device_of
from torch.utils.data import Subset
from torchvision.transforms import Resize, Compose, ToPILImage, ToTensor
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

######## feature shift
def generateAandB(Astd,Bstd,dim):
  I = np.eye(dim[1])
  I = np.stack([I] * dim[0])
  A = np.random.normal(loc=0, scale=Astd, size=dim)
  A = A + I
  B = np.random.normal(loc=0, scale=Bstd, size=dim)
  print(np.sum(A),np.sum(B))
  return A,B

def transform_image(image,A,B):
  image = np.array(image)
  new_image = torch.from_numpy(np.matmul(A, image)+B)
  return new_image.float()

######## label shift
def determine_labels_per_class(num_classes, num_labels, alpha):
    dirichlet_distribution = np.random.dirichlet([alpha] * num_classes)
    labels_per_class = np.round(dirichlet_distribution * num_labels).astype(int)
    if(num_labels - labels_per_class[:-1].sum()<=0):
      labels_per_class[-1] = 0
    else:
      labels_per_class[-1] = num_labels - labels_per_class[:-1].sum()
    print(labels_per_class)
    return labels_per_class

###### wasserstein distance
def find_max_delta(my_model,criterion,image,label,alpha,device,learning_rate = 0.01):
  image = image.unsqueeze(0).to(device)
  #image.requires_grad = False
  label = torch.tensor([label]).to(device)
  delta = torch.zeros_like(image, requires_grad=True,device=device)
  for iteration in range(20): 
      adv_input = image + delta
      output = my_model(adv_input)
      delta_norm2 = torch.norm(delta, p=2)
      loss = criterion(output, label) - alpha*(delta_norm2**2)
      loss.backward()
      with torch.no_grad():
        grad = delta.grad
        grad_norm = torch.norm(grad, p=2)
        if(grad_norm != 0):
          normalized_grad = grad / (grad_norm)
          delta += learning_rate  * normalized_grad  # Avoid division by zero
        else:
          print(f"grad equals 0!{iteration}")
          delta.grad.zero_()
          break
          #normalized_grad = grad / (grad_norm + 1e-8)
        delta.grad.zero_()

  delta_optimized = delta.squeeze(0).cpu().detach().numpy()
  return delta_optimized

#########apply affine shifts
def apply_feature_shift(testset,Astd,Bstd):
  dim = np.array(testset[0][0]).shape
  A,B = generateAandB(Astd,Bstd,dim)
  testset_transformed = [(transform_image(image,A,B), label) for image, label in testset]
  return CustomDataset(testset_transformed)

def apply_label_shift(testset,class_indices, num_samples_per_class):
    subset_indices = []
    for cls, num_samples in enumerate(num_samples_per_class):
        if len(class_indices[cls]) < num_samples:
            print(f"Not enough samples for class {cls}. Available: {len(class_indices[cls])}, Requested: {num_samples}")
            num_samples = len(class_indices[cls])

        sampled_indices = random.choices(class_indices[cls], k=num_samples)
        subset_indices.extend(sampled_indices)

    subset_dataset = Subset(testset, subset_indices)
    return subset_dataset

def apply_wass_adversary(my_model,criterion,testset,alpha,device):
  testset_transformed = []
  distance = 0
  for image, label in testset:
    delta = find_max_delta(my_model,criterion, image,label,alpha,device)
    distance += np.linalg.norm(delta) ** 2
    new_image = image.numpy()
    testset_transformed.append((torch.from_numpy(new_image+delta),label))
  was_dist = np.sqrt(distance/len(testset_transformed))
  return was_dist,CustomDataset(testset_transformed)

#########apply color/resolution shift
def determine_resolution_color_coef(num_classes, num_labels, high_coef_alpha = (0.4,0.7) , low_coef_alpha = (0.7,1)):
    alphas = np.random.uniform(high_coef_alpha[0], high_coef_alpha[1], size=num_classes)
    for c in [i for i in range(num_classes//2)]:
      alphas[c] = np.random.uniform(low_coef_alpha[0], low_coef_alpha[1])

    dirichlet_distribution = np.random.dirichlet(alphas)
    #print(alphas)
    labels_per_class = np.round(dirichlet_distribution * num_labels).astype(int)
    if(num_labels - labels_per_class[:-1].sum()<=0):
      labels_per_class[-1] = 0
    else:
      labels_per_class[-1] = num_labels - labels_per_class[:-1].sum()
    
    return labels_per_class

class RandomResized(Dataset):
    def __init__(self, dataset, coefficients,resolutions = [14,16,18,20,26,28,30,32], final_size = (32,32)):
        self.dataset = dataset
        self.resolutions = resolutions
        self.transforms = [Compose([ToPILImage(), Resize(res), Resize(final_size), ToTensor()]) for res in resolutions]
        resolution_indices = []
        for idx, coeff in enumerate(coefficients):
            resolution_indices.extend([idx] * coeff)
        np.random.shuffle(resolution_indices)
        self.applied_transforms = [self.transforms[i] for i in resolution_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        transform = self.applied_transforms[idx]
        img = transform(img)
        return img, label

class GradualColorTransform:
    def __init__(self, intensity):
        self.intensity = intensity  

    def __call__(self, img):
        gray_img = transforms.functional.to_grayscale(img, num_output_channels=3)
        return Image.blend(gray_img, img, self.intensity)

class RandomColorized(Dataset):
    def __init__(self, dataset, coefficients, color_levels=[0, 0.14, 0.28, 0.42, 0.56, 0.7, 0.84, 1]):
        self.dataset = dataset
        self.color_levels = color_levels
        self.transforms = [Compose([ToPILImage(), GradualColorTransform(intensity), ToTensor()]) for intensity in color_levels]
        color_indices = []
        for idx, coeff in enumerate(coefficients):
            color_indices.extend([idx] * coeff)
        np.random.shuffle(color_indices)
        self.applied_transforms = [self.transforms[i] for i in color_indices]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        transform = self.applied_transforms[idx]
        img = transform(img)
        return img, label