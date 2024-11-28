import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import random
from torch import nn
from torchvision import datasets, models


class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        # Load pre-trained DenseNet161
        self.densenet = models.densenet161(pretrained=True)
        num_ftrs = self.densenet.classifier.in_features
        # Replace classifier to adapt to 100 types of tasks
        self.densenet.classifier = nn.Linear(num_ftrs, 100)

    def forward(self, x):
        return self.densenet(x)

# Merge two images
def merge_images(cifar_img, folder_img, ratio=0.7):
    return ratio * cifar_img + (1 - ratio) * folder_img


# calculate cross-entropy loss
def calculate_loss(surrogate_model, images, device, target_class=0):
    surrogate_model.eval()
    inputs = torch.stack([img.to(device) for img in images])
    outputs = surrogate_model(inputs)
    targets = torch.full((outputs.size(0),), target_class, dtype=torch.long, device=device)
    loss = F.cross_entropy(outputs, targets)
    return loss.item()


# Load index images from the CIFAR-100 dataset
def load_cifar100_indices(filepath):
    with open(filepath, 'r') as f:
        indices = [int(line.strip()) for line in f.readlines()]

    # Load CIFAR-100 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    cifar_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

    cifar_images = [cifar_dataset[i][0] for i in indices]
    return cifar_images


# Load images from folder
def load_folder_images(folder_path):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])

    folder_images = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = transform(img)
            folder_images.append(img)
    return folder_images


# Save selected images to a specified folder
def save_selected_images(images, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, img in enumerate(images):
        img = transforms.ToPILImage()(img)  # Convert back to PIL format
        img.save(os.path.join(output_folder, f'selected_image_{i}.png'))

# Save images into train/test subfolders based on a split ratio
def save_images_with_split(images, output_folder, train_ratio=0.8):
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    random.shuffle(images)
    split_index = int(len(images) * train_ratio)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for i, img in enumerate(train_images):
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(train_folder, f'image_{i}.png'))

    for i, img in enumerate(test_images):
        img = transforms.ToPILImage()(img)
        img.save(os.path.join(test_folder, f'image_{i}.png'))

# PSO fitness function
def fitness_function(particle, cifar_images, folder_images, surrogate_model, device, target_class=0):
    selected_images = [folder_images[int(i)] for i in particle]
    merged_images = [merge_images(cifar_img, random.choice(selected_images)) for cifar_img in cifar_images]
    loss = calculate_loss(surrogate_model, merged_images, device, target_class)
    print(f'Current Loss: {loss}')
    return -loss


# PSO Algorithm
def pso(cifar_images, folder_images, surrogate_model, device, target_class=0, num_particles=30, max_iter=100, w=0.5,
        c1=1.5, c2=1.5):
    dim = 50  # select 50 images
    lb, ub = 0, len(folder_images) - 1  # Index upper and lower limits

    # Initialize Particle Swarm Optimization
    swarm = np.random.randint(lb, ub, (num_particles, dim))

    # Ensure that the selected image index is unique
    for i in range(num_particles):
        swarm[i] = np.random.choice(range(lb, ub + 1), size=dim, replace=False)

    velocities = np.random.uniform(-1, 1, (num_particles, dim))
    personal_best = np.copy(swarm)
    personal_best_fitness = np.full(num_particles, -np.inf)

    global_best = None
    global_best_fitness = -np.inf

    for iter in range(max_iter):
        for i in range(num_particles):
            # Calculate fitness
            fitness = fitness_function(swarm[i], cifar_images, folder_images, surrogate_model, device, target_class)

            # Update personal best
            if fitness > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best[i] = np.copy(swarm[i])

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best = np.copy(swarm[i])

        # Update particle velocity and position
        for i in range(num_particles):
            r1, r2 = np.random.rand(dim), np.random.rand(dim)
            velocities[i] = w * velocities[i] + c1 * r1 * (personal_best[i] - swarm[i]) + c2 * r2 * (
                    global_best - swarm[i])
            swarm[i] = swarm[i] + velocities[i]

            # Ensure that the particle position is within the range
            swarm[i] = np.clip(swarm[i], lb, ub)  # Ensure that the particle position is within the range

            # Remove duplicate indexes
            unique_indices = np.unique(swarm[i])
            if len(unique_indices) < dim:
                # If the quantity is insufficient after deduplication, randomly choose to supplement
                missing_count = dim - len(unique_indices)
                additional_indices = np.random.choice(
                    [idx for idx in range(lb, ub + 1) if idx not in unique_indices],
                    size=missing_count,
                    replace=False
                )
                swarm[i] = np.concatenate((unique_indices, additional_indices))
            else:
                swarm[i] = unique_indices[:dim]  # Ensure the size is dim

        print(f'Iteration {iter + 1}/{max_iter}, Best Fitness: {-global_best_fitness}')

    # Return to the optimal individual
    return [int(i) for i in global_best]


# Add noise of different pixels to the image
def add_pixel_noise(images, noise_matrices):
    noisy_images = []
    for img, noise_matrix in zip(images, noise_matrices):
        noise = torch.tensor(noise_matrix, dtype=img.dtype, device=img.device)  # Convert the noise matrix to a type consistent with the image
        noisy_img = img + noise
        noisy_img = torch.clamp(noisy_img, 0, 1)  # Ensure that pixel values are between 0-1
        noisy_images.append(noisy_img)
    return noisy_images

# PSO optimized noise (different noise for each pixel)
def optimize_pixel_noise(cifar_images, selected_images, surrogate_model, device, num_particles=30, max_iter=150, w=0.5,
                         c1=1.5, c2=1.5):

    img_shape = selected_images[0].shape
    dim = img_shape[0] * img_shape[1] * img_shape[2]  # The total number of pixels in each image

    lb, ub = -0.04, 0.04  # Boundary of each element in the noise matrix

    # Initialize Particle Swarm Optimization
    swarm = np.random.uniform(lb, ub, (num_particles, len(selected_images), dim))  # Each particle has len (selected_images) * dim noise values
    velocities = np.random.uniform(-0.01, 0.01, (num_particles, len(selected_images), dim))
    personal_best = np.copy(swarm)
    personal_best_fitness = np.full(num_particles, -np.inf)

    global_best = None
    global_best_fitness = -np.inf

    for iter in range(max_iter):
        for i in range(num_particles):
            noise_matrices = swarm[i].reshape(len(selected_images), *img_shape)  # Flatten noise into a matrix of image size
            noisy_images = add_pixel_noise(selected_images, noise_matrices)

            # Merge noisy images with CIFAR-100 images
            merged_images = [merge_images(cifar_img, noisy_img) for cifar_img, noisy_img in zip(cifar_images, noisy_images)]

            # Calculate fitness (negative loss)
            fitness = -calculate_loss(surrogate_model, merged_images, device)

            # Update personal best
            if fitness > personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best[i] = np.copy(swarm[i])

            # Update global optimum
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best = np.copy(swarm[i])

        # Update speed and location
        for i in range(num_particles):
            r1, r2 = np.random.rand(len(selected_images), dim), np.random.rand(len(selected_images), dim)
            velocities[i] = (w * velocities[i]
                             + c1 * r1 * (personal_best[i] - swarm[i])
                             + c2 * r2 * (global_best - swarm[i]))
            swarm[i] = swarm[i] + velocities[i]
            swarm[i] = np.clip(swarm[i], lb, ub)  # Ensure that the noise level is within the range

        print(f'Iteration {iter + 1}/{max_iter}, Best Fitness: {-global_best_fitness}')

    # Return the optimal noise matrix
    return global_best.reshape(len(selected_images), *img_shape)


# main function
def main():
    cifar100_path = './cifar100_0.05.txt'
    folder_path = './panda600'
    output_folder_before = './output/cifar100_densenet_i100/selected_images'
    output_folder_after = './output/cifar100_densenet_i100/optimized_images'

    cifar_images = load_cifar100_indices(cifar100_path)
    folder_images = load_folder_images(folder_path)

    # Load surrogate model
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    surrogate_model = models.densenet161(pretrained=True)
    surrogate_model.classifier = nn.Linear(2208, 100)  # CIFAR-100 has 100 classes
    surrogate_model = surrogate_model.to(device)
    surrogate_model.load_state_dict(torch.load('surrogate_densenet161_cifar100.pth'))
    # PSO optimization to select 50 images
    selected_images_indices = pso(cifar_images, folder_images, surrogate_model, device)
    selected_images = [folder_images[i] for i in selected_images_indices]

    # Save the original selected images
    save_selected_images(selected_images, output_folder_before)

    # PSO optimization to add noise to selected images
    optimal_noise_level = optimize_pixel_noise(cifar_images, selected_images, surrogate_model, device)

    # Add optimal noise to selected images
    optimized_images = add_pixel_noise(selected_images, optimal_noise_level)

    # Save the optimized images with a train/test split
    save_images_with_split(optimized_images, output_folder_after, train_ratio=0.8)

    print(f'Selected images saved in {output_folder_before}')
    print(f'Optimized images with noise saved in {output_folder_after}/train and {output_folder_after}/test')

    print(f'Selected images saved in {output_folder_before}')
    print(f'Optimized images with noise saved in {output_folder_after}')


if __name__ == '__main__':
    main()