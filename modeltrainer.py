import copy
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
from tqdm import tqdm
from typing import Any

class ModelTrainer:
    def __init__(self, device: str, resnet_model: bool, loss_function):
        self.__device = device
        self.__resnet_model = resnet_model
        self.__loss_function = loss_function

    def train_model(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        size,
        num_epochs,
        dataloaders,
        dataset_sizes,
        verbose: bool=False
    ):
        start_time = time.time()

        best_model_wts = copy.deepcopy(model.state_dict())
        best_accuracy = 0.0
        best_loss = 100.0
        epochs_at_best_acc = 0
        epochs_at_best_loss = 0
        losses = [0] * num_epochs
        accuracies = [0] * num_epochs

        if verbose:
            self.__plot(losses, accuracies, num_epochs, best_loss, best_accuracy, size)

        for epoch in tqdm(range(num_epochs)):
            if verbose:
                print('Epoch {}/{}'.format(epoch + 1, num_epochs))
                print('-' * 10)

            # Training and validation per epoch
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(self.__device)
                labels = labels.to(self.__device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history if in training phase
                # with torch.set_grad_enabled(phase == 'train'):
                if self.__resnet_model:
                    model.fc.weight.requires_grad = (phase == 'train')
                    model.fc.bias.requires_grad = (phase == 'train')
                else:
                    for name, param in model.named_parameters():
                        if "6" in name and "classifier" in name:
                            param.requires_grad = (phase == 'train')

                # Check that all weights are frozen except the last layer if training
                if phase == 'train' and self.__resnet_model:
                    for name, param in model.named_parameters():
                        if param.requires_grad == True and "fc" not in name:
                            print("\n\n\nWeights before the last layer are not frozen\n\n\n")
                            exit(0)
                elif phase == 'train':
                    for name, param in model.named_parameters():
                        if (param.requires_grad == True) and not ("6" in name and "classifier" in name):
                            print("\n\n\nWeights before the last layer are not frozen\n\n\n")
                            exit(0)

                # Check that all weights are frozen if not training
                else:
                    for name, param in model.named_parameters():
                        if param.requires_grad == True:
                            print("\n\n\nWeights are not frozen during testing\n\n\n")
                            exit(0)
 
                outputs = model(inputs)
                _, preds = torch.max(input=outputs, dim=1)
                loss = criterion(outputs, labels)

                # Backward
                # Optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # Get accuracy
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            losses[epoch] = epoch_loss
            accuracies[epoch] = epoch_acc


            if verbose:
                print('{} Loss: {:.4f} Accuracy: {:.4f}'.format(
                    phase, epoch_loss, epoch_acc))
            
            # Deep copy the model
            if phase == 'val' and epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_at_best_acc = epoch

            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                epochs_at_best_loss = epoch

            print()

        self.__plot(losses, accuracies, num_epochs, best_loss, best_accuracy, size)
        time_elapsed = time.time() - start_time
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best validation Accuracy: {:4f}\n\n'.format(best_accuracy))

        # Load best model weights
        model.load_state_dict(best_model_wts)
        return model, best_accuracy, best_loss, epochs_at_best_acc, epochs_at_best_loss

    def __plot(self, losses, accuracies, for_epochs, best_loss, best_acc, size):
        epoch_range = np.arange(for_epochs)
        print(type(epoch_range))
        accuracy = []

        for result in accuracies:
            accuracy.append(torch.Tensor.cpu(result).item())
        plt.plot(epoch_range, accuracy, scaley=False)
        plt.xlabel('Epochs\nBest Loss: {:.4f}  Best Acc: {:.4f} Loss Fn: {}  Size: {:d}'.format(
                best_loss, best_acc, self.__loss_function, size))
        plt.ylabel('Accuracy & Loss')


        loss = []
        for result in losses:
            loss.append(result / 2.3)
        plt.plot(epoch_range, loss, scaley=False)
        plt.xlabel('Epochs\nBest Loss: {:.4f}  Best Acc: {:.4f} Loss Fn: {}  Size: {:d}'.format(
                best_loss, best_acc, self.__loss_function, size))
        plt.show()
        return