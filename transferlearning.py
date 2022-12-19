from copy import deepcopy
from modeltrainer import ModelTrainer
import numpy as np
from transferlearningdataset import TransferLearningDataset
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

class TransferLearning:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        num_epochs: int,
        learning_rate: float,
        momentum: float,
        lr_decay_step_size: int,
        gamma: float,
        loss_fn,
        image_height: int,
        image_width: int,
        input_channels: int,
        resnet_model: bool,
        tuning: bool):

        self.__batch_size = batch_size
        self.__num_workers = num_workers
        self.__num_epochs = num_epochs
        self.__learning_rate = learning_rate
        self.__momentum = momentum
        self.__lr_decay_step_size = lr_decay_step_size
        self.__gamma = gamma
        self.__loss_fn = loss_fn
        self.__image_height = image_height
        self.__image_width = image_width
        self.__input_channels = input_channels
        self.__resnet_model = resnet_model
        self.__tuning = tuning
        self.__device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.__model_trainer = ModelTrainer(device=self.__device, resnet_model=self.__resnet_model, loss_function=self.__loss_fn)

    def __get_trainloader_for_size(self, size, trainset):
        image_index = 0
        label_index = 1

        counter = np.zeros(size)
        dataset = TransferLearningDataset([], [])

        for data in trainset:
            image = data[image_index]
            label = data[label_index]

            if counter[label] <= size:
                dataset.data.append(deepcopy(image))
                dataset.targets.append(deepcopy(label))
                counter[label] += 1

            if sum(counter) == size * 10:
                break

        trainloader = torch.utils.data.DataLoader(dataset, 
                                                batch_size=self.__batch_size, 
                                                shuffle=True, 
                                                num_workers=self.__num_workers)

        return trainloader

    def __make_model(self):
        if self.__resnet_model:
            model_ft = models.resnet50(pretrained=True)
            for __, param in model_ft.named_parameters():
                if param.requires_grad: 
                    param.requires_grad = False
            model_ft.fc.weight.requires_grad = True
            model_ft.fc.bias.requires_grad = True
        else:
            model_ft = models.vgg19(pretrained=True)
            for name, param in model_ft.named_parameters():
                param.requires_grad = False
            for name, param in model_ft.named_parameters():
                if "6" in name and "classifier" in name:
                    param.requires_grad = True
        
        return model_ft

    def run(self):
        train_transform =  transforms.Compose([
                transforms.Resize(self.__image_height),
                # transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        val_transform =  transforms.Compose([
                transforms.Resize(self.__image_height),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

        data_dir = './data'

        trainset = datasets.MNIST(data_dir, download=True, train=True,
                                        transform=train_transform)
        testset = datasets.MNIST(data_dir, download=True, train=False,
                                        transform=val_transform)
        image_datasets = {'train': trainset, 'val': testset}


        class_names = image_datasets['train'].classes

        testloader = torch.utils.data.DataLoader(testset, batch_size=self.__batch_size,
                                            shuffle=True, num_workers=self.__num_workers)

        if self.__tuning:
            train = [90]
        else:
            # train = [10, 30, 50, 70, 90]
            train = [10]

        sizes = len(train)
        best_error_at = [100] * sizes
        best_accuracy_at = [0] * sizes
        epochs_to_acc_pr_at = [0] * sizes
        epochs_to_err_pr_at = [0] * sizes

        for size in range (sizes):
            trainloader = self.__get_trainloader_for_size(train[size], trainset)
            dataloaders = {'train': trainloader, 'val': testloader}
            dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

            if self.__resnet_model:
                model_ft = self.__make_model()
                num_ftrs = model_ft.fc.in_features
                model_ft.fc = nn.Linear(num_ftrs, len(class_names))
            else:
                model_ft = self.__make_model()
                num_ftrs = model_ft.classifier[6].in_features
                model_ft.classifier[6] = nn.Linear(num_ftrs, len(class_names))

            model_ft = model_ft.to(self.__device)
        
            optimizer_ft = optim.SGD(
                filter(lambda p: p.requires_grad, model_ft.parameters()),
                lr=self.__learning_rate,
                momentum=self.__momentum)
        
            exp_lr_scheduler = lr_scheduler.StepLR(
                optimizer_ft,
                step_size=self.__lr_decay_step_size,
                gamma=self.__gamma)

            model_ft, best_accuracy_at[size], best_error_at[size], epochs_to_acc_pr_at[size], epochs_to_err_pr_at[size]  = self.__model_trainer.train_model(
                model_ft,
                self.__loss_fn,
                optimizer_ft,
                exp_lr_scheduler,
                train[size],
                num_epochs=self.__num_epochs,
                dataloaders=dataloaders,
                dataset_sizes=dataset_sizes)

            print("At size {:d} best error was {:4f} at epoch {:d}, and best accuracy was {:4f} at epoch {:d}\n\n".format(train[size], best_error_at[size],
                                                                                                                epochs_to_err_pr_at[size],
                                                                                                                best_accuracy_at[size],
                                                                                                                epochs_to_acc_pr_at[size]))

        print("[", end='')
        for result in best_accuracy_at:
            if result != torch.Tensor.cpu(best_accuracy_at[0]).item():
                    print(", ", end='')
            print(torch.Tensor.cpu(result).item(), end='')
        print("]")
        print(best_error_at)
        print(epochs_to_acc_pr_at)
        print(epochs_to_err_pr_at)


cudnn.benchmark = True
batch_size = 4
num_workers = 4
num_epochs = 5
learning_rate = 0.001
momentum = 0.9
lr_decay_step_size = 7
gamma = 0.1
loss_function = nn.CrossEntropyLoss()
image_height = 224
image_width = 224
input_channels = 3
resnet_model = False
tuning = False

TransferLearning(
    batch_size=batch_size,
    num_workers=num_workers,
    num_epochs=num_epochs,
    learning_rate=learning_rate,
    momentum=momentum,
    lr_decay_step_size=lr_decay_step_size,
    gamma=gamma,
    loss_fn=loss_function,
    image_height=image_height,
    image_width=image_width,
    input_channels=input_channels,
    resnet_model=resnet_model,
    tuning=tuning
).run()