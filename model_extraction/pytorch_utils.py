from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.modules.loss as losses
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data
import time
import os


def train(model, loss_func, optimizer, batch_size, num_of_batches, device, verbosity=1, data_path=None):
    """
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    # Prepare data
    if data_path is not None:
        traindir = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
            ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=4, pin_memory=True, sampler=None)


    model.train()
    timer = time.time()

    iteration_costs = dict()
    for c in ["data_load", "resetting_optimizer", "forward_pass", "loss_calc", "backward_pass", "weight_update"]:
        iteration_costs["{}_units".format(c)] = list()

    for batch_i in range(num_of_batches):
        t1 = time.time_ns()
        # 1. Load data
        if data_path is not None:
            inputs, labels = next(iter(train_loader))
            if device == "gpu":
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            inputs = (inputs,)
            labels = (labels,)
        else:
            inputs, labels = get_dummy_input_output(model, batch_size, device)
        t2 = time.time_ns()
        iteration_costs["data_load_units"].append(t2-t1)
        t1 = t2
        # zero the parameter gradients
        optimizer.zero_grad()
        t2 = time.time_ns()
        iteration_costs["resetting_optimizer_units"].append(t2-t1)
        t1 = t2
        # 1. forward
        outputs = model(*inputs)
        t2 = time.time_ns()
        iteration_costs["forward_pass_units"].append(t2-t1)
        t1 = t2
        # if this is a single output just wrap it in a list for consistency
        if isinstance(outputs, torch.Tensor):
            outputs = [outputs]
        # 2. Loss
        loss = None
        for i, output in enumerate(outputs):
            if loss is None:
                loss = loss_func(output, labels[i])
            else:
                loss += loss_func(output, labels[i])
        t2 = time.time_ns()
        iteration_costs["loss_calc_units"].append(t2-t1)
        t1 = t2
        # 3. Backward
        loss.backward()
        t2 = time.time_ns()
        iteration_costs["backward_pass_units"].append(t2-t1)
        t1 = t2
        # 4. Update weights
        optimizer.step()
        t2 = time.time_ns()
        iteration_costs["weight_update_units"].append(t2-t1)
        t1 = t2
        if verbosity >= 1 and time.time() - timer >= 5:
            timer = time.time()
            print("Finished {:.2f}%".format((batch_i+1)/num_of_batches*100))
    print("Finished training")
    return iteration_costs


def traverse_module(module, processing_func, only_process_leafs=True):
    """
    Traverses the module in execution order. TODO confirm that module.children() gives us the execution order
    :param module: The module or model to traverse
    :param only_process_leafs: Whether we should only process the most inner modules (The layers).
    :param processing_func: The function which will be called on the modules. Should only have one required argument in
    which the module will be passed

    :return the number of modules processed by the given function
    """
    num_modules_processed=0
    is_parent = is_parent_module(module)
    if is_parent:
        for child in module.children():
            num_modules_processed += traverse_module(child, processing_func)
    if not (only_process_leafs and is_parent):
        num_modules_processed += processing_func(module)
    return num_modules_processed


def count_trainable_params(module):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = int(sum([np.prod(p.size()) for p in model_parameters]))
    return params


def get_standard_input_size(model):
    if type(model).__name__ == "Inception3":
        return (3, 299, 299),
    elif isinstance(model, DummyModel):
        return model.get_input_size()
    else:
        return (3, 224, 224),


def get_standard_output_size(model):
    if type(model).__name__ == "Inception3":
        if model.aux_logits:
            return (1000,), (1000,)
        else:
            return (1000,),
    elif isinstance(model, DummyModel):
        return model.get_output_size()
    else:
        return (1000,),


def get_dummy_input_output(model, batch_size, device="cpu"):
    if device == "gpu":
        device = "cuda"
    inputs = list()
    for size in get_standard_input_size(model):
        size = [batch_size] + list(size)
        inputs.append(torch.rand(size=size, device=device))
    labels = list()
    for size in get_standard_output_size(model):
        label = torch.randint(low=0, high=size[0], size=(batch_size,), device=device, dtype=torch.long)
        labels.append(label)
    return tuple(inputs), tuple(labels)


def get_module_name_from_op(torch_node, return_depth=False):
    scope = torch_node.scopeName()
    name = list()
    recording = False
    depth = 0
    for c in scope:
        if recording:
            if c == ']':
                recording = False
            else:
                name.append(c)
        else:
            if c == '[':
                depth += 1
                recording = True
                if len(name) > 0:
                    name.append(".")
    name = "".join(name)
    if return_depth:
        return name, depth
    else:
        return name


def get_module(model, module_name):
    if len(module_name) == 0:
        return model
    for name, module in model.named_modules():
        if name == module_name:
            return module


def get_module_name(model, module):
    for name, m in model.named_modules():
        if m == module:
            return name


def is_parent_module(module):
    return hasattr(module, "children") and len(list(module.children())) > 0


def get_model(model_name):
    import sys
    try:
        if model_name in dir(sys.modules[__name__]):
            module = sys.modules[__name__]
            model = getattr(module, model_name)()
        else:
            module = __import__("torchvision.models", fromlist=[model_name])
            model = getattr(module, model_name)
            if model_name == "inception_v3" or model_name == "googlenet":
                model = model(aux_logits=False, init_weights=True)
            else:
                model = model()
    except AttributeError:
        raise Exception("'{}' is not a valid dummy or torchvision model.\n".format(model_name))
    return model


class DummyModel(torch.nn.Module, ABC):
    @abstractmethod
    def get_input_size(self):
        pass

    @abstractmethod
    def get_output_size(self):
        pass


class Dummy2LayerModel(DummyModel):
    def get_input_size(self):
        return [[512]]

    def get_output_size(self):
        return [[1]]

    def __init__(self):
        super(Dummy2LayerModel, self).__init__()
        self.linear1 = nn.Linear(512, 16384)
        self.linear2 = nn.Linear(16384, 1)

    def forward(self, input):
        return self.linear2(self.linear1(input))

class DummyMultiModel(DummyModel):
    """
    The graph:
    C: Conv2D, R: ReLu, P: MaxPooling2D, L: Linear
                                          -> C -> R -> aux_out
    main_in -> C -> R -> P -> C -> R -> P
                                          ->
                                             L -> R -> L -> R -> main_out
                         aux_in -> L -> R ->
    """
    def get_input_size(self):
        return (3, 128, 128), (128,)

    def get_output_size(self):
        return (256,), (256,)

    def __init__(self):
        super(DummyMultiModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=8)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=8)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=8, stride=4)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4)
        self.relu3 = nn.ReLU(inplace=True)
        self.linear1 = nn.Linear(128, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(2560, 512)
        self.relu5 = nn.ReLU(inplace=True)
        self.linear3 = nn.Linear(512, 256)
        self.relu6 = nn.ReLU(inplace=True)

    def forward(self, *inputs):
        main_in, aux_in = inputs
        x = self.pool1(self.relu1(self.conv1(main_in)))
        x = self.pool2(self.relu2(self.conv2(x)))
        aux_out = self.relu3(self.conv3(x))
        aux_out = torch.flatten(aux_out, start_dim=1)
        # aux_out = aux_out.view(-1, np.prod(aux_out.size()[1:]))
        # x = x.view(-1, torch.prod(x.size()[1:]))  # Flatten
        x = torch.flatten(x, start_dim=1)
        # y = aux_in.view(-1, torch.prod(aux_in.size()[1:]))
        y = torch.flatten(aux_in, start_dim=1)
        y = self.relu4(self.linear1(y))
        main_out = torch.cat((x, y), dim=1)
        main_out = self.relu5(self.linear2(main_out))
        main_out = self.relu6(self.linear3(main_out))
        return main_out, aux_out
