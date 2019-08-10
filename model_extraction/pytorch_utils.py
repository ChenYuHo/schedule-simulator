from abc import ABC, abstractmethod

import numpy as np
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.modules.loss as losses
import torch.nn.functional as F
import torchvision.transforms as transforms
import time


def train(model, loss_func, optimizer, batch_size, num_of_batches, device, verbosity=1):
    """
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    model.train()
    transform = transforms.Compose([transforms.ToTensor()])
    timer = time.time()
    for batch_i in range(num_of_batches):
        # 0. Generate fake data
        inputs, labels = get_dummy_input_output(model, batch_size, device)
        # zero the parameter gradients
        optimizer.zero_grad()
        # 1. forward
        outputs = model(*inputs)
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
        # 3. Backward
        loss.backward()
        # 4. Update weights
        optimizer.step()
        if verbosity >= 1 and time.time() - timer >= 5:
            timer = time.time()
            print("Finished {:.2f}%".format((batch_i+1)/num_of_batches*100))
    print("Finished training")


def traverse_module(module, processing_func, only_process_leafs=True):
    """
    Traverses the module in execution order. TODO confirm that module.children() gives us the execution order
    :param module: The module or model to traverse
    :param only_process_leafs: Whether we should only process the most inner modules (The layers).
    :param processing_func: The function which will be called on the modules. Should only have one required argument in
    which the module will be passed
    """
    is_parent = is_parent_module(module)
    if is_parent:
        for child in module.children():
            traverse_module(child, processing_func)
    if not (only_process_leafs and is_parent):
        processing_func(module)


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
            if model_name == "inception_v3":
                model = model(pretrained=False, aux_logits=False)
            else:
                model = model(pretrained=False)
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
