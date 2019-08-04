import numpy as np
from torchsummary import summary
import torch
from torchvision.datasets import FakeData
import torchvision.transforms as transforms
import time


def train(model, loss_func, optimizer, batch_size, num_of_batches, device, verbosity=1):
    """
    https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    """
    model.train()
    transform = transforms.Compose([transforms.ToTensor()])
    fake_data = FakeData(size=num_of_batches*batch_size, transform=transform,
                         image_size=get_standard_input_shape(model))
    trainloader = torch.utils.data.DataLoader(fake_data, batch_size=batch_size, num_workers=0)
    timer = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        if device == "gpu":
            inputs = inputs.to("cuda")
            labels = labels.to("cuda")
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        if type(model).__name__ == "Inception3":
            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
            outputs, aux_outputs = model(inputs)
            loss1 = loss_func(outputs, labels)
            loss2 = loss_func(aux_outputs, labels)
            loss = loss1 + 0.4 * loss2
        else:
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        if verbosity >= 1 and time.time() - timer >= 5:
            timer = time.time()
            print("Finished {:.2f}%".format((i+1)/num_of_batches*100))
    print("Finished training")


def traverse_module(module, processing_func, only_process_leafs=True):
    """
    Traverses the module in execution order. TODO confirm that module.children() gives us the execution order
    :param module: The module or model to traverse
    :param only_process_leafs: Whether we should only process the most inner modules (The layers).
    :param processing_func: The function which will be called on the modules. Should only have one required argument in
    which the module will be passed
    """
    has_children = False
    if hasattr(module, "children"):
        for child in module.children():
            traverse_module(child, processing_func)
            has_children = True
    if not (only_process_leafs and has_children):
        processing_func(module)


def count_trainable_params(module):
    model_parameters = filter(lambda p: p.requires_grad, module.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


def get_standard_input_shape(model):
    if type(model).__name__ == "Inception3":
        return 3, 299, 299
    else:
        return 3, 224, 224

