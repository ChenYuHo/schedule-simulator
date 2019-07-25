import torch
import torchvision.models as models
from torchvision.datasets import FakeData
import torch.optim as optim
import torch.nn as nn
import torchvision.transforms as transforms
import time
import json


def train(model, loss_func, optimizer, batch_size, num_of_batches):
    model.train()
    transform = transforms.Compose([transforms.ToTensor()])
    fake_data = FakeData(size=num_of_batches*batch_size, transform=transform, image_size=(3, 224, 224))
    trainloader = torch.utils.data.DataLoader(fake_data, batch_size=batch_size, num_workers=2)
    timer = time.time()
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model(inputs)
        loss = loss_func(outputs, labels)
        loss.backward()
        optimizer.step()
        if time.time() - timer >= 5:
            timer = time.time()
            print("Finished {:.2f}%".format((i+1)/num_of_batches*100))
    print("Finished training")


def init(module, input):
    global start_time
    start_times.append(time.time_ns())


def get_hook(index, type):
    def hook(module, input, output):
        layer_time_stamps[index]["{}_pass_ts".format(type)].append(time.time_ns())
    return hook


start_times = list()
layer_time_stamps = list()
layers = list()


def register_hooks(module):
    has_children = False
    if hasattr(module, "children"):
        for child in module.children():
            register_hooks(child)
            has_children = True
    if not has_children:
        module.register_forward_hook(get_hook(len(layers), "forward"))
        module.register_backward_hook(get_hook(len(layers), "backward"))
        layers.append(dict(type=type(module).__name__, forward_pass_cost=list(), backward_pass_cost=list()))
        layer_time_stamps.append(dict(forward_pass_ts=list(), backward_pass_ts=list()))


model = models.vgg16(pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
model.register_forward_pre_hook(init)
register_hooks(model)
x = torch.randn((1, 3, 224, 224), requires_grad=True)
with torch.autograd.profiler.profile() as prof:
    train(model=model, loss_func=criterion, optimizer=optimizer, batch_size=16, num_of_batches=32)

prof.export_chrome_trace("pytorch.chrometrace.json")

for batch_i in range(len(layer_time_stamps[0].values().__iter__().__next__())):
    last_ts = start_times[batch_i]
    for t in ["forward_pass", "backward_pass"]:
        for layer_i, layer_dict in enumerate(layers):
            if t == "backward_pass":
                layer_i = len(layers) - layer_i - 1
            next_ts = layer_time_stamps[layer_i]["{}_ts".format(t)][batch_i]
            layer_dict["{}_cost".format(t)].append(next_ts-last_ts)
            last_ts = next_ts

layer_costs = dict()
for layer_i, layer_dict in enumerate(layers):
    layer_costs["{}_{}".format(layer_i, layer_dict["type"])] = dict(forward_pass_units=layer_dict["forward_pass_cost"],
                                                                    backward_pass_units=layer_dict["backward_pass_cost"])
report = dict(args=dict(model=type(model).__name__, loss=str(criterion), optimizer=str(optimizer)), layer_costs=layer_costs)
with open("pytorch_profiling_reports/{}.profile.json".format(type(model).__name__), "w") as f:
    json.dump(report, f, indent=4)
