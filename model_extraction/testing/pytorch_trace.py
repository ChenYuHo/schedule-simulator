from model_extraction.pytorch_utils import train
import torch
import torchvision
import torch.nn.modules.loss as losses
import torch.optim as optims
import torch.autograd.profiler as prof

device = "cpu"
model = torchvision.models.inception_v3(pretrained=False)
loss = losses.CrossEntropyLoss()
if device == "gpu":
    model.cuda()
    loss.cuda()
optim = optims.SGD(model.parameters(), lr=0.001)
# Run
with prof.profile(use_cuda=device == "gpu") as profiler:
    train(model=model, loss_func=loss, optimizer=optim, batch_size=8,
          num_of_batches=2, device=device, verbosity=1)

profiler.export_chrome_trace("pytorch_{}.chrometrace.json".format(device))