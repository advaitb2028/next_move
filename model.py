class CNN(torch.nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(in_channels = 1, out_channels = 64 , kernel_size = 3, padding = 1)
    self.conv2 = torch.nn.Conv2d(in_channels= 64, out_channels= 128, kernel_size=3, padding=1)
    self.output = torch.nn.LazyLinear(13)
    self.activation = torch.nn.ReLU()
    self.Flatten = torch.nn.Flatten()

  def forward(self, x):
    x = self.conv1(x)
    x = self.activation(x)
    x = self.conv2(x)
    x = self.activation(x)
    x = self.Flatten(x)
    x = self.output(x)
    return x
