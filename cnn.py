# Convolutional Neural Network Architecture for Piece Classification

import torch
import torchvision
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
  transform_func = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.Grayscale(num_output_channels=1)
  ])

  dataset = torchvision.datasets.ImageFolder(root = "Final_Dataset", transform = transform_func)

  ex_img_tensor, ex_label = dataset.__getitem__(0)
  print(ex_img_tensor)
  print(f"Height:{len(ex_img_tensor[0])}")
  print(f"Width:{len(ex_img_tensor[0][0])}")
  print(ex_label)

  train_dataset, temp_set  = train_test_split(dataset, test_size = 0.2, random_state = 1)
  val_dataset, test_dataset = train_test_split(temp_set, test_size = 0.5, random_state = 1)
  print(len(train_dataset))
  print(len(val_dataset))
  print(len(test_dataset))

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 10, shuffle = True)
  val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 10, shuffle = True)
  test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 10, shuffle = True)

if torch.backends.mps.is_available():
  device = torch.device('mps')
elif torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')

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


cnn_model = CNN().to(device)

loss_func = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr = 0.0005)


def validate():
  cnn_model.eval()
  total_correct = 0.0
  total_pieces = 0.0
  with torch.no_grad():
    for batch, (features, piece_labels) in enumerate(val_dataloader):
      features = features.to(device).to(dtype = torch.float32)
      piece_labels = piece_labels.to(device).to(dtype = torch.long)
      total_pieces += len(piece_labels)
      pred = cnn_model(features)
      pred_labels = torch.argmax(pred, 1)
      total_correct += ((pred_labels == piece_labels).sum().item())
      #print(f"Total Correct Test batch {total_correct}")
    return total_correct / total_pieces

def train():
  cnn_model.train()
  #forward pass
  #calculate performance statistics
  # backwards pass
  total_correct = 0.0
  total_pieces = 0.0
  for batch, (features, piece_labels) in enumerate(train_dataloader):
    features = features.to(device).to(dtype = torch.float32)
    piece_labels = piece_labels.to(device).to(dtype = torch.long)
    total_pieces += len(piece_labels)

    pred = cnn_model(features)
    pred_labels = torch.argmax(pred, 1)
    #print(f"Predicted Labels {pred_labels}")
    total_correct += ((pred_labels == piece_labels).sum().item())

    #print(f"Total Correct Train Batch {total_correct}")
    loss = loss_func(pred, piece_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

  return total_correct / total_pieces

if __name__ == "__main__":
  print(cnn_model)
  epochs = 10
  for i in range(1, epochs + 1):
    print(f"Epoch {i}")
    train_accuracy = train()
    print(f"Train Accuracy: {train_accuracy}")
    val_accuracy = validate()
    print(f"Validation Accuracy: {val_accuracy}")
  torch.save(cnn_model.state_dict(), 'model_weights.pth')

  # TEST:
  def test():
    cnn_model.eval()
    total_correct = 0.0
    total_pieces = 0.0
    with torch.no_grad():
      for batch, (features, piece_labels) in enumerate(test_dataloader):
        features = features.to(device).to(dtype=torch.float32)
        piece_labels = piece_labels.to(device).to(dtype=torch.long)
        print(piece_labels)
        total_pieces += len(piece_labels)
        pred = cnn_model(features)
        pred_labels = torch.argmax(pred, 1)
        total_correct += ((pred_labels == piece_labels).sum().item())
        # print(f"Total Correct Test batch {total_correct}")
      return total_correct / total_pieces

  test_accuracy = test()
  print(f"FINAL ACCURACY: {test_accuracy}")

### PREDICTING A SINGLE IMAGE:
from PIL import Image


"""
def predict_image(model, device_param):
  # 1. Define the transformations
  # Must match your training data pipeline!
  transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=1),  # Match in_channels=1
    torchvision.transforms.Resize((224, 224)),  # Match training resolution
    torchvision.transforms.ToTensor(),  # Converts to [0, 1] range
  ])

  # 2. Load and transform
  img = Image.open(r"whiterook.png")
  img_tensor = transform(img)  # Shape: [1, 224, 224]
  img.show()

  # 3. Add Batch Dimension (unsquash)
  # Becomes Shape: [1, 1, 224, 224]
  img_tensor = img_tensor.unsqueeze(0).to(device_param)

  # 4. Inference
  model.eval()  # Set to evaluation mode
  with torch.no_grad():  # Disable gradient calculation
    output = model(img_tensor)
    prediction = torch.argmax(output, dim=1)

  return prediction.item()

# Usage:

if __name__ == "__main__":
  print(dataset.class_to_idx)
  result = predict_image(cnn_model, device)
  print(f"Predicted Class Index: {result}")

"""












