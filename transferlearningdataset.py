from torch.utils.data import Dataset

# from Haohan Jiang
class TransferLearningDataset(Dataset):
  def __init__(self, data, targets):
    self.data = data
    self.targets = targets

  def __getitem__(self, index):
    x = self.data[index]
    y = self.targets[index]
    return x, y

  def __len__(self):
    return len(self.data)