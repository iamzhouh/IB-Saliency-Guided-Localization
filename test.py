import torch
from tqdm import tqdm

def test(model, test_loader, device = "cpu"):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(test_loader, desc="Test Model"):
            data, label = data
            data, label = data.to(device), label.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, dim = 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('accuracy on test set: %.2f %% \n' % (100 * correct / total))
    return (round((100 * correct / total), 2))