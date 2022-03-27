import torch

from lightning_regressor import Regressor
from data.dataset import RoutesDataset

def infer():
    ckp_path = "flight-slicer/f849c8ca56674ea99af2715f8d08ed47/checkpoints/epoch=2-step=587.ckpt"

    model = Regressor.load_from_checkpoint(ckp_path)
    model.eval()

    test_set = RoutesDataset(200, 600, flatten_channels=False)

    x, y = test_set[0]
    x = torch.unsqueeze(torch.tensor(x), 0)
    y = torch.tensor(y)

    with torch.no_grad():
        print(model(x))

if __name__ == "__main__":
    infer()