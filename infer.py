import torch

from lightning_regressor import Regressor
from data.dataset import RoutesDataset
from models.mlp import MLP

from data.utils import resize


def shift_pred_to_original_position(pred, original_len, resized_len):
    # padding was used
    if original_len < resized_len:
        pred = pred - (resized_len - original_len)

    # samples were uniformly removed
    elif original_len > resized_len:
        pred = pred * (original_len / resized_len)

    return pred


class DeepSlicer:
    def __init__(self, ckp_path: str, sample_size=600):
        self.regr = Regressor.load_from_checkpoint(ckp_path)
        self.regr.eval()

        self._sample_size = sample_size

        if isinstance(self.regr.model, MLP):
            self.flatten_data = True
        else:
            self.flatten_data = False

    def __call__(self, x, y):
        original_signal_len = len(x)
        x, y, _, _ = resize(x, y, 0, 0, target_len=self._sample_size)

        sample = torch.zeros(1, 2, self._sample_size)
        sample[0, 0] = torch.as_tensor(x)
        sample[0, 1] = torch.as_tensor(y)

        if self.flatten_data:
            sample = sample.flatten()

        with torch.no_grad():
            pred = self.regr(sample)[0]

        pred = shift_pred_to_original_position(pred, original_signal_len, self._sample_size)
        pred = pred.cpu().numpy().astype(int)

        return pred


def infer(ckp_path):
    model = Regressor.load_from_checkpoint(ckp_path)
    model.eval()

    test_set = RoutesDataset(200, 600, flatten_channels=False)

    x, y = test_set[0]
    x = torch.unsqueeze(torch.tensor(x), 0)
    y = torch.tensor(y)

    with torch.no_grad():
        print(model(x))


if __name__ == "__main__":
    net = DeepSlicer("flight-slicer/f849c8ca56674ea99af2715f8d08ed47/checkpoints/epoch=2-step=587.ckpt")
    length = 600
    x = torch.zeros(length)
    y = torch.zeros(length)

    print(net(x, y))
