import torch
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from lightning.pytorch.callbacks import TQDMProgressBar


intercept = 12.545946747385665
# Coefficients from the regression model
coefficients = np.array([-1.20364237, -0.0850418307, -0.145285252, -0.00152442672, 0.00930735822, 0.0128988544, 7.7642916e-05, 0.000754992958, 0.000124434252])
# Initialize the PolynomialFeatures and LinearRegression model
poly = PolynomialFeatures(degree=2, include_bias=False)
model = LinearRegression()
# Manually set the coefficients and intercept
model.intercept_ = intercept
model.coef_ = coefficients
# Available memory in GB

def predict_memory_usage(batch_size, n_protein_nodes, n_compound_nodes, model=model, poly=poly):
    X = np.array([[batch_size, n_protein_nodes, n_compound_nodes]])
    X_poly = poly.fit_transform(X)
    y = model.predict(X_poly)[0]
    return y


class TankBindSampler(torch.utils.data.sampler.Sampler):
    def __init__(
        self,
        dataset,
        max_num: int=1024,
        mode: str = 'node',
        shuffle: bool = False,
        skip_too_big: bool = False,
        num_steps=4000,
        available_memory_gb=30,
    ):
        if max_num <= 0:
            raise ValueError(f"`max_num` should be a positive integer value "
                             f"(got {max_num})")
        if mode not in ['node', 'edge']:
            raise ValueError(f"`mode` choice should be either "
                             f"'node' or 'edge' (got '{mode}')")

        self.dataset = dataset
        self.max_num = max_num
        self.mode = mode
        self.shuffle = shuffle
        self.skip_too_big = skip_too_big
        self.num_steps = num_steps
        self.max_steps = num_steps or len(dataset)
        self.available_memory_gb = available_memory_gb

    def __iter__(self):
        if self.shuffle:
            indices = torch.randperm(len(self.dataset)).tolist()
        else:
            indices = range(len(self.dataset))

        samples = []
        num_steps = 0
        num_processed = 0

        while (num_processed < len(self.dataset)
               and num_steps < self.max_steps):
            current_max_protein_nodes = 0
            current_max_compound_nodes = 0
            batch_size = 0
            for i in indices[num_processed:]:
                data = self.dataset[i]
                num_protein_nodes = (data["protein"].node_scalar_features.shape[0]//8+1)*8
                num_compound_nodes = (data["compound"].x.shape[0]//8+1)*8
                potential_new_max_protein_nodes = max(current_max_protein_nodes, num_protein_nodes)
                potential_new_max_compound_nodes = max(current_max_compound_nodes, num_compound_nodes)
                if (mem:=predict_memory_usage(batch_size + 1, potential_new_max_protein_nodes, potential_new_max_compound_nodes, model, poly)) < self.available_memory_gb and batch_size < self.max_num:
                    samples.append(i)
                    num_processed += 1
                    batch_size += 1
                    current_max_protein_nodes = potential_new_max_protein_nodes
                    current_max_compound_nodes = potential_new_max_compound_nodes
                else:
                    break
                
            if samples:
                yield samples
            samples = []
            num_steps += 1

    def __len__(self) -> int:
        if self.num_steps is None:
            raise ValueError(f"The length of '{self.__class__.__name__}' is "
                             f"undefined since the number of steps per epoch "
                             f"is ambiguous. Either specify `num_steps` or "
                             f"use a static batch sampler.")

        return self.num_steps
    
class DynamicSamplerProgressBar(TQDMProgressBar):
    def __init__(self, total_train_samples):
        super().__init__()
        self.total_train_samples = total_train_samples
    def on_train_epoch_start(self, trainer, *_):
        self.train_progress_bar.reset(self.total_train_samples)
        self.train_progress_bar.initial = 0
        self.train_progress_bar.set_description(f"Epoch {trainer.current_epoch}")

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        new_n = self.train_progress_bar.n + batch.batch_size
        if self._should_update(new_n, self.train_progress_bar.total):
            _update_n(self.train_progress_bar, new_n)
            self.train_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))


def _update_n(bar, value) -> None:
    if not bar.disable:
        bar.n = value
        bar.refresh()