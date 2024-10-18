import numpy as np


class TimeSeriesDataset:
    def __init__(self, name, size, error_perc=None, error_std=None):
        self.load_dataset_(name, size, error_perc, error_std)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return self.segments[index], self.labels[index]

    def load_dataset_(self, name, size, error_perc, error_std):
        data = None
        if error_perc is None or error_std is None:
            data = np.load(f"data/generated/{name}_{size}.npz")
        else:
            data = np.load(
                f"data/gaussian/{name}/{name}_{size}_{error_perc}_{error_std}.npz"
            )
        segments_array = data["segments_array"]
        original_labels = data["original_labels"]
        cutpoints = data["cutpoints"]

        self.segments = [
            segments_array[cutpoints[i] : cutpoints[i + 1]]
            for i in range(len(cutpoints) - 1)
        ]
        self.labels = original_labels
