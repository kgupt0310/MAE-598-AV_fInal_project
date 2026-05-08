import numpy as np


class PointCloudCorruption:
    def __init__(self):
        self.remove_prob = 0.15
        self.noise_std = 0.008
        self.num_outliers = 30

    def __call__(self, points, num_points):
        points = points.copy()

        noise = np.random.normal(0, self.noise_std, points.shape)
        corrupted = points + noise

        keep_mask = np.random.rand(len(corrupted)) > self.remove_prob
        corrupted = corrupted[keep_mask]

        if len(corrupted) == 0:
            corrupted = points.copy()

        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        outliers = np.random.uniform(mins, maxs, size=(self.num_outliers, 3))
        corrupted = np.concatenate([corrupted, outliers], axis=0)

        if len(corrupted) >= num_points:
            idx = np.random.choice(len(corrupted), num_points, replace=False)
            corrupted = corrupted[idx]
        else:
            idx = np.random.choice(
                len(corrupted),
                num_points - len(corrupted),
                replace=True,
            )
            corrupted = np.concatenate([corrupted, corrupted[idx]], axis=0)

        return corrupted.astype(np.float32)
