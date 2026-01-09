
from PIL import Image
from torch.utils.data import Dataset
class FruitFreshnessDataset(Dataset):
    def __init__(self, image_paths, fruit_lookup, freshness_lookup, transform=None):
        self.image_paths = image_paths
        self.fruit_lookup = fruit_lookup
        self.freshness_lookup = freshness_lookup
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)


        parts = path.split(os.sep)
        fruit = parts[-3]
        freshness = parts[-2]

        fruit_id = self.fruit_lookup[fruit]
        freshness_id = self.freshness_lookup[freshness]

        return image, (fruit_id, freshness_id)
