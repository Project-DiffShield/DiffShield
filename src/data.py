import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CelebAHQDataset(Dataset):
    def __init__(self, root_dir, image_size=512):
        """
        Args:
            root_dir (string): Directory with all the images.
            image_size (int): Size to resize the images to.
        """
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            # Typically normalize to [-1, 1] for diffusion models
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_name).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, img_name

def get_dataloader(root_dir, batch_size=4, image_size=512, shuffle=True):
    dataset = CelebAHQDataset(root_dir, image_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
