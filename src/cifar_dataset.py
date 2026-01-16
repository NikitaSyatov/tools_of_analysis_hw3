import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image

CLASS_NAMES = ['bicycle', 'motorcycle', 'background']
CLASS_MAPPING = {
    'bicycle': 0,
    'motorcycle': 1,
    'background': 2
}

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CifarDataset:
    def __init__(self, train=True, transform=None, bg_samples_per_class=200):
        self.train = train
        self.transform = transform
        self.bg_samples_per_class = bg_samples_per_class
        
        self.original_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=train, download=True
        )
        
        self.data = self.original_dataset.data  # numpy array shape (50000, 32, 32, 3) or (10000, 32, 32, 3)
        self.targets = np.array(self.original_dataset.targets)
        
        self.class_names = self.original_dataset.classes
        self.target_classes = {
            'bicycle': self.class_names.index('bicycle'),  # 8
            'motorcycle': self.class_names.index('motorcycle'),  # 48
        }
        
        self._prepare_data()
    
    def _prepare_data(self):
        self.images = []
        self.labels = []
        
        for class_name, class_idx in self.target_classes.items():
            mask = self.targets == class_idx
            class_data = self.data[mask]
            
            for img_array in class_data:
                img = Image.fromarray(img_array)
                self.images.append(img)
                self.labels.append(CLASS_MAPPING[class_name])
        
        background_indices = []
        for i in range(100):
            if i not in self.target_classes.values():
                background_indices.append(i)
        
        np.random.seed(2321)
        selected_bg_classes = np.random.choice(background_indices, 10, replace=False)
        
        bg_count = 0
        for bg_class in selected_bg_classes:
            mask = self.targets == bg_class
            class_data = self.data[mask]
            
            num_samples = min(len(class_data), self.bg_samples_per_class)
            indices = np.random.choice(len(class_data), num_samples, replace=False)
            
            for idx in indices:
                img = Image.fromarray(class_data[idx])
                self.images.append(img)
                self.labels.append(CLASS_MAPPING['background'])
                bg_count += 1
        
        print(f"Dataset prepared. Total samples: {len(self.images)}")
        print(f"Class distribution: Bicycle: {self.labels.count(0)}, "
              f"Motorcycle: {self.labels.count(1)}, "
              f"Background: {self.labels.count(2)}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label