
import torchvision
import numpy as np

def __remap_labels(dataset, class_names):
    images = dataset.data
    labels = np.array(dataset.targets)
    
    # background = 2
    new_labels = np.full_like(labels, 2)
    
    # bicycle = 0
    bicycle_mask = labels == class_names.index('bicycle')
    new_labels[bicycle_mask] = 0
    
    # motorcycle = 1
    motorcycle_mask = labels == class_names.index('motorcycle')
    new_labels[motorcycle_mask] = 1
    
    return images, new_labels

def load_cifar():
    train_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True
    )

    class_names = train_dataset.classes

    train_images, train_labels = __remap_labels(train_dataset, class_names)
    test_images, test_labels = __remap_labels(test_dataset, class_names)

    return (train_images, train_labels), (test_images, test_labels)
