from torch.utils.data import Dataset, DataLoader

from train import train_model 

if __name__ == "__main__":
    print("Training model...")
    model = train_model(num_epochs=10, batch_size=32)
