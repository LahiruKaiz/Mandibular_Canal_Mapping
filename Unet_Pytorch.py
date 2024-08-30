import torch
from torch import nn
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import torchvision.transforms.functional as TF


def save_checkpoint(model, optimizer, epoch, loss, batch, acc, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'batch': batch,
        'acc': acc
    }
    torch.save(checkpoint, filepath)

#Setup the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

class CustomDataset(Dataset):

    def __init__(self, dir):
        images = []
        masks = []
        for root, dirs, files in os.walk(dir):
            if not dirs:
                for file in files:
                    if root.split('/')[-1] == 'images':
                        images.append(os.path.join(root, file))

                    elif root.split('/')[-1] == 'masks':
                        masks.append(os.path.join(root, file))

        self.image_paths = images
        self.mask_paths = masks

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = np.load(self.image_paths[index])['data']
        image = image.astype(np.float32)
        mask = np.load(self.mask_paths[index])['data']
        mask = mask.astype(np.float32)

        image = torch.from_numpy(image).float().unsqueeze(dim =0)
        mask = torch.from_numpy(mask).float().unsqueeze(dim =0)


        return image/1000, torch.round(mask)
    

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=1, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)
        # output = torch.sigmoid(self.final_conv(x))

        return self.final_conv(x)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("Loading model")
model = UNET()
model.to(device)

def MeanDice(y_pred, y_true):
    return torch.mean((2 * (torch.sum(y_pred * y_true) / (torch.sum(y_pred) + torch.sum(y_true)))))

def MeanDiceLoss(y_pred, y_true):
    return 1 - torch.mean((2 * (torch.sum(y_pred * y_true) / (torch.sum(y_pred) + torch.sum(y_true)))))

def ModDiceLoss(y_pred, y_true):
    batch_size, _, _, _ = y_pred.shape
    loss = torch.tensor(0.)
    for i in range(batch_size):
        if torch.sum(y_true) == 0 or torch.sum(y_pred) == 0:
            y_t = y_true[i]*(-1) + 1
            y_p = y_pred[i]*(-1) + 1
            loss = torch.add(loss, (1 - torch.div(torch.mul(2, torch.sum(y_t * y_p)), (torch.sum(y_p) + torch.sum(y_t)))))
        else:
            y_t = y_true[i]
            y_p = y_pred[i]
            loss = torch.add(loss, (1 - torch.div(torch.mul(2, torch.sum(y_t * y_p)), (torch.sum(y_p) + torch.sum(y_t)))))

    return torch.div(loss, batch_size)

loss_fn = nn.BCEWithLogitsLoss()


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               optimizer: torch.optim.Optimizer,
               start_batch = 0,
               train_loss = 0,
               train_acc= 0, epoch = 0):
    
    version = 0
    
    # Put model in train mode
    model.train()

    length = len(dataloader)
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader, start= start_batch):
        X, y = X.to(device), torch.round(y).to(device)
        # print(batch)
        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()
        y_pred = torch.round(torch.sigmoid(y_pred))
        # Calculate and accumulate accuracy metric across all batches
        a = MeanDice(y_pred, y)
        train_acc += a
        
        print(f"{batch+1}/{length}  Loss: {train_loss/(batch+1)} Dice: {train_acc/(batch+1)}", end = '\r')
        
        if (batch+1)%2 == 0:
            version += 1
            checkpoint = {
                'epoch': epoch,
                'batch': batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
                'acc':train_acc
            }
            torch.save(checkpoint, f'/media/xavier/New Volume/checkP/checkpoint_{version}.pth')
        
    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def val_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader):
    # Put model in eval mode
    model.eval()

    # Setup test loss and test accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), torch.round(y).to(device)

            # 1. Forward pass
            val_pred = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(val_pred, y)
            val_loss += loss.item()
            val_pred = torch.round(torch.sigmoid(val_pred))
            # Calculate and accumulate accuracy
            val_acc += MeanDice(val_pred, y)

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          epochs: int = 5):
    
    start_batch = 0
    loss = 0
    acc= 0
    epoch = 0
    
    names = []
        
    for root, dirs, filenames in os.walk('/media/xavier/New Volume/checkP/'):
        names = filenames
        break
        
        
    version_list = [int(f.split('.')[0].split('_')[1]) for f in names]
    version_list = sorted(version_list, reverse=True)
        
    for item in version_list:
        try:
            checkpoint = torch.load(f'/media/xavier/New Volume/checkP/checkpoint_{item}.pth')
            os.system("cp '/media/xavier/New Volume/checkP/checkpoint_"+str(item)+".pth' '/media/xavier/New Volume/complete/latest.pth' ")
            break
        except:
            pass
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        loss = checkpoint['loss']
        acc = checkpoint['acc']
        print(f"Resuming training from epoch {epoch + 1}, batch {start_batch + 1}")
        
        for f in names:
            os.remove(f"/media/xavier/New Volume/checkP/{f}")
        
        print('Check points deleted')
    except UnboundLocalError:
        print("No checkpoint found, starting training from scratch.")


    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           optimizer=optimizer,
                                           start_batch= start_batch,
                                           train_loss= loss,
                                           train_acc= 0,
                                           epoch= epoch)
        val_loss, val_acc = val_step(model=model,
            dataloader=val_dataloader)

        # 4. Print out what's happening
        print(f"Epoch: {epoch+1} | train_loss: {train_loss:.8f} , train_acc: {train_acc:.8f} | val_loss: {val_loss:.8f} , val_dice: {val_acc:.8f}")
        torch.save(obj= model.state_dict(), f = '/media/xavier/New Volume/checkP/test.pth')




print('Loading Train Dataset')
train_dataset = CustomDataset("/media/xavier/New Volume/TF_data/train")
print('Loading Val Dataset')
val_dataset = CustomDataset("/media/xavier/New Volume/TF_data/val")

BATCH_SIZE = 10
print(f"Batch size: {BATCH_SIZE}")

train_dataloader = DataLoader(dataset= train_dataset, batch_size= BATCH_SIZE, shuffle= True)
val_dataloader = DataLoader(dataset= val_dataset, batch_size= 1, shuffle= False)

torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Set number of epochs
NUM_EPOCHS = 50
learning_rate = 0.001


optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# Train model_0
model_0_results = train(model=model,
                        train_dataloader= train_dataloader,
                        val_dataloader= val_dataloader,
                        optimizer=optimizer,
                        epochs=NUM_EPOCHS)
