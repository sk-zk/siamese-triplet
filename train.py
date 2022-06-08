from argparse import Namespace, ArgumentParser
import os
from PIL import Image
import timm
import torch
from torch.utils.data import Dataset
from torch.optim import lr_scheduler
import torch.optim as optim
from torchvision import transforms

from trainer import fit
from losses import OnlineTripletLoss
from utils import AllTripletSelector, HardestNegativeTripletSelector, RandomNegativeTripletSelector, \
    SemihardNegativeTripletSelector  # Strategies for selecting triplets within a minibatch
from metrics import AverageNonzeroTripletsMetric
from datasets import BalancedBatchSampler

cuda = torch.cuda.is_available()


class TripletDataset(Dataset):
    def __init__(self, img_dir, input_size, is_train=True):
        self.img_dir = img_dir
        self.classes, self.imgs = self.__get_classes_and_images(img_dir)
        self.labels = torch.tensor(list(map(lambda x: self.classes.index(x[0]), self.imgs)), dtype=torch.int)
        before_crop = int(input_size * (8.0/7.0))
        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((before_crop, before_crop)),
                transforms.RandomCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __get_class_imgs(self, class_dir, class_id):
        imgs = []
        for entry in os.scandir(class_dir):
            if entry.is_dir():
                continue  
            img_path = entry.path
            imgs.append((class_id, img_path)) 
        return imgs

    def __get_classes_and_images(self, directory):
        classes = set()
        imgs = []
        for entry in os.scandir(directory):
            if not entry.is_dir():
                continue     
            class_dir = entry.path
            class_id = int(entry.name)
            classes.add(class_id)
            imgs = imgs + self.__get_class_imgs(class_dir, class_id)
        return list(classes), imgs

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, str(self.imgs[index][0]), self.imgs[index][1])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = self.imgs[index][0]
        return image, label
    
    def __len__(self):
        return len(self.imgs)


def train(opt: Namespace):
    model = get_model(opt.net)
    input_size = model.default_cfg["input_size"][1]

    train_dataset = TripletDataset(os.path.join(opt.data, "train"), input_size)
    val_dataset = TripletDataset(os.path.join(opt.data, "val"), input_size, is_train=False)

    train_batch_sampler = BalancedBatchSampler(train_dataset.labels, n_classes=opt.classes, n_samples=opt.samples)
    val_batch_sampler = BalancedBatchSampler(val_dataset.labels, n_classes=opt.classes, n_samples=opt.samples)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    online_train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=train_batch_sampler, **kwargs)
    online_val_loader = torch.utils.data.DataLoader(val_dataset, batch_sampler=val_batch_sampler, **kwargs)

    # Set up the network and training parameters
    margin = 1.
    loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = lr_scheduler.StepLR(optimizer, 8, gamma=0.1, last_epoch=-1)
    log_interval = 100

    fit(online_train_loader, online_val_loader, model, loss_fn, optimizer, scheduler, opt.epochs, cuda,
        log_interval, metrics=[AverageNonzeroTripletsMetric()])


def get_model(net: str):
    model = timm.create_model(net, pretrained=True, num_classes=0)
    if cuda:
        model.cuda()
    return model


if __name__ == '__main__':
    parser = ArgumentParser(description='Triplet loss training')
    parser.add_argument('--data', type=str, help='Path to dataset')
    parser.add_argument('--net', default='efficientnet_b1', type=str, help='Net to use')
    parser.add_argument('--classes', default=8, type=int, help='Classes to sample per BalancedBatchSampler batch')
    parser.add_argument('--samples', default=8, type=int, help='Images per class to sample per '
                                                               'BalancedBatchSampler batch')
    parser.add_argument('--lr', default=1e-4, type=int, help='Learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='Number of epochs')
    opt = parser.parse_args()
    train(opt)
