import time
import torch
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from utils import *
import pdb
import numpy as np
import cv2
from vis import plot_bfov

# Parâmetros de dados, modelo e treinamento (mantidos do código original)
data_folder = '/home/mstveras/ssd-360'
keep_difficult = True
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
workers = 4
lr = 1e-4
momentum = 0.9
weight_decay = 5e-4
print_freq = 200  # Frequência de impressão

# Carregar ou inicializar modelo e otimizador
checkpoint = None
if checkpoint is None:
    model = SSD300(n_classes=n_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
else:
    checkpoint = torch.load(checkpoint)
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

model = model.to(device)
criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

# DataLoader
train_dataset = PascalVOCDataset(split ='train', keep_difficult=False, max_images=10, new_w = 300, new_h = 300)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           collate_fn=train_dataset.collate_fn, num_workers=workers,
                                           pin_memory=True)

# Funções auxiliares para monitoramento
class AverageMeter(object):
    """Computa e armazena a média e a atualização corrente"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def plot_image(image, box2, target_height, target_width):
    # Convert image from PyTorch tensor to numpy array and resize
    img = image.cpu().detach().numpy().transpose(1, 2, 0)
    img = (img * 255).astype(np.uint8)
    img = cv2.resize(img, (target_width, target_height))

    # Process each box to plot
    for box in box2.cpu():
        u00, v00 = box[0] * target_width, box[1] * target_height
        a_lat, a_long = np.radians(box[2]*45), np.radians(box[3]*45)
        color = (0, 255, 0)  # Example color, adjust as needed
        img = plot_bfov(img, v00, u00, a_long, a_lat, color, target_height, target_width)
    cv2.imwrite('final_image.png', img)



# Configuração das épocas
num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    start = time.time()

    for i, (images, boxes, labels, _) in enumerate(train_loader):
        data_time.update(time.time() - start)

        images = images.to(device)
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        optimizer.zero_grad()
        # Forward prop.
        predicted_locs, predicted_scores = model(images)

        image = images[0]
        box = boxes[0]
        h,w  = 300,300
        plot_image(image, box, h, w)

        # Cálculo da perda
        loss = criterion(predicted_locs, predicted_scores, boxes, labels)
        losses.update(loss.item(), images.size(0))

        # Backward prop. e otimização
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    save_checkpoint(epoch, model, optimizer)

# Salvar o modelo
model_file = "best.pth"
torch.save(model.state_dict(), model_file)
print(f"Model saved to {model_file} after Epoch {epoch + 1}")
