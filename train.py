import time
import torch
import torch.optim
import torch.utils.data
from model import SSD300, MultiBoxLoss
from datasets import PascalVOCDataset
from PIL import Image, ImageDraw, ImageFont
from utils import *
import pdb
import numpy as np
from numpy import rad2deg
import cv2
from vis import *
from detect import detection

data_folder = '/home/mstveras/ssd-360'
n_classes = len(label_map)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 20
workers = 4
lr = 1e-5
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
train_dataset = PascalVOCDataset(split ='train', keep_difficult=False, max_images=1000, new_w = 300, new_h = 300)
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
        #format [-pi,pi], [-pi/2,pi/2], [0,1], [0,1]
        boxes = [b.to(device) for b in boxes]
        labels = [l.to(device) for l in labels]
        optimizer.zero_grad()
        
        # Forward prop.
        image = images[0]
        box2 = boxes[0]
        target_height, target_width = 300, 300
        color = (0,255,0)
        gt_image = plot_image(image, box2, target_height, target_width, color)
        cv2.imwrite('gt.png', gt_image)

        predicted_locs, predicted_scores = model(images)

        #image = images[0]
        #box2 = predicted_locs[0].cpu().detach()[0:5]
        #target_height, target_width = 300, 300
        #plot_image(image, box2, target_height, target_width)

        loss = criterion(predicted_locs, predicted_scores, boxes, labels)

        #pdb.set_trace()
        losses.update(loss.item(), images.size(0))

        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - start)
        start = time.time()

        img_path = '/home/mstveras/torch-ssd/img2.jpg'
        original_image = Image.open(img_path, mode='r')
        original_image = original_image.convert('RGB')

        det_boxes = detection(model, original_image, min_score=0.6, max_overlap=0.3, top_k=5)
        h, w = 300,300
        image=cv2.imread(img_path)

        for i in range(len(det_boxes)):
            #k = random.randint(0, 8500)
            box = det_boxes[i]
            u00, v00 = ((rad2deg(box[0])/360)+0.5)*300, ((rad2deg(box[1])/180)+0.5)*300
            a_lat, a_long = box[2], box[3]
            #color = color_map.get(classes[i], (255, 255, 255))
            color = (0,255,0)
            image = plot_bfov(image, v00, u00, a_long, a_lat, color, h, w)

        cv2.imwrite('final_image.png', image)


        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  data_time=data_time, loss=losses))
    save_checkpoint(epoch, model, optimizer)

#model_file = "best.pth"
#torch.save(model.state_dict(), model_file)
#print(f"Model saved to {model_file} after Epoch {epoch + 1}")
