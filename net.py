import torch
import timm
import torch.nn as nn
import pytorch_lightning as pl
from pytorchcv.model_provider import get_model
import torchmetrics.functional as metrics

class Model(pl.LightningModule):
      def __init__(self, architecture='inception_v3', loss_func=nn.L1Loss()):
        super().__init__()

        self.model = timm.create_model(architecture, pretrained=True, num_classes=1)
	      self.loss_func = loss_func
        
        self.model.norm = nn.Identity()
  	    self.model.pre_legits = nn.Identity()
        self.model.head = nn.Sequential(nn.Linear(192,128),nn.Linear(128,1))


        self.lr = 1e-2
        self.lr_patience = 5
        self.lr_min = 1e-7

        self.labels_p = []
        self.labels_gt = []

    	     
    def rand_bbox(self,size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

    def cutmix_data(self,x, y, alpha=1.0):
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1               
        batch_size = x.size()[0]
        index = torch.randperm(batch_size)
      
        y_a = y
        y_b = y[index]
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(x.size(), lam)
        x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))

        return x, y_a, y_b, lam
    
    
    def mixed_loss(self,loss_func, pred, y_a, y_b, lam):
        return lam * loss_func(pred, y_a) + (1 - lam) * loss_func(pred, y_b)def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        image , targets_a, targets_b, lam = self.cutmix_data(images, labels)
        output = self.forward(image)
        loss = self.mixed_loss(self.loss_func, output, targets_a, targets_b, lam)
        labels = lam * targets_a + (1 - lam) * targets_b
        mae = metrics.mean_absolute_error(output, labels)
        return {'loss': loss, 'mae': mae}

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)
        return {'loss': loss, 'mae': mae}

    def test_step(self, batch, batch_idx):
        images, labels = batch
        labels = labels.reshape(1, -1).t()
        output = self.forward(images)
        loss = self.loss_func(output, labels)
        mae = metrics.mean_absolute_error(output, labels)

        self.labels_p = self.labels_p + output.squeeze().tolist()
        self.labels_gt = self.labels_gt + labels.squeeze().tolist()
        return {"loss": loss, "mae": mae}


    def training_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        self.log('Loss/Train', loss)
        self.log('MAE/Train', mae)

    def validation_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        self.log('Loss/Val', loss, prog_bar=True)
        self.log('MAE/Val', mae, prog_bar=True)

    def test_epoch_end(self, outs):
        loss = torch.stack([x['loss'] for x in outs]).mean()
        mae = torch.stack([x['mae'] for x in outs]).mean()
        self.log('Loss/Test', loss, prog_bar=True)
        self.log('MAE/Test', mae, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.lr_patience, min_lr=self.lr_min)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": 'Loss/Val', "interval": 'epoch'}
