import torch
import torch.nn as nn
import numpy as np

class ConstastLoss(nn.Module):
    def __init__(self, chunk_size, ip_weight):
        super().__init__()
        
        self.l1 = nn.L1Loss()
        self.chunk_size = chunk_size
        self.ip_weight = ip_weight
        self.weight = 1.0/8
        
    def forward(self, x):
        x_positive, x_negative = self._make_apn(x)
        loss = 0
        
        for i in range(x.size(0)):
            d_ap = self.l1(x[i], x_positive[i])
            d_an = self.l1(x[i], x_negative[i])
            
            contrastive = d_ap / (d_an + 1e-7)
            loss += self.weight * contrastive
            
        return loss
        
        
    def _make_apn(self, x):
        x_center = None
        x_positive = None
        x_negative = None
        
        for i in range(0, x.size(0), self.chunk_size):
            center = (torch.sum(x[i:i+self.chunk_size,:], dim=0) / self.chunk_size).unsqueeze(0)
            x_center = center if x_center is None else torch.cat((x_center, center), dim=0)
            for j in range(self.chunk_size):
                inter_pts = torch.lerp(x[i+j,:], center, self.ip_weight)
                x_positive = inter_pts if x_positive is None else torch.cat((x_positive, inter_pts), dim=0)
                
        distance = torch.matmul(x, torch.transpose(x, 0, 1)).cpu().detach().numpy()
        dist_indices = np.argsort(distance, axis=1)
        for i in range(dist_indices.shape[0]):
            keep = True
            start = i - int(i % 4)
            end = i + (3 - int(i % 4))
            pt = -1 
            while keep:
                index = dist_indices[i][pt]
                if index >= start and index <= end:
                    pt -= 1
                else:
                    neg = (x[index, :]).unsqueeze(0)
                    
                    x_negative = neg if x_negative is None else torch.cat((x_negative,neg),dim=0)
                    keep = False
                    
        return x_positive, x_negative
            
    
    