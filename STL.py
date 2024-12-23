import torch
import torch.nn as nn

class Dense2DSpatialTransformer(nn.Module):
    def __init__(self):
        super(Dense2DSpatialTransformer, self).__init__()

    def forward(self, input1, input2):
        return self._transform(input1, input2[:, 0], input2[:, 1])

    def _transform(self, input1, dHeight, dWidth):
        batchSize = dHeight.shape[0]
        hgt = dHeight.shape[1]
        wdt = dHeight.shape[2]

        H_mesh, W_mesh = self._meshgrid(hgt, wdt)
        H_mesh = H_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        W_mesh = W_mesh.unsqueeze_(0).expand(batchSize, hgt, wdt)
        H_upmesh = dHeight + H_mesh
        W_upmesh = dWidth + W_mesh

        return self._interpolate(input1, H_upmesh, W_upmesh)

    def _meshgrid(self, hgt, wdt):
        h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
        w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()
        return h_t, w_t

    def _interpolate(self, input, H_upmesh, W_upmesh):
        nbatch = input.shape[0]
        nch    = input.shape[1]
        height = input.shape[2]
        width  = input.shape[3]

        img = torch.zeros(nbatch, nch, height+2, width+2).cuda()
        img[:, :, 1:-1, 1:-1] = input
        img[:, :, 0, 1:-1] = input[:, :, 0, :]
        img[:, :, -1, 1:-1] = input[:, :, -1, :]
        img[:, :, 1:-1, 0] = input[:, :, :, 0]
        img[:, :, 1:-1, -1] = input[:, :, :, -1]
        img[:, :, 0, 0] = input[:, :, 0, 0]
        img[:, :, 0, -1] = input[:, :, 0, -1]
        img[:, :, -1, 0] = input[:, :, -1, 0]
        img[:, :, -1, -1] = input[:, :,-1, -1]

        imgHgt = img.shape[2]
        imgWdt = img.shape[3]

        # H_upmesh, W_upmesh = [H, W] -> [BHW,]
        H_upmesh = H_upmesh.view(-1).float()+1.0  # (BHW,)
        W_upmesh = W_upmesh.view(-1).float()+1.0  # (BHW,)

        # H_upmesh, W_upmesh -> Clamping
        hf = torch.floor(H_upmesh).int()
        hc = hf + 1
        wf = torch.floor(W_upmesh).int()
        wc = wf + 1

        hf = torch.clamp(hf, 0, imgHgt-1)  # (BHW,)
        hc = torch.clamp(hc, 0, imgHgt-1)  # (BHW,)
        wf = torch.clamp(wf, 0, imgWdt-1)  # (BHW,)
        wc = torch.clamp(wc, 0, imgWdt-1)  # (BHW,)

        # Find batch indexes
        rep = torch.ones([height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
        bHW = torch.matmul((torch.arange(0, nbatch).float()*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

        # Box updated indexes
        W = imgWdt
        # x: W, y: H, z: D
        idx_00 = bHW + hf*W + wf
        idx_10 = bHW + hf*W + wc
        idx_01 = bHW + hc*W + wf
        idx_11 = bHW + hc*W + wc

        # Box values
        img_flat = img.view(-1, nch).float()  # (BDHW,C) //// C=1

        val_00 = torch.index_select(img_flat, 0, idx_00.long())
        val_10 = torch.index_select(img_flat, 0, idx_10.long())
        val_01 = torch.index_select(img_flat, 0, idx_01.long())
        val_11 = torch.index_select(img_flat, 0, idx_11.long())

        dHeight = hc.float() - H_upmesh
        dWidth  = wc.float() - W_upmesh

        wgt_00 = (dHeight*dWidth).unsqueeze_(1)
        wgt_10 = (dHeight * (1-dWidth)).unsqueeze_(1)
        wgt_01 = ((1-dHeight) * dWidth).unsqueeze_(1)
        wgt_11 = ((1-dWidth) * (1-dHeight)).unsqueeze_(1)

        output = val_00*wgt_00 + val_10*wgt_10 + val_01*wgt_01 + val_11*wgt_11
        output = output.view(nbatch, height, width, nch).permute(0, 3, 1, 2)  #B, C, H, W
        return output

