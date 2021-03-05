#export
from fastai.torch_basics import *
from fastai import vision
from torchvision.models.utils import load_state_dict_from_url
import torch.nn as nn 


class XResNet(nn.Sequential):
    @delegates(ResBlock)
    def __init__(self, block, expansion, layers, p=0.0, c_in=3, n_out=1000, stem_szs=(32,32,64),
                 widen=1.0, sa=False, act_cls=defaults.activation, ndim=2, ks=3, stride=2, **kwargs):
        store_attr('block,expansion,act_cls,ndim,ks')
        if ks % 2 == 0: raise Exception('kernel size has to be odd!')
        stem_szs = [c_in, *stem_szs]
        stem = [ConvLayer(stem_szs[i], stem_szs[i+1], ks=ks, stride=stride if i==0 else 1, 
                          act_cls=act_cls, ndim=ndim)
                for i in range(3)]

        block_szs = [int(o*widen) for o in [64,128,256,512] +[256]*(len(layers)-4)]
        block_szs = [64//expansion] + block_szs
        blocks    = self._make_blocks(layers, block_szs, sa, stride, **kwargs)

        super().__init__(
            *stem, MaxPool(ks=ks, stride=stride, padding=ks//2, ndim=ndim),
            *blocks,
            AdaptiveAvgPool(sz=1, ndim=ndim), Flatten(), nn.Dropout(p),
            nn.Linear(block_szs[-1]*expansion, n_out),
        )
        init_cnn(self)

    def _make_blocks(self, layers, block_szs, sa, stride, **kwargs):
        return [self._make_layer(ni=block_szs[i], nf=block_szs[i+1], blocks=l,
                                 stride=1 if i==0 else stride, sa=sa and i==len(layers)-4, **kwargs)
                for i,l in enumerate(layers)]

    def _make_layer(self, ni, nf, blocks, stride, sa, **kwargs):
        return nn.Sequential(
            *[self.block(self.expansion, ni if i==0 else nf, nf, stride=stride if i==0 else 1,
                      sa=sa and i==(blocks-1), act_cls=self.act_cls, ndim=self.ndim, ks=self.ks, **kwargs)
              for i in range(blocks)])

def init_cnn(m):
    if getattr(m, 'bias', None) is not None: nn.init.constant_(m.bias, 0)
    if isinstance(m, (nn.Conv1d,nn.Conv2d,nn.Conv3d,nn.Linear)): nn.init.kaiming_normal_(m.weight)
    for l in m.children(): init_cnn(l)

def _xresnet(pretrained, expansion, layers, **kwargs):
    # TODO pretrain all sizes. Currently will fail with non-xrn50
    url = 'https://s3.amazonaws.com/fast-ai-modelzoo/xrn50_940.pth'
    res = XResNet(ResBlock, expansion, layers, **kwargs)
    if pretrained: res.load_state_dict(load_state_dict_from_url(url, map_location='cpu')['model'], strict=False)
    return res

def xresnet34 (pretrained=False, **kwargs): return _xresnet(pretrained, 1, [3, 4,  6, 3], **kwargs)
def xresnet18 (pretrained=False, **kwargs): return _xresnet(pretrained, 1, [2, 2,  2, 2], **kwargs)
def xresnet50 (pretrained=False, **kwargs): return _xresnet(pretrained, 4, [3, 4,  6, 3], **kwargs)
