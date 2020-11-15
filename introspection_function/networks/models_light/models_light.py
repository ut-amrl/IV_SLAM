import torch
import torch.nn as nn
from typing import List

from . import  mobilenet
from torch.nn import BatchNorm2d


class IntrospectionModule(nn.Module):
    def __init__(self, net_enc, net_dec, enc_input_size=(512, 512),
                logistic_func=False):
        super(IntrospectionModule, self).__init__()
        self.encoder = net_enc
        self.decoder = net_dec
        self.enc_input_size = enc_input_size
        self.logistic_func = logistic_func

    def forward(self, input):
        input_r : torch.Tensor = nn.functional.interpolate(
                    input, size= self.enc_input_size,
                    mode='bilinear', align_corners=False)

        pred : torch.Tensor = self.decoder(self.encoder(input_r,      
                    return_feature_maps=True))
        if self.logistic_func:
            pred = torch.sigmoid(20 * (pred - 0.5))

        return pred



class ModelBuilder:
    # custom weights initialization
    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)
        #elif classname.find('Linear') != -1:
        #    m.weight.data.normal_(0.0, 0.0001)

    @staticmethod
    def build_encoder(arch='mobilenetv2dilated', fc_dim=512, weights=''):
        pretrained = True if len(weights) == 0 else False
        arch = arch.lower()
        if arch == 'mobilenetv2dilated':
            orig_mobilenet = mobilenet.__dict__['mobilenetv2'](pretrained=pretrained)
            net_encoder = MobileNetV2Dilated(orig_mobilenet, dilate_scale=8)
        else:
            raise Exception('Architecture undefined!')

        # encoders are usually pretrained
        # net_encoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_encoder')
            net_encoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_encoder

    @staticmethod
    def build_decoder(arch='c1_deepsup',
                      fc_dim=512, num_class=150,
                    weights='', regression_mode = False, 
                    inference_mode=False,
                    out_size=(512, 512)):
        arch = arch.lower()
        if arch == 'c1_deepsup':
            net_decoder = C1DeepSup(
                num_class=num_class,
                fc_dim=fc_dim,
                regression_mode = regression_mode,
                inference_mode = inference_mode,
                out_size = out_size)
        else:
            raise Exception('Architecture undefined!')

        net_decoder.apply(ModelBuilder.weights_init)
        if len(weights) > 0:
            print('Loading weights for net_decoder')
            net_decoder.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)
        return net_decoder


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    "3x3 convolution + BN + relu"
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            )



class MobileNetV2Dilated(nn.Module):
    def __init__(self, orig_net, dilate_scale=8):
        super(MobileNetV2Dilated, self).__init__()
        from functools import partial

        # take pretrained mobilenet features
        self.features = orig_net.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

        if dilate_scale == 8:
            # for i in range(self.down_idx[-2], self.down_idx[-1]):
            #     self.features[i].apply(
            #         partial(self._nostride_dilate, dilate=2)
            #     )
            # for i in range(self.down_idx[-1], self.total_idx):
            #     self.features[i].apply(
            #         partial(self._nostride_dilate, dilate=4)
            #     )

            i = 0
            for mod in self.features:
                if i >= self.down_idx[-2] and i < self.down_idx[-1]:
                    mod.apply(
                        partial(self._nostride_dilate, dilate=2))
                elif i >= self.down_idx[-1] and i < self.total_idx:
                    mod.apply(
                        partial(self._nostride_dilate, dilate=4))
                i+=1

        elif dilate_scale == 16:
            # for i in range(self.down_idx[-1], self.total_idx):
            #     self.features[i].apply(
            #         partial(self._nostride_dilate, dilate=2)
            #     )
            for mod in self.features:
                mod.apply(
                    partial(self._nostride_dilate, dilate=2))

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            # the convolution with stride
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            # other convoluions
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)

    def forward(self, x, 
            return_feature_maps:bool=True):
        if return_feature_maps:
            conv_out: List[torch.Tensor] = []
            # for i in range(self.total_idx):
            #     x = self.features[i](x)
            #     if i in self.down_idx:
            #         conv_out.append(x)
            i = 0
            for mod in self.features:
                x = mod(x)
                if i in self.down_idx:
                    conv_out.append(x)
                i += 1
            conv_out.append(x)
            return conv_out

        else:
            return [self.features(x)]


# last conv, deep supervision
class C1DeepSup(nn.Module):
    def __init__(self, num_class=150, fc_dim=2048,
                 regression_mode=True, inference_mode=True,
                 out_size=(512, 512)):
        super(C1DeepSup, self).__init__()
        self.regression_mode = regression_mode
        self.inference_mode = inference_mode
        self.out_size = out_size

        self.cbr = conv3x3_bn_relu(fc_dim, fc_dim // 4, 1)
        self.cbr_deepsup = conv3x3_bn_relu(fc_dim // 2, fc_dim // 4, 1)

        # last conv
        self.conv_last = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)
        self.conv_last_deepsup = nn.Conv2d(fc_dim // 4, num_class, 1, 1, 0)

    def forward(self, conv_out : List[torch.Tensor]):
        conv5 = conv_out[-1]

        x = self.cbr(conv5)
        x = self.conv_last(x)

        # Only supports inference mode
        x = nn.functional.interpolate(
            x, size= self.out_size, mode='bilinear', align_corners=False)

        if not self.regression_mode:
            x = nn.functional.softmax(x, dim=1)
        return x


