"""
Created on 14:29 at 13/01/2022
@author: bo
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F


class XceptionCls(nn.Module):
    def __init__(self, wavenumber, num_class=2, stem_kernel=21, depth=128, stem_max_dim=64,
                 within_dropout=True, quantification=False, detection=True, reduce_channel_first=False, 
                 data_input_channel=1, cast_quantification_to_classification=False):
        super(XceptionCls, self).__init__()
        self.wavenumber = wavenumber
        self.depth = depth
        self.stem_kernel = stem_kernel
        self.num_class = num_class
        self.quantification = quantification
        self.reduce_channel_first = reduce_channel_first
        self.detection = detection
        self.cast_quantification_to_classification = cast_quantification_to_classification
        self.extractor = Xception(self.wavenumber, stem_kernel, 2, act="leakyrelu",
                                  depth=depth, stem_max_dim=stem_max_dim,
                                  within_dropout=within_dropout, data_input_channel=data_input_channel)

        self.feature_dim = wavenumber // 2 * self.depth

        if self.reduce_channel_first:
            self.reduce_channel_block = nn.Conv1d(self.depth, 1, 1)
            self.feature_dim = wavenumber // 2
        
        if self.detection:
            self.cls_block = nn.Linear(self.feature_dim, num_class)
        if self.quantification:
            if cast_quantification_to_classification:
                self.quan_block = nn.Linear(self.feature_dim, num_class)
            else:
                self.quan_block = nn.Linear(self.feature_dim, 1)

    def forward(self, x):
        feat = self.extractor(x)
        if self.reduce_channel_first:
            feat = self.reduce_channel_block(feat)
            feat = feat.squeeze(1)
        feat = feat.view(len(x), self.feature_dim)
        if self.detection:
            y_pred = self.cls_block(feat)
        else:
            y_pred = []
        if self.quantification:
            if self.cast_quantification_to_classification:
                y_quan_pred = self.quan_block(feat)
            else:
                y_quan_pred = self.quan_block(feat)[:, 0]
        else:
            y_quan_pred = []
        return feat, y_pred, y_quan_pred
    
    def forward_test(self, x):
        feat = self.extractor.forward_test(x)
        if self.reduce_channel_first:
            feat = self.reduce_channel_block(feat)
            feat = feat.squeeze(1)
        feat = feat.view(len(x), self.feature_dim)
        if self.detection:
            y_pred = self.cls_block(feat)
        else:
            y_pred = []
        if self.quantification:
            if self.cast_quantification_to_classification:
                y_quan_pred = self.quan_block(feat)
            else:
                y_quan_pred = self.quan_block(feat)[:, 0]
        else:
            y_quan_pred = []
        return feat, y_pred, y_quan_pred
        


class UnifiedCNN(nn.Module):
    def __init__(self, input_shape, num_classes, block_type, quantification=False, detection=True,
                 cast_quantification_to_classification=False):
        super(UnifiedCNN, self).__init__()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.quantification = quantification
        self.detection = detection 
        self.cast_quantification_to_classification = cast_quantification_to_classification
        self.feature_dimension = input_shape[1] // 2 ** 3 * 64
        print("----------------model information-------------------")
        print("The input shape:", self.input_shape)
        print("The number of classes:", self.num_classes)
        print("The block type:", block_type)
        print("The feature dimension", self.feature_dimension)
        print("----------------------------------------------------")
        stride = 1

        self.encblock0 = CNNBlock(block_type, self.input_shape[0], 32, 21, stride, "enc_block0")
        self.encblock1 = CNNBlock(block_type, 32, 64, 11, stride, "enc_block1")
        self.encblock2 = CNNBlock(block_type, 64, 64, 5, stride, "enc_block2")

        if self.detection:
            self.fc_layer = nn.Sequential()
            self.fc_layer.add_module("cls_fc0", nn.Linear(self.feature_dimension, 2048))
            self.fc_layer.add_module("cls_bn0", nn.BatchNorm1d(2048))
            self.fc_layer.add_module("cls_fc1", nn.Linear(2048, 1024))
            self.fc_layer.add_module("cls_drop", nn.Dropout())
            self.fc_layer.add_module("cls_fc2", nn.Linear(1024, num_classes))
            self.fc_layer.apply(kaiming_init)

        if self.quantification:
            self.quan_layer = nn.Sequential()
            self.quan_layer.add_module("quan_fc0", nn.Linear(self.feature_dimension, 2048))
            self.quan_layer.add_module("quan_bn0", nn.BatchNorm1d(2048))
            self.quan_layer.add_module("quan_fc1", nn.Linear(2048, 1024))
            self.quan_layer.add_module("quan_drop", nn.Dropout())
            if self.cast_quantification_to_classification:
                self.quan_layer.add_module("quan_fc2", nn.Linear(1024, num_classes))
            else:
                self.quan_layer.add_module("quan_fc2", nn.Linear(1024, 1))
            # self.quan_layer.apply(kaiming_init)
        
    def forward(self, spectrum):
        x = self.encblock0(spectrum)
        x = self.encblock1(x)
        x = self.encblock2(x)
        feat = torch.reshape(x, [-1, self.feature_dimension])
        if self.quantification:
            if not self.cast_quantification_to_classification:
                quan_pred = self.quan_layer(feat)[:, 0]
            else:
                quan_pred = self.quan_layer(feat)
        else:
            quan_pred = []
        if self.detection:
            pred = self.fc_layer(feat)
        else:
            pred = []
        return feat, pred, quan_pred
    
    def forward_test(self, spectrum):
        x = self.encblock0(spectrum)
        x = self.encblock1(x)
        x = self.encblock2(x)
        feat = torch.reshape(x, [-1, self.feature_dimension])
        if self.quantification:
            if not self.cast_quantification_to_classification:
                quan_pred = self.quan_layer(feat)[:, 0]
            else:
                quan_pred = self.quan_layer(feat)
        else:
            quan_pred = []
        if self.detection:
            pred = self.fc_layer(feat)
        else:
            pred = []
        return feat, pred, quan_pred
        

    def forward_test_batch_on_fc(self, spectrum):
        x = self.encblock0(spectrum)
        x = self.encblock1(x)
        x = self.encblock2(x)
        x = torch.reshape(x, [-1, self.feature_dimension])
        x = self.fc_layer(x)
        return nn.Softmax(dim=-1)(x)


def calc_num_param(model):
    num_param = 0.0
    for name, p in model.named_parameters():
        _num = np.prod(p.shape)
        num_param += _num
    print("There are %.2f million parameters" % (num_param / 1e+6))


class SeparableConv1dACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=False):
        """Separable Convolutional Network
        If the input shape is : [2, 32, 128], and we want to get output size of [2, 64, 128] with kernel 3.
        In the normal convolutional operation, the number of parameters is:
            32 * 64 * 3
        In the separable convolution, the number of parameter is:
            1 * 1 * 3 * 32 + 1 * 32 * 64 = 3 * 32 * (1 + 64/3) round to 3 * 32 * 21, which has 3 times less number of
            parameters compared to the original operation
        """
        super(SeparableConv1dACT, self).__init__()
        padding = int((kernel_size - 1) // 2)
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=bias, groups=in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1,
                                   stride=1, padding=0, bias=bias)
        self.conv1.apply(normal_init)
        self.pointwise.apply(normal_init)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class XceptionStemBlock(nn.Module):
    def __init__(self, kernel, depth=1, max_dim=64, data_input_channel=1):
        super(XceptionStemBlock, self).__init__()
        if max_dim == 64 or max_dim == 128:
            input_dim = [data_input_channel, 32]
            output_dim = [32, 64]
        elif max_dim == 32:
            input_dim = [data_input_channel, 16]
            output_dim = [16, 32]

        act = nn.LeakyReLU(0.3)
        self.depth = depth
        self.stem_1 = nn.Sequential()
        input_channel = input_dim[0]
        output_channel = output_dim[0]
        pad = int((kernel - 1) // 2)
        for i in range(2):
            self.stem_1.add_module("stem1_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                        output_channel,
                                                                        kernel_size=kernel,
                                                                        stride=1,
                                                                        padding=pad))
            self.stem_1.add_module("stem1_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
            self.stem_1.add_module("stem1_act_%d" % (i + 1), act)
            input_channel = output_channel

        output_channel = output_dim[1]
        if depth == 2:
            self.stem_2 = nn.Sequential()
            for i in range(2):
                self.stem_2.add_module("stem2_conv_%d" % (i + 1), nn.Conv1d(input_channel,
                                                                            output_channel,
                                                                            kernel_size=kernel,
                                                                            stride=1,
                                                                            padding=pad))
                self.stem_2.add_module("stem2_bn_%d" % (i + 1), nn.BatchNorm1d(output_channel))
                self.stem_2.add_module("stem2_act_%d" % (i + 1), act)
                input_channel = output_channel
            self.stem_2.apply(normal_init)

        self.stem_1.apply(normal_init)

    def forward(self, x):
        x = self.stem_1(x)
        x = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x = self.stem_2(x)
        return x

    def forward_test(self, x):
        x = self.stem_1(x)
        x_pool = nn.MaxPool1d(2)(x)
        if self.depth == 2:
            x_further = self.stem_2(x_pool)
        else:
            x_further = x_pool
        return x, x_pool, x_further


class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, repeats, kernel_size,
                 stride=1, act="relu", start_with_act=True, grow_first=True):
        super(XceptionBlock, self).__init__()
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
            self.skipbn = nn.BatchNorm1d(out_channels)
        else:
            self.skip = None

        if act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "leakyrelu":
            self.act = nn.LeakyReLU(0.3, inplace=True)
        else:
            print("------The required activation function doesn't exist--------")
        rep = []
        filters = in_channels
        if grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(in_channels, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        for i in range(repeats)[1:]:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))
            filters = out_channels

        if not grow_first:
            rep.append(self.act)
            rep.append(SeparableConv1dACT(filters, out_channels, kernel_size, bias=False))
            rep.append(nn.BatchNorm1d(out_channels))

        if not start_with_act:
            rep = rep[1:]

        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        return x + skip


class Xception(nn.Module):
    def __init__(self, wavenumber, stem_kernel, num_xception_block=2, act="leakyrelu", depth=128,
                 stem_max_dim=64, within_dropout=False, data_input_channel=1):
        super(Xception, self).__init__()
        self.depth = depth
        self.num_xception_block = num_xception_block        
        self.stem = XceptionStemBlock(stem_kernel, 2, stem_max_dim, data_input_channel=data_input_channel)
        self.block1 = XceptionBlock(stem_max_dim, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=False, grow_first=True)
        self.block2 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                    stride=1, act=act, start_with_act=True, grow_first=True)
        if num_xception_block == 3:
            self.block3 = XceptionBlock(depth, depth, repeats=2, kernel_size=stem_kernel,
                                        stride=1, act=act, start_with_act=True, grow_first=True)
        if num_xception_block == 2:
            self.feature_dimension = wavenumber // 2
        elif num_xception_block == 3:
            self.feature_dimension = wavenumber // 4
        self.within_dropout = within_dropout

    def forward(self, x):
        x = self.stem(x)
        if self.within_dropout is True:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block1(x)
        if self.num_xception_block == 3:
            x = nn.MaxPool1d(2)(x)
        if self.within_dropout is True:
            x = nn.Dropout(p=0.5, inplace=True)(x)
        x = self.block2(x)
        if self.num_xception_block == 3:
            x = self.block3(x)
        return x
    
    def forward_test(self, x):
        x = self.stem(x)
        x = self.block1(x)
        if self.num_xception_block == 3:
            x = nn.MaxPool1d(2)(x)
        x = self.block2(x)
        if self.num_xception_block == 3:
            x = self.block3(x)
        return x
        

    def test_dropout(self, x, num_sample):
        """Get the MC Dropout probability of the prediction at the test time
        Args:
            x: feature maps from the last block in the inception network
            num_sample: the generated number of samples
        """
        x_g = []
        for i in range(num_sample):
            _x = nn.Dropout(p=0.5)(x)
            x_g.append(_x)
        x_g = torch.cat(x_g, dim=0).reshape([num_sample, self.depth, self.feature_dimension])
        return x_g

    def forward_multilevel_feature(self, x):
        x_init, x_pool, x = self.stem.forward_test(x)
        x_block1 = self.block1(x)
        x_block2 = self.block2(x_block1)
        return x_init, x_pool, x, x_block1, x_block2

    def forward_similarity_on_multilevel_features(self, test_data, reference_features):
        """Calculate the simialrity on multiple levels of features
        Maybe I can add auxiliary tasks along the network, because this is similar to the auxiliary task as well?
        Args:
            test_data: [num_test_data, 1, wave_number], tensor
            reference_features: dictionary, where the keys are "level-%d" % i for i in [1, 2, 3, 4]
        """
        _, x_pool, x, x_block1, x_block2 = self.forward_multilevel_feature(test_data)  # [100, channel, wavenumber]
        test_features = [x_pool, x, x_block1, x_block2]
        dot_similarity = {}
        l1norm_similarity = {}
        for key in reference_features.keys():
            dot_similarity[key] = []
            l1norm_similarity[key] = []
        for i, single_key in enumerate(reference_features.keys()):
            _reference_feature = reference_features[single_key]
            _test_feature = test_features[i]
            for j, _s_t_feat in enumerate(_test_feature):
                _dot_product = (_s_t_feat * _reference_feature)  # [num_reference_data, channel, wavenumber]
                _l1norm_value = (_s_t_feat - _reference_feature).abs()  # [num_reference_data, channel, wavenumber]
                dot_similarity[single_key].append(_dot_product.sum(dim=(-1, -2)))
                l1norm_similarity[single_key].append(_l1norm_value.sum(dim=(-1, -2)))
        return dot_similarity, l1norm_similarity


class CNNBlock(nn.Module):
    def __init__(self, blocktype, input_channel, output_channel, kernel, stride, name):
        super(CNNBlock, self).__init__()
        activation_layer = nn.LeakyReLU(0.3)
        padding = int((kernel - 1) // 2)
        encblock = nn.Sequential()
        encblock.add_module("%s_conv0" % name, nn.Conv1d(input_channel, output_channel, kernel, stride=stride,
                                                         padding=padding))
        encblock.add_module("%s_bn0" % name, nn.BatchNorm1d(output_channel))
        encblock.add_module("%s_act0" % name, activation_layer)
        if blocktype == "lenet":
            encblock.add_module("%s_conv1" % name, nn.Conv1d(output_channel, output_channel, kernel, stride=stride,
                                                             padding=padding))
            encblock.add_module("%s_bn1" % name, nn.BatchNorm1d(output_channel))
            encblock.add_module("%s_act1" % name, activation_layer)
        self.encblock = encblock
        self.encblock.apply(normal_init)

    def forward(self, x):
        x = self.encblock(x)
        x = nn.MaxPool1d(2)(x)
        return x


def normal_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d, nn.Conv1d)):
        init.normal_(m.weight, 0, 0.05)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def kaiming_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)
