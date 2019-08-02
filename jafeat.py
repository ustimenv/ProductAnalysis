from gluoncv.model_zoo import *
from gluoncv.model_zoo.ssd.vgg_atrous import VGGAtrousBase
from mxnet.gluon import nn


# Resnet multilayer feature extractor, does not really work
class JafeatRes(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(JafeatRes, self).__init__(**kwargs)
        self.block = BottleneckV2
        self.layers = [3, 4, 6, 3]
        self.channels = [64, 256, 512, 1024, 2048]
        last_gamma = False
        use_se = False
        norm_kwargs = None

        assert len(self.layers) == len(self.channels) - 1

        with self.name_scope():
            self.features = nn.HybridSequential(prefix='')

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Conv2D(self.channels[0], 7, 2, 3, use_bias=False))

            self.features.add(nn.BatchNorm())
            self.features.add(nn.Activation('relu'))
            self.features.add(nn.MaxPool2D(3, 2, 1))

            in_channels = self.channels[0]
            for i, num_layer in enumerate(self.layers):
                stride = 1 if i == 0 else 2
                self.features.add(self._make_layer(num_layer, self.channels[i + 1],
                                                   stride, i + 1, in_channels=in_channels,
                                                   last_gamma=last_gamma, use_se=use_se,
                                                   norm_layer=nn.BatchNorm, norm_kwargs=norm_kwargs))
                in_channels = self.channels[i + 1]

    def _make_layer(self, layers, channels, stride, stage_index, in_channels=0, last_gamma=False,
                    use_se=False, norm_layer=nn.BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential(prefix='stage%d_' % stage_index)

        with layer.name_scope():
            layer.add(self.block(channels, stride, channels != in_channels, in_channels=in_channels,
                                 last_gamma=last_gamma, use_se=use_se, prefix='',
                                 norm_layer=norm_layer, norm_kwargs=norm_kwargs))
            for _ in range(layers - 1):
                layer.add(self.block(channels, 1, False, in_channels=channels,
                          last_gamma=last_gamma, use_se=use_se, prefix='',
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs))

        return layer

    def hybrid_forward(self, F, x, **kwargs):
        output = []
        # apply some basic transforms indiscriminantly
        for i in range(0, 5):
            x = self.features[i](x)
        # for every layer (4 in total), return the respective feature map
        for i in range(5, 9):
            x = self.features[i](x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2), pooling_convention='full')
            output.append(x)
        return output


class JafeatVgg(VGGAtrousBase):
    layers, channels = ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512])

    def __init__(self, batch_norm=True, **kwargs):
        super(JafeatVgg, self).__init__(self.layers, self.channels, batch_norm, **kwargs)

        extras = [((256, 1, 1, 0), (512, 3, 2, 1)),
                  ((128, 1, 1, 0), (256, 3, 2, 1)),
                  ((128, 1, 1, 0), (256, 3, 1, 0)),
                  ((128, 1, 1, 0), (256, 3, 1, 0))]

        with self.name_scope():
            self.extras = nn.HybridSequential()
            for i, config in enumerate(extras):
                extra = nn.HybridSequential(prefix='extra%d_' % (i))
                with extra.name_scope():
                    for f, k, s, p in config:
                        extra.add(nn.Conv2D(f, k, s, p, **self.init))
                        if batch_norm:
                            extra.add(nn.BatchNorm())
                        extra.add(nn.Activation('relu'))
                self.extras.add(extra)

    def hybrid_forward(self, F, x, init_scale):
        x = F.broadcast_mul(x, init_scale)
        assert len(self.stages) == 6
        outputs = []
        for stage in self.stages[:3]:
            x = stage(x)
            x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                          pooling_convention='full')
        x = self.stages[3](x)
        norm = self.norm4(x)
        outputs.append(norm)
        x = F.Pooling(x, pool_type='max', kernel=(2, 2), stride=(2, 2),
                      pooling_convention='full')
        x = self.stages[4](x)
        x = F.Pooling(x, pool_type='max', kernel=(3, 3), stride=(1, 1), pad=(1, 1),
                      pooling_convention='full')
        x = self.stages[5](x)
        outputs.append(x)
        for extra in self.extras:
            x = extra(x)
            outputs.append(x)
        return outputs



