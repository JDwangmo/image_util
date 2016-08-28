#encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-07-11'
    Email:   '383287471@qq.com'
    Describe: 
"""
from __future__ import print_function

import logging

from common import CnnBaseClass

class ImageNet(CnnBaseClass):
    def __init__(self,
                 rand_seed=1337,
                 verbose=0,
                 feature_encoder=None,
                 optimizers='sgd',
                 input_shape=None,
                 num_labels=None,
                 nb_epoch=10,
                 earlyStoping_patience=50,
                 model_network_type='simple',
                 **kwargs
                 ):
        '''

        :param rand_seed:
        :param verbose:
        :param feature_encoder:
        :param optimizers:
        :param input_shape:
        :param num_labels:
        :param nb_epoch:
        :param earlyStoping_patience:
        :param model_network_type:
        :param kwargs: full_connected_layer_units, l1_conv_filter_type, l2_conv_filter_type
        '''

        self.model_network_type = model_network_type
        self.kwargs = kwargs


        CnnBaseClass.__init__(
            self,
            rand_seed=rand_seed,
            verbose=verbose,
            feature_encoder=feature_encoder,
            optimizers=optimizers,
            input_shape=input_shape,
            num_labels=num_labels,
            nb_epoch=nb_epoch,
            earlyStoping_patience=earlyStoping_patience,
            **kwargs
        )

        if feature_encoder is not None:
            self.input_shape = (1,feature_encoder.output_shape[0],feature_encoder.output_shape[1])

        # 构建模型
        self.build_model()

    def build_cnn_model_from_qiangjia(self):
        '''
            qiangjia版本的cnn网络

        :param layer1:
        :param hidden1:
        :param rows:
        :param cols:
        :param nkerns:
        :param nb_classes:
        :param lr:
        :param decay:
        :param momentum:
        :return:
        '''

        layer1 = self.kwargs.get('layer1',None)
        hidden1 = self.kwargs.get('hidden1',None)
        nkerns = self.kwargs.get('nkerns',None)

        assert layer1 is not None,'请设置 layer1!'
        assert hidden1 is not None,'请设置 hidden1!'
        assert nkerns is not None,'请设置 nkerns!'

        from keras.models import Sequential
        from keras.layers import Convolution2D,Activation,MaxPooling2D,Flatten,Merge,Dense,Dropout

        layer1_model1 = Sequential()
        layer1_model1.add(Convolution2D(layer1, nkerns[0], nkerns[0],
                                        border_mode='valid',
                                        input_shape=self.input_shape))

        layer1_model1.add(Activation('tanh'))
        layer1_model1.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model1.add(Flatten())  # 平铺

        layer1_model2 = Sequential()
        layer1_model2.add(Convolution2D(layer1, nkerns[1], nkerns[1],
                                        border_mode='valid',
                                        input_shape=self.input_shape))
        layer1_model2.add(Activation('tanh'))
        layer1_model2.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model2.add(Flatten())  # 平铺

        layer1_model3 = Sequential()
        layer1_model3.add(Convolution2D(layer1, nkerns[2], nkerns[2],
                                        border_mode='valid',
                                        input_shape=self.input_shape))
        layer1_model3.add(Activation('tanh'))
        layer1_model3.add(MaxPooling2D(pool_size=(2, 2)))
        layer1_model3.add(Flatten())  # 平铺

        model = Sequential()

        model.add(Merge([layer1_model2, layer1_model1, layer1_model3], mode='concat', concat_axis=1))  # merge

        model.add(Dense(hidden1))
        model.add(Activation('tanh'))
        model.add(Dropout(0.5))

        model.add(Dense(self.num_labels))
        model.add(Activation('softmax'))
        if self.verbose > 0:
            model.summary()
        return model

    def build_simple_cnn(self):
        '''
            创建简单的CNN模型，最多两层卷积层

            1. 输入层，2D，（n_batch,input_length）
            2. Embedding层,3D,（n_batch,input_length,embedding_dim）
            3. 输入dropout层，对Embedding层进行dropout.3D.
            4. Reshape层： 将embedding转换4-dim的shape，4D
            5. 第一层多size卷积层（含1-max pooling），使用三种size.
            6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
            7. output hidden层
            8. output Dropout层
            9. softmax 分类层
        :return:
        '''
        full_connected_layer_units = self.kwargs.get('full_connected_layer_units',None)
        l1_conv_filter_type = self.kwargs.get('l1_conv_filter_type',None)
        l2_conv_filter_type = self.kwargs.get('l2_conv_filter_type',None)

        assert full_connected_layer_units is not None,'请设置 full_connected_layer_units!'
        assert l1_conv_filter_type is not None,'请设置 l1_conv_filter_type!'
        assert l2_conv_filter_type is not None,'请设置 l2_conv_filter_type!'

        from keras.models import Model
        from keras.layers import ZeroPadding2D,Flatten,Activation,Input,BatchNormalization
        from keras import backend as K

        l1_input = Input(shape=self.input_shape)
        # l1_input = BatchNormalization()(l1_input)
        # l1_input_padding = ZeroPadding2D((2, 2))(l1_input)

        # 5. 第一层卷积层：多size卷积层（含1-max pooling），使用三种size.
        l2_conv = self.create_convolution_layer(
            input_layer=l1_input,
            convolution_filter_type=l1_conv_filter_type,
        )
        # print (self.embedding_layer_output.get_weights())
        # model = Model(input=l1_input, output=[l5_cnn])
        # model.summary()
        # quit()

        # 6. 第二层卷积层：单size卷积层 和 max pooling 层
        # l3_conv = ZeroPadding2D((2, 2))(l2_conv)
        l3_conv = self.create_convolution_layer(
            input_layer=l2_conv,
            convolution_filter_type=l2_conv_filter_type,
        )

        if self.kwargs.has_key('l3_conv_filter_type'):
            # l3_conv = ZeroPadding2D((2, 2))(l3_conv)

            l3_conv = self.create_convolution_layer(
                input_layer=l3_conv,
                convolution_filter_type=self.kwargs.get('l3_conv_filter_type',[]),
            )

        # 6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
        # l4_flatten = Flatten()(l3_conv)
        # l6_flatten= BatchNormalization(axis=1)(l6_flatten)
        # 7. 全连接层
        l5_full_connected_layer = self.create_full_connected_layer(
            input_layer=l3_conv,
            units=full_connected_layer_units,
        )

        l6_output = self.create_full_connected_layer(
            input_layer=l5_full_connected_layer,
            units=[[self.num_labels, 0., 'none', 'none']],
        )

        # 8. softmax 分类层
        l8_softmax_output = Activation("softmax")(l6_output)
        model = Model(input=[l1_input], output=[l8_softmax_output])

        self.conv1_feature_output = K.function([l1_input, K.learning_phase()], [l2_conv])

        if self.verbose > 0:
            model.summary()

        return model

    def build_simple_cnn_with_3conv(self):
        '''
            创建简单的CNN模型，三层卷积层

            1. 输入层，2D，（n_batch,input_length）
            2. Embedding层,3D,（n_batch,input_length,embedding_dim）
            3. 输入dropout层，对Embedding层进行dropout.3D.
            4. Reshape层： 将embedding转换4-dim的shape，4D
            5. 第一层多size卷积层（含1-max pooling），使用三种size.
            6. Flatten层： 卷积的结果进行拼接,变成一列隐含层
            7. output hidden层
            8. output Dropout层
            9. softmax 分类层
        :return:
        '''

        from keras.layers import Dense, Convolution2D, Activation, MaxPooling2D, Flatten, Dropout, ZeroPadding2D
        from keras.models import Sequential
        model = Sequential()
        win_shape = 3
        model.add(ZeroPadding2D((1, 1), input_shape=(1, 15, 15)))
        model.add(Convolution2D(32, win_shape, win_shape,
                                border_mode='valid',
                                ))
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(64, win_shape, win_shape,
                                border_mode='valid'
                                ))
        model.add(Activation('tanh'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(ZeroPadding2D((1, 1)))
        model.add(Convolution2D(128, win_shape, win_shape,
                                border_mode='valid'
                                ))
        model.add(Activation('tanh'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(output_dim=100, init="glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(p=0.5))
        model.add(Dense(output_dim=50, init="glorot_uniform"))
        model.add(Activation("relu"))
        model.add(Dropout(p=0.5))
        model.add(Dense(output_dim=self.num_labels, init="glorot_uniform"))
        model.add(Activation("softmax"))

        if self.verbose > 0:
            model.summary()

        return model

    def create_network(self):
        '''
            1. 创建 CNN 网络

        :return: cnn model network
        '''

        if self.model_network_type == 'simple':
            model = self.build_simple_cnn()
        elif self.model_network_type == 'simple_3_conv':
            model = self.build_simple_cnn_with_3conv()
        elif self.model_network_type == 'simple_from_qiangjia':
            model = self.build_cnn_model_from_qiangjia()
        else:
            raise NotImplementedError

        return model

    @staticmethod
    def get_feature_encoder(**kwargs):
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import StandardScaler
        class FeatureEncoder(object):
            def __init__(self):
                self.output_shape = kwargs.get('output_shape',(15,15))

            def fit(self,X=None):
                X = X.astype(dtype='float64')
                # self.normalizer = Normalizer(norm='l2')
                self.scaler = StandardScaler(with_mean=True, with_std=True)

                input_shape = X.shape
                X = X.reshape(input_shape[0],-1)

                # self.normalizer.fit(X)
                self.scaler.fit(X)

                return self

            def fit_transform(self, X=None):
                return self.fit(X).transform(X)

            def transform(self,
                          X,
                          ):
                X = X.astype(dtype='float64')
                input_shape = X.shape
                X = X.reshape(input_shape[0], -1)
                # X = self.normalizer.transform(X).reshape(input_shape)
                # 取左下角
                X = self.scaler.transform(X).reshape(input_shape)
                X = X[:,:,15-self.output_shape[0]:,:self.output_shape[1]]
                return X

        feature_encoder = FeatureEncoder()

        return feature_encoder

    def print_model_descibe(self):
        import pprint
        detail = {'rand_seed': self.rand_seed,
                  'verbose': self.verbose,
                  'model_network_type':self.model_network_type,
                  'optimizers': self.optimizers,
                  'input_shape': self.input_shape,
                  'num_labels': self.num_labels,
                  'l1_conv_filter_type': self.kwargs.get('l1_conv_filter_type',None),
                  'l2_conv_filter_type': self.kwargs.get('l2_conv_filter_type',None),
                  'full_connected_layer_units': self.kwargs.get('full_connected_layer_units',None),
                  'nb_epoch': self.nb_epoch,
                  'earlyStoping_patience': self.earlyStoping_patience,
                  'lr': self.lr,
                  'batch_size': self.batch_size,
                  }
        pprint.pprint(detail)
        logging.debug(detail)
        return detail


if __name__ == '__main__':
    image_net = ImageNet(
        rand_seed=1337,
        verbose=1,
        feature_encoder=None,
        optimizers='sgd',
        input_shape=(1,15,15),
        num_labels=2,
        nb_epoch=10,
        earlyStoping_patience=50,
        model_network_type='simple',
        l1_conv_filter_type=[
            # [32, 2, 2, 'valid', (2, 2), 0.5, 'none', 'none','flatten'],
            [32, 4, 4, 'valid', (2, 2), 0., 'none', 'none'],
            # [4, 5, -1, 'valid', (2, 1), 0., 'none', 'none'],
        ],
        l2_conv_filter_type=[
            [32, 2, 2, 'valid', (2, 2), 0.5, 'none', 'none'],
        ],
        full_connected_layer_units = [
            (50,0.5,'relu','none'),
            # (100,0.25,'relu','none')
        ]
    )
    image_net.batch_predict()
    image_net.print_model_descibe()