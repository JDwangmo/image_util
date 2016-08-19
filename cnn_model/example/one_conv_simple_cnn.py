# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-08-15'
    Email:   '383287471@qq.com'
    Describe:  单层卷积层的 CNN 模型
"""



from cnn_model.image_net_model import ImageNet
import numpy as np
import pickle
class ImageCNNWithOneConv(object):

    @staticmethod
    def get_model(
            feature_encoder,
            num_filter,
            num_labels,
            hidden1,
            filter1,
            filter2,
            filter3,
            **kwargs
    ):

        image_net = ImageNet(
            rand_seed=1377,
            verbose=kwargs.get('verbose',0),
            feature_encoder=feature_encoder,
            # optimizers='adadelta',
            optimizers='sgd',
            input_shape=(1, 15, 15),
            # model_network_type='simple_3_conv',
            model_network_type='simple',
            l1_conv_filter_type=[
                # [num_filter, 1, 1, 'valid', (2, 2), 0., 'relu', 'none','flatten'],
                # [num_filter, 2, 2, 'valid', (2, 2), 0., 'relu', 'none','flatten'],
                # [num_filter, filter1, 3, 'valid', (2, 2), 0., 'relu', 'none','flatten'],
                [num_filter, filter2, 5, 'valid', (2, 2), 0., 'none', 'none'],
                # [num_filter, filter3, 7, 'valid', (2, 2), 0., 'none', 'none'],
                # [num_filter, 12, 3, 'valid', (2, 2), 0., 'none', 'none'],
            ],
            l2_conv_filter_type=[
                # [128, 3, 3, 'valid', (2, 2), 0., 'relu', 'none'],

            ],
            full_connected_layer_units=[
                (hidden1, 0.8, 'relu', 'none'),
                # (50, 0.5, 'relu', 'none'),
            ],
            num_labels=num_labels,
            nb_epoch=30,
            nb_batch=32,
            earlyStoping_patience=30,
            lr=1e-2,
            show_validate_accuracy=False if kwargs.get('verbose',0)==0 else True,
        )

        # image_net = ImageNet(
        #     rand_seed=1377,
        #     verbose=kwargs.get('verbose', 0),
        #     feature_encoder=feature_encoder,
        #     # optimizers='adadelta',
        #     optimizers='sgd',
        #     input_shape=(1, 15, 15),
        #     # model_network_type='simple_3_conv',
        #     model_network_type='simple_from_qiangjia',
        #     layer1=num_filter,
        #     hidden1=hidden1,
        #     nkerns =[3,5,7],
        #     num_labels=num_labels,
        #     nb_epoch=30,
        #     nb_batch=32,
        #     earlyStoping_patience=30,
        #     lr=1e-2,
        # )
        if kwargs.get('verbose',0)>0:
            image_net.print_model_descibe()
        # quit()
        return image_net

    @staticmethod
    def cross_validation(
            train_data=None,
            test_data=None,
            cv_data=None,
            output_shape=None,
            num_filter_list=None,
            hidden1_list=None,
            filter1_list=None,
            filter2_list=None,
            filter3_list=None,
            num_labels = 34,
            verbose = 0,
            output_badcase = False,
            **kwargs
           ):
        print('='*80)
        from data_processing_util.cross_validation_util import transform_cv_data, get_k_fold_data, get_val_score
        # 1. 获取交叉验证的数据
        if cv_data is None:
            assert train_data is not None, 'cv_data和train_data必须至少提供一个！'
            cv_data = get_k_fold_data(
                rand_seed = 0,
                k=10,
                train_data=train_data,
                test_data=test_data,
                include_train_data=True,
            )

        # 2. 将数据进行特征编码转换
        feature_encoder = ImageNet.get_feature_encoder(output_shape=output_shape)

        cv_data = transform_cv_data(feature_encoder, cv_data,verbose=verbose)
        # 交叉验证
        for num_filter in num_filter_list:
            for hidden1 in hidden1_list:
                for filter1 in filter1_list:
                    for filter2 in filter2_list:
                        for filter3 in filter3_list:
                            print('=' * 40)
                            print('num_filter, hidden1,filter1,filter2,filter3 is %d,%d,%d,%d,%d.'%(num_filter,
                                                                                           hidden1,
                                                                                           filter1,
                                                                                           filter2,
                                                                                           filter3,
                                                                                           ))
                            _,_,test_predict_result = get_val_score(
                                ImageCNNWithOneConv,
                                cv_data=cv_data,
                                verbose=verbose,
                                get_predict_result=False,
                                num_filter=num_filter,
                                hidden1 = hidden1,
                                filter1=filter1,
                                filter2=filter2,
                                filter3=filter3,
                                num_labels=num_labels,
                            )

                            if output_badcase:
                                # 输出badcase
                                test_X,test_y = test_data
                                is_badcase = test_y!=test_predict_result[0]
                                print(is_badcase)
                                print('badcase个数：%d'%(np.sum(is_badcase)))
                                print('占总数的：%f'%np.mean(is_badcase))
                                # print(test_X[is_badcase])
                                # print(test_predict_result[0])
                                with open(kwargs.get('badcase_file_path','./badcase.tmp'),'wb') as fout:
                                    pickle.dump(test_X[is_badcase],fout)
                                    pickle.dump(test_y[is_badcase],fout)
                                    pickle.dump(test_predict_result[0][is_badcase],fout)
                                    print('badcase 保存到：%s'%fout.name)


