# encoding=utf8
"""
    Author:  'jdwang'
    Date:    'create date: 2016-11-28'; 'last updated date: 2016-11-28'
    Email:   '383287471@qq.com'
    Describe: 图像中一些常用的函数
                - load_image_from_file： 将 bmp 等后缀的图片，转为 多维数组（array）
"""
from __future__ import print_function
import Image
import os
import numpy as np

class DataUtil(object):
    """图像的处理工具类，提供常用函数：
        - load_image_from_file： 将 bmp 等后缀的图片，转为 多维数组（array）

    """
    def __init__(self):
        pass

    def load_image_from_file(self,file_path):
        """  加载一张原始图片 ，并转为数组

        Parameters
        ----------
        file_path : str
            图片地址

        Returns
        -------

        Examples
        -------
        >>> dutil = DataUtil()
        >>> image_path = '/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/9905_1128/data/5/201611281102544772.bmp'
        >>> dutil.load_image_from_file(image_path)

        """
        image = Image.open(file_path)
        array = np.asarray(image)
        # print(array.shape)
        return array


if __name__ == '__main__':
    dutil = DataUtil()
    dutil.load_image_from_file('/home/jdwang/PycharmProjects/digitRecognition/int_weight_predict/9905_1128/data/5/201611281102544772.bmp')