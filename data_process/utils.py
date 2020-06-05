import os
import numpy as np
import pandas as pd
import logging

# remember our working directory is .../SIGIR
def create_dataset_if_not_exist(paths, start, past_length , pred_length , slice , timestep , freq):
    for name in paths:
        path = paths[name]
        if not os.path.exists(path):
            logging.info('there is no dataset [%s] , creating...' % name)
            os.system('python data_process/preprocessing.py -st={} -d={} -t={} -p={} -s={} -n={} -f={}'
                      .format( start
                              , name
                              , past_length
                              , pred_length
                              , slice
                              , timestep
                              , freq))
        else:
            logging.info(' dataset [%s] was found , good~~~' % name)


def add_time_mark_to_file(path):
    '''
    给重复的文件名添加 time 字段
    :param path: 路径
    :return: 返回新的路径名
    '''
    count = 1
    if not os.path.exists(path):
        return path
    file_name_list = os.path.splitext(os.path.basename(path))
    father = os.path.split(path)[0]

    new_path = os.path.join(father ,file_name_list[0]+'_%d'%(count)+file_name_list[1])
    while os.path.exists(new_path):
        count += 1
        new_path = os.path.join(father, file_name_list[0] + '_%d'%(count) + file_name_list[1])

    return new_path