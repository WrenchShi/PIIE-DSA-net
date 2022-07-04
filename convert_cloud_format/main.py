import h5py
import os
import numpy as np
import pandas as pd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from load_ply import read_ply_data
from prepare_h5 import room_to_blocks
from laspy.file import File


def aseert_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

class lin_data:#林师兄数据处理类
    def __init__(self):
        self.train_files = []
        self.test_files = []
        self.val_files = []
        self.src_file_type = None
        self.dis_file_type = None

    def load_train_file_names(self, train_path):#读取训练数据目录
        train_files = os.listdir(train_path)
        if not train_files:
            print('empty train_files')
            assert False
        self.train_files = [train_path + '/' + x for x in train_files]
        print('load train files done')

    def load_test_file_names(self, test_path):#读取测试数据目录
        test_files = os.listdir(test_path)
        if not test_files:
            print('empty test_files')
            assert False
        self.test_files = [test_path + '/' + x for x in test_files]
        print('load test files done')

    def load_val_file_names(self, val_path):#读取验证数据目录
        val_files = os.listdir(val_path)
        if not val_files:
            print('empty val_files')
            assert False
        self.val_files = [val_path + '/' + x for x in val_files]
        print('load val files done')

    def generate_dis_path(self, dis_path, *params):#生成输出目录
        if (not self.train_files) and (not self.test_files) and (not self.val_files):
            print('empty data path ')
            assert False
        if not os.path.exists(dis_path):
            os.mkdir(dis_path)
        if len(params) > 0 and params[0] == 'train':
            if not os.path.exists(dis_path + '/' + 'train'):
                os.mkdir(dis_path + '/' + 'train')
            print('generate dis train path done')
        if len(params) > 1 and params[1] == 'test':
            if not os.path.exists(dis_path + '/' + 'test'):
                os.mkdir(dis_path + '/' + 'test')
            print('generate dis test path done')
        if len(params) > 2 and params[2] == 'val':
            if not os.path.exists(dis_path + '/' + 'val'):
                os.mkdir(dis_path + '/' + 'val')
            print('generate dis val path done')


    def trans_format(self, src_file_type, dis_file_type):#格式形状改变
        self.src_file_type = src_file_type
        self.dis_file_type = dis_file_type
        mession_type = [self.train_files, self.test_files, self.val_files]
        m_type_names = ['train', 'test', 'val']
        for i in range(len(mession_type)):
            m_type = mession_type[i]
            for s_f in m_type:
                if dis_file_type == 'npy':
                    if src_file_type == 'ply':
                        xyz, rgb, labels = read_ply_data(s_f, with_rgb=True)
                        clouds = np.hstack((xyz, rgb, labels))
                    elif src_file_type == 'las':
                        las_file = File(s_f, mode='r')
                        point_x = np.array(las_file.x).reshape(-1, 1)
                        point_y = np.array(las_file.y).reshape(-1, 1)
                        point_z = np.array(las_file.z).reshape(-1, 1)
                        point_r = np.array(las_file.red).reshape(-1, 1)
                        point_g = np.array(las_file.green).reshape(-1, 1)
                        point_b = np.array(las_file.blue).reshape(-1, 1)
                        point_label = np.array(las_file.raw_classification).reshape(-1, 1)
                        clouds = np.hstack((point_x, point_y, point_z, point_r, point_g, point_b, point_label))
                    elif src_file_type == 'txt':
                        clouds = np.loadtxt(s_f)
                        if clouds.shape[1] == 4:
                            new_clouds = np.zeros((clouds.shape[0], 7))
                            new_clouds[:,0:3] = clouds[:,0:3]
                            new_clouds[:,6:7] = clouds[:,3:4]
                            clouds = new_clouds
                    save_path = s_f.split('/')
                    single_f_name = save_path[-1][:-len(src_file_type)] + dis_file_type
                    save_path = save_path[:-1]
                    save_path.append(m_type_names[i] + '_' + dis_file_type)
                    save_path = '/'.join(save_path)
                    aseert_mkdir(save_path)
                    np.save(save_path + '/' + single_f_name, clouds)

        print('trans done')

def surich_split_numpy(src_path, dis_path):#苏黎世数据数组
    size = 250
    stride = 250
    threshold = 2000
    npy_datas = os.listdir(src_path)
    for npy_data in npy_datas:
        cloud = np.load(src_path + '/' + npy_data)
        cloud[:, 0] -= min(cloud[:, 0])
        cloud[:, 1] -= min(cloud[:, 1])
        limit = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limit[0] - size) / stride)) + 1
        depth = int(np.ceil((limit[1] - size) / stride)) + 1
        cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
        count = 0
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            if np.sum(cond) < threshold:
                continue
            block = cloud[cond, :]
            for i in range(block.shape[0]):
                if block[i][-1] == 15:
                    block[i][-1] = 8
                elif block[i][-1] == 17:
                    block[i][-1] = 9
                block[i][-1] -= 1
            np.save(dis_path + '/' + npy_data[:-4] + '_' + str(count) + '.npy', block)
            count += 1

def save_paconv_h5(fname, pc):#将pc中的数据按部分存成h5
    fp = h5py.File(fname, 'w')
    coords = pc[:, 0:3]
    points = pc[:, 6:12]
    labels = pc[:, 12:13]
    points = np.hstack((coords, points))
    fp.create_dataset('data', data=points, compression='gzip', dtype='float32')
    fp.create_dataset('label', data=labels, compression='gzip', dtype='int64')
    fp.close()
    print('saved:', fname)

#paconv h5 'data': x y z r_ g_ b_ x_ y_ z_   'label': seg_label
def generate_paconv_h5(root_path, src_data_path):
    size = 100
    stride = 80
    threshold = 100
    num_points = 4096
    s3dis_path = root_path + '/' + 's3dis'
    aseert_mkdir(s3dis_path)
    npy_path = s3dis_path + '/' + 'trainval_fullarea'
    aseert_mkdir(npy_path)
    h5_path = s3dis_path + '/' + 'trainval'
    aseert_mkdir(h5_path)
    list_path = s3dis_path + '/' + 'list'
    aseert_mkdir(list_path)
    src_paths = os.listdir(src_data_path)
    train_list = []
    test_list = []
    count = 0
    for src_path in src_paths:
        train_test_src_path = src_data_path + '/' + src_path
        all_files = os.listdir(train_test_src_path)
        for train_test_f in all_files:
            pc = np.load(train_test_src_path + '/' + train_test_f)
            batch = room_to_blocks(pc, size, stride, threshold, num_points)
            num_block = batch.shape[0]
            for i in range(num_block):
                h5_name = str(count) + '.h5'
                if 'train' in src_path:
                    train_list.append('trainval/' + h5_name)
                elif 'test' in src_path:
                    test_list.append('trainval/' + h5_name)
                count += 1
                pc_save = batch[i]
                save_paconv_h5(h5_path + '/' + h5_name, pc_save)
    train_list = '\n'.join(train_list)
    test_list = '\n'.join(test_list)
    train_str = open(list_path + '/' + 'train124.txt', 'w')
    train_str.write(train_list)
    train_str.close()
    train_str2 = open(list_path + '/' + 'train3.txt', 'w')
    train_str2.write(train_list)
    train_str2.close()
    test_str = open(list_path + '/' + 'val3.txt', 'w')
    test_str.write(test_list)
    test_str.close()

class zurich:
    def __init__(self, src_path):
        self.data = os.listdir(src_path)
        self.data = [src_path + '/' + x for x in self.data]
        self.color_bar = {1:[255, 0, 0], 2:[0, 255, 0], 3:[0, 0, 255], 4:[125, 125, 0], 5:[125, 0, 125], 6:[0, 125, 125],
                          7:[255, 125, 0], 15:[255, 0, 125], 17:[0, 255, 125]}

    def txt_to_h5(self, dis_path):
        aseert_mkdir(dis_path)
        self.h5_path = dis_path
        count = 0
        size = 20
        stride = 18
        threshold = 100
        num_points = 4096
        for txt_data in self.data:
            pc = np.loadtxt(txt_data)
            batch = room_to_blocks(pc, size, stride, threshold, num_points)
            num_block = batch.shape[0]
            for i in range(num_block):
                h5_name = str(count) + '.h5'
                count += 1
                pc_save = batch[i]
                save_paconv_h5(dis_path + '/' + h5_name, pc_save)

    def visualize_class(self, dis_path):
        for _ in self.data:
            s_pc = np.load(_)
            npy_name = _.split('/')[-1]
            for i in range(s_pc.shape[0]):
                s_pc[i][3:6] = np.array(self.color_bar[s_pc[i][-1]])
            for key in self.color_bar.keys():
                class_pc = s_pc[np.where(s_pc[:,-1] == key)]
                np.savetxt(dis_path + '/' + npy_name + '_' + str(key) + '.txt', class_pc)
    def visulize(self, dis_path):
        for _ in self.data:
            s_pc = np.load(_)
            npy_name = _.split('/')[-1]
            for i in range(s_pc.shape[0]):
                s_pc[i][3:6] = np.array(self.color_bar[s_pc[i][-1]])
            # np.savetxt(dis_path + '/' + npy_name[:-3] + 'txt', s_pc)
            print(s_pc.shape, npy_name)
    def plot_color_bar(self):
        # rgb = []
        fig = plt.figure()
        ans = fig.add_subplot(111)
        withs = 2
        hights = 1
        color_bars = list(self.color_bar.keys())
        for idx, key in enumerate(color_bars):
            rect = plt.Rectangle(
                (idx*withs, 0),
                withs,
                hights,
                color=np.array(self.color_bar[key])/255.0,
                alpha=1
            )
            ans.add_patch(rect)
        plt.xlim(0, (idx+1)*withs)
        plt.ylim(0,1)
        plt.show()
    def plot_points_num(self):
        area1 = [192196, 7823241, 223546, 1503625, 9501108, 1288012]
        plt.bar(range(len(area1)), area1)
        plt.show()
        # for key in self.color_bar.keys():
        #     rgb.append(self.color_bar[key])
        # rgb = np.array(rgb)/255.0
        # icamp = colors.ListedColormap(rgb, name='1,  2,  3,  4,  5,  6,  7,  15,  17')

        # plt.show(icamp)
    def convert_class(self, src_path, dis_path):
        convert_dic = {0:0, 1:1, 2:2, 3:2, 4:2, 5:3}
        pcs = os.listdir(src_path)
        for pc in pcs:
            cloud = np.load(src_path+ '/' + pc)
            cloud = cloud[np.where(cloud[:, -1] <= 5 )]
            for i in range(cloud.shape[0]):
                cloud[i][-1] = convert_dic[int(cloud[i][-1])]
            np.save(dis_path + '/' + pc, cloud)


class visualize_pc:#存pc
    def __init__(self, file_or_path):
        if '.' in file_or_path.split('/')[-1]:
            self.data = [file_or_path]
        else:
            self.data = os.listdir(file_or_path)
            self.data = [file_or_path + '/' + x for x in self.data]

    def to_txt(self, in_type):
        if in_type == 'paconv_h5':
            for s_data in self.data:
                cloud = h5py.File(s_data)
                cloud = cloud['data']
                cloud = np.array(cloud)
                np.savetxt(s_data[:-2] + 'txt', cloud)

def print_npy_label(path):
    npys = os.listdir(path)
    for npy in npys:
        data = np.load(path + '/' + npy)
        print(np.unique(data[:,-1]))

# Pressthe green button in the gutter to run the script.
if __name__ == '__main__':
    paconv_data = lin_data()
    # paconv_data.load_train_file_names('/home/yym/lhm/iccv2021城市3D语义数据集/train')
    # paconv_data.load_test_file_names('/home/yym/lhm/iccv2021城市3D语义数据集/test')  #同val
    # paconv_data.generate_dis_path('/home/yym/lhm/sensaturban_data', 'train', 'test')
    # paconv_data.trans_format('ply', 'npy')
    # generate_paconv_h5('/home/yym/lhm/PAConv-main/scene_seg/data', )

    # paconv_data.load_train_file_names('/home/yym/lhm/zurich/txt/train')
    # paconv_data.load_test_file_names('/home/yym/lhm/zurich/txt/test')
    # paconv_data.trans_format('txt', 'npy')
    # surich_split_numpy('/home/yym/lhm/zurich/npy/train', '/home/yym/lhm/zurich/train_test_split/train')
    generate_paconv_h5('/home/yym/lhm/PAConv-main/scene_seg/data', '/home/yym/lhm/zurich/train_test_split')
    #
    # zurich = visualize_pc('/home/yym/lhm/zurich/h5_vis')
    # zurich.to_txt('paconv_h5')

    # print_npy_label('/home/yym/lhm/zurich/train_test_split/train')
    # data = zurich('/home/yym/lhm/zurich/npy/all')
    # data.visualize_class('/home/yym/lhm/zurich/npy/class_vis')
    # data.visulize('/home/yym/lhm/zurich/npy/vis')
    # data.plot_color_bar()
    # data.plot_points_num()
    # data.convert_class('/home/yym/lhm/zurich/train_test_split/test', '/home/yym/lhm/zurich/train_test_split/new_test')




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
