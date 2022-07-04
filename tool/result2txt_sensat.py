import pickle
import os
import numpy as np


data_root = '/home/t2/linhemin/PAConv-main/scene_seg/data/s3dis/Area5_npy'
pick_root = '/home/t2/linhemin/PAConv-main/scene_seg/exp/s3dis/pointnet2_paconv/result/best_epoch/val5_0.5/pred_5.pickle'
save_path = '/home/t2/linhemin/PAConv-main/scene_seg/pred_cloud_orin_sensat'
if not os.path.exists(save_path):
    os.mkdir(save_path)

color_map = {0:'[85, 107, 47]', 1:'[0, 255, 0]', 2:'[255, 170, 0]', 3:'[40, 50, 100]', 4:'[0, 0, 0]',
            5:'[0, 0, 255]', 6:'[255, 0, 255]', 7:'[201, 201, 201]', 8:'[89, 47, 95]', 9:'[255, 0, 0]',
            10:'[255, 255, 0]', 11:'[0, 255, 255]', 12:'[0, 190, 255]'
             }


def sample_cloud(cloud, num_samples):
    n = cloud.shape[0]
    if n >= num_samples:
        indices = np.random.choice(n, num_samples, replace=False)
    else:
        indices = np.random.choice(n, num_samples - n, replace=True)
        indices = list(range(n)) + list(indices)
    sampled = cloud[indices, :]
    return sampled

npys = os.listdir(data_root)
npys.sort()
for i in range(len(npys)):
    npy = npys[i]
    cloud = np.load(data_root + '/' + npy)
    with open(pick_root, 'rb') as file:
        data = pickle.load(file)
    #cloud[:, -2] = data['pred'][i]
    #new_cloud = np.concatenate((cloud[:, 0:3], cloud[:,-2].reshape(cloud.shape[0], 1)), axis=1)
    new_cloud = cloud[:, 0:6]
    for j in range(new_cloud.shape[0]):
        color_li = color_map[int(data['pred'][i][j])][1:-1].split(', ')
        color_li = [int(x) for x in color_li]
        new_cloud[j][3:6] = np.array(color_li)
    num_samples = new_cloud.shape[0] // 2
    # new_cloud = sample_cloud(new_cloud, num_samples)
    np.savetxt(save_path + '/' +npy[:-3] + 'txt', new_cloud, fmt='%f %f %f %d %d %d')

