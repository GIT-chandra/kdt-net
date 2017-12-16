'''
This script processes all models and prepares numpy arrays ready to be used while model training.

Specifically, it creates the KD-Trees, etracts the leaves and maintains the jumbled indices for use later.
'''

from kd_helpers import *
import glob
import numpy as np
import os

class_names = {
"02691156":"Airplane_02691156",
"02773838":"Bag_02773838",
"02954340":"Cap_02954340",
"02958343":"Car_02958343",
"03001627":"Chair_03001627",
"03261776":"Earphone_03261776",
"03467517":"Guitar_03467517",
"03624134":"Knife_03624134",
"03636649":"Lamp_03636649",
"03642806":"Laptop_03642806",
"03790512":"Motorbike_03790512",
"03797390":"Mug_03797390",
"03948459":"Pistol_03948459",
"04099429":"Rocket_04099429",
"04225987":"Skateboard_04225987",
"04379243":"Table_04379243"}

# possible sizes after KD-Tree is prepared
INP_SZ_1 = 1024
INP_SZ_2 = 2048
INP_SZ_3 = 4096

def get_fname(folder_name,suffix):
    global class_names
    words = folder_name.split('/')
    return class_names[words[len(words)-1]] + '_' + suffix

def augment_kd(kd_leaves,kd_inds):
    '''
    This makes sure that output is 2048 in length, for compatibility with model
    '''
    if len(kd_leaves) == INP_SZ_2:
        return [kd_leaves],[kd_inds]
    elif len(kd_leaves) == INP_SZ_1:
        kdl_rand = np.zeros(np.shape(kd_leaves))
        # add random noise to point coordinates
        kdl_rand[:,:,0] = kd_leaves[:,:,0] + np.random.randn(INP_SZ_1,3)
        # use the same split information
        kdl_rand[:,:,1] = kd_leaves[:,:,1]
        ind_set1 = [int(f) for f in list(np.linspace(0,INP_SZ_2 - 2,INP_SZ_1))]
        ind_set2 = [int(f) for f in list(np.linspace(1,INP_SZ_2 - 1,INP_SZ_1))]
        aug_kdl = np.zeros((INP_SZ_2,3,2))
        aug_kdi = np.zeros(INP_SZ_2)
        aug_kdl[ind_set1] = kd_leaves
        aug_kdl[ind_set2] = kdl_rand
        aug_kdi[ind_set1] = kd_inds
        aug_kdi[ind_set2] = kd_inds
        return [aug_kdl],[aug_kdi]
    else: # send as two separate models
        ind_set1 = [int(f) for f in list(np.linspace(0,INP_SZ_3 - 2,INP_SZ_2))]
        ind_set2 = [int(f) for f in list(np.linspace(1,INP_SZ_3 - 1,INP_SZ_2))]
        kdl1 = kd_leaves[ind_set1]
        kdl2 = kd_leaves[ind_set2]
        kdi1 = kd_inds[ind_set1]
        kdi2 = kd_inds[ind_set2]
        return [kdl1,kdl2],[kdi1,kdi2]

if __name__ = '__main__':
    # Collecting all of the data
    data_folders = ["./data/train_data/*","./data/val_data/*","./data/test_data/*"]
    label_folders = ["./data/train_label/*","./data/val_label/*"]

    data_fnames = ["X_train.npy","X_val.npy","X_test.npy"]
    ind_map_fnames = ["ind_map_train.npy","ind_map_val.npy","ind_map_test.npy"]
    label_fnames = ["y_train.npy","y_val.npy"]
    print("Processing data..")

    for i in range(1):  #iterating over train, val
        main_data_folder = data_folders[i]
        main_label_folder = label_folders[i]

        # sort to get same ordering and hence avoid mapping the labels of one model onto another
        data_classes = sorted(glob.glob(main_data_folder))
        label_classes = sorted(glob.glob(main_label_folder))
        for data_class,label_class in zip(data_classes,label_classes):
            print(data_class)
            model_files = sorted(glob.glob(data_class + '/*'))
            label_files = sorted(glob.glob(label_class + '/*'))
            data = []   # to store the precessed model
            ind_maps = []   # indices of points, which are now a permutation of the original
            labels = [] # labels of the points in the model
            for model_file,label_file in zip(model_files,label_files):
                print(model_file)
                pts = read_pts(model_file)
                lbls = read_labels(label_file)
                kd_leaves,kd_inds = create_kd_tree(pts)
                kdls,kdis = augment_kd(kd_leaves,kd_inds)
                for kdl,kdi in zip(kdls,kdis):
                    data.append(kdl)
                    ind_maps.append(kdi)
                    inds_for_lbls = [int(f) for f in kdi]
                    labels.append(lbls[inds_for_lbls])
            np.save(get_fname(data_class,data_fnames[i]),data)
            np.save(get_fname(data_class,ind_map_fnames[i]),ind_maps)
            np.save(get_fname(label_class,label_fnames[i]),labels)


    # Processing the test set (only points)
    print("Processing test data...")
    test_folder = data_folders[2]
    data_classes = sorted(glob.glob(test_folder))
    for data_class in data_classes:
        print(data_class)
        model_files = sorted(glob.glob(data_class + '/*'))
        data = []
        ind_maps = []
        for model_file in model_files:
            print(model_file)
            pts = read_pts(model_file)
            kd_leaves,kd_inds = create_kd_tree(pts)
            kdl,kdi = augment_kd(kd_leaves,kd_inds)
            for l,i in zip(kdl,kdi):
                data.append(l)
                ind_maps.append(i)
        np.save(get_fname(data_class,data_fnames[2]),data)
        np.save(get_fname(data_class,ind_map_fnames[2]),ind_maps)
