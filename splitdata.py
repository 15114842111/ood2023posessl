#将我们的数据集转化成验证时候要求的数据集
import os
import shutil
path_pic_ori=r'/data/gmmm/data/images'
path_pic_des= r'/data/gmmm/data_split'
dir_pic_ori = os.listdir(path_pic_ori)
if not os.path.exists(path_pic_des):
    os.makedirs(path_pic_des)
for pic in dir_pic_ori:
    # print(pic)
    ty=pic.split('.')[-1]
    # print(ty)
    cat_name=pic.split('.')[0].split('_')[-1]
    # print(cat_name)
    path_new=os.path.join(path_pic_des,cat_name)
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    path_new_1 = os.path.join(path_new,pic)
    print(path_new_1)
    path_pic=os.path.join(path_pic_ori,pic)
    print(path_pic)
    shutil.copyfile(path_pic,path_new_1)
    # shutil.copyfile(path_new, path_new)
