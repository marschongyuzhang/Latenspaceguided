import json
from labelme.utils.shape import labelme_shapes_to_label
import numpy as np
import cv2
import os

def test():
    image_origin_path = r"./home/mars/chongyu_project/Inside-Outside-Guidance-master/test_img/n123.png/" #the original pic
    image = cv2.imread(image_origin_path)

    json_path = r'./home/mars/chongyu_project/Inside-Outside-Guidance-master/test_img/n123.json/' #the json file
    data = json.load(open(json_path))

    lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
    print(lbl_names)
    mask=[]
    class_id=[]
    for i in range(1,len(lbl_names)): 
        key = [k for k, v in lbl_names.items() if v == i][0]
        print(key)
        mask.append((lbl==i).astype(np.uint8)) 
        class_id.append(i) 
    print(class_id)
    # print(mask)
    # print(class_id)
    mask=np.asarray(mask,np.uint8)
    mask=np.transpose(np.asarray(mask,np.uint8),[1,2,0])


def get_finished_json(root_dir):
    import glob
    json_filter_path = root_dir + "\*.json"
    jsons_files = glob.glob(json_filter_path)
    return jsons_files


def get_dict(json_list):
    dict_all = {}
    for json_path in json_list:
        dir,file = os.path.split(json_path)
        file_name = file.split('.')[0]
        image_path = os.path.join(dir,file_name+'.jpg')
        dict_all[image_path] = json_path
    return dict_all


def process(dict_):
    for image_path in dict_:
        mask = []
        class_id = []
        key_ = []
        image = cv2.imread(image_path)
        json_path = dict_[image_path]
        data = json.load(open(json_path))
        lbl, lbl_names = labelme_shapes_to_label(image.shape, data['shapes'])
        for i in range(1, len(lbl_names)):
            key = [k for k, v in lbl_names.items() if v == i][0]
            mask.append((lbl == i).astype(np.uint8)) 
            class_id.append(i) 
            key_.append(key)
        mask = np.asarray(mask, np.uint8)
        mask = np.transpose(np.asarray(mask, np.uint8), [1, 2, 0])
        image_name = os.path.basename(image_path).split('.')[0]
        dir_ = os.path.dirname(image_path)
        for i in range(0, len(class_id)):
            image_name_ = "{}_mask_{}_{}.jpg".format(image_name,key_[i],i)
            dir_path =  os.path.join(dir_, 'mask',key_[i]) 
            checkpath(dir_path)
            image_path_ = os.path.join(dir_path,image_name_)
            print(image_path_)
            retval, im_at_fixed = cv2.threshold(mask[:,:,i], 0, 255, cv2.THRESH_BINARY)
            cv2.imwrite(image_path_, im_at_fixed)


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    root_dir = r'./home/mars/chongyu_project/Inside-Outside-Guidance-master/test_img/' # the root
    json_file = get_finished_json(root_dir)
    image_json = get_dict(json_file)
    process(image_json)
