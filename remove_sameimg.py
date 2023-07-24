import os 
path_mask_old = "/home/mars/iog/Inside-Outside-Guidance-master/test_img/mask1"
path_pic4 = "/home/mars/iog/Inside-Outside-Guidance-master/test_img/n02992211"

files_mask_old = os.listdir(path_mask_old)
#print(files_mask_old)
files_path4 = os.listdir(path_pic4)

for i in files_mask_old:
    for j in files_path4:
        name_mask1 = i.split('.')[0]
        name_mask2 = j.split('.')[0]
        #print(name_mask1,name_mask2)
        if name_mask1 == name_mask2:
            
            file_path = os.path.join(path_pic4,j)
            Remove = os.remove(file_path)
            #print(Remove)
    
