import model_torch
import modules_torch
import _utils_torch
from _utils_torch import *
import loss

def preprocessing(train_data_list, test_data_list):
    
    label = 1
    X_train = []
    Y_train = []
    for item in tqdm(train_data_list):
        folder = '../Data/' + item 
        items = glob.glob(folder + '/*.nii')
        for file in items:
            if 'label' in file:
                sam_mask = sitk.ReadImage(file)
                sam_mask = sitk.GetArrayFromImage(sam_mask).astype(np.uint8)
                sam_mask[sam_mask!=label] = 0
                sam_mask[sam_mask==label] = 1
                for _slice in range(len(sam_mask)):
                    slice_mask = resize(sam_mask[_slice,...],(512,512),anti_aliasing=False,preserve_range=True,order=0)
                    Y_train.append(slice_mask)
            else:
                sam_img = sitk.ReadImage(file)
                sam_img = sitk.GetArrayFromImage(sam_img)
                sam_img = normalize_dcm(sam_img).astype(np.float32)
                for _slice in range(len(sam_img)):
                    slice_img = resize(sam_img[_slice,...],(512,512),anti_aliasing=True,preserve_range=True,order=0)
                    X_train.append(slice_img)
    
    
    X_test = []
    Y_test = []
    for item in tqdm(test_data_list):
        folder = '../Data/' + item 
        items = glob.glob(folder + '/*.nii')
        for file in items:
            if 'label' in file:
                sam_mask = sitk.ReadImage(file)
                sam_mask = sitk.GetArrayFromImage(sam_mask).astype(np.uint8)
                sam_mask[sam_mask!=label] = 0
                sam_mask[sam_mask==label] = 1
                for _slice in range(len(sam_mask)):
                    slice_mask = resize(sam_mask[_slice,...],(512,512),anti_aliasing=False,preserve_range=True,order=0)
                    Y_test.append(slice_mask)
            else:
                sam_img = sitk.ReadImage(file)
                sam_img = sitk.GetArrayFromImage(sam_img)
                sam_img = normalize_dcm(sam_img).astype(np.float32)
                for _slice in range(len(sam_img)):
                    slice_img = resize(sam_img[_slice,...],(512,512),anti_aliasing=True,preserve_range=True,order=0)
                    X_test.append(slice_img)
    
            
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

    return X_train, Y_train, X_test, Y_test