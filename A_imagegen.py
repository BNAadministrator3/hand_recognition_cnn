#This file is regarding the image inverse generation
from cnn_utils import *
import os
import shutil

base = os.path.join(os.getcwd(),'datasets','specific')
if os.path.exists(base):
    shutil.rmtree(base)
os.makedirs(base)
for i in range(6):
    os.makedirs(os.path.join(base,'train',str(i)))
    os.makedirs(os.path.join(base,'test', str(i)))

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
# index = 6
# plt.imsave(os.path.join(base,'2','trial'),X_train_orig[index])
# plt.imshow(X_train_orig[index])
# print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
# plt.show()


for index in range(X_train_orig.shape[0]):
    dst_dir = os.path.join(base,'train', str(np.squeeze(Y_train_orig[:, index])))
    counts = len(os.listdir(dst_dir))
    plt.imsave(os.path.join(dst_dir, str(np.squeeze(Y_train_orig[:, index])) +'_' + str(counts)), X_train_orig[index])
for index in range(X_test_orig.shape[0]):
    dst_dir = os.path.join(base,'test', str(np.squeeze(Y_test_orig[:, index])))
    counts = len(os.listdir(dst_dir))
    plt.imsave(os.path.join(dst_dir, str(np.squeeze(Y_test_orig[:, index])) +'_' + str(counts)), X_test_orig[index])
