import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from tqdm import trange

counter = 14                        #Number of Convolutional Layers in the model
SSIM_Values = np.zeros((counter))
for cut in trange(counter):
    temp_arr = []
    for row in range(0, cut+1):
        fileP = './results/Weedling/Small_Resnet_Full/Class_0/' + str(row+1) + '.png'
        reference_Image = plt.imread(fileP)
        if row+1 != counter:
            for column in range(cut+1, counter):
                fileP = './results/Weedling/Small_Resnet_Full/Class_0/' + str(column+1) + '.png'
                current_Image = plt.imread(fileP)
                temp_arr.append(ssim(reference_Image, current_Image, multichannel=True))
    SSIM_Values[cut] = np.mean(temp_arr)
file_P = "./results/SSIM_CUT_Class_0.csv"
np.savetxt(file_P, SSIM_Values, delimiter=",", fmt="%f")