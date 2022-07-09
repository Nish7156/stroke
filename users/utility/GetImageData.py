import os
from django.conf import settings
class ProcessCtScanImage:
    def preprocess(self,filename):
        print("Working on images")
        filepath = settings.MEDIA_ROOT + "\\" + filename
        print("File Images ",filepath)
        import matplotlib.pyplot as plt
        import cv2
        from skimage.feature import greycomatrix, greycoprops
        # PATCH_SIZE = 20
        # image = cv2.imread("i.jpg",0)

        sky_locations = [(288, 37), (81, 37), (151, 127), (238, 172), (403, 170), (501, 170), (511, 47), (336, 9),
                         (173, 14), (173, 14)]
        sky_test = [(41, 92), (41, 198), (151, 217), (230, 143), (176, 62)]

        grass_locations = [(400, 366), (225, 320), (163, 365), (111, 300), (26, 377), (26, 283), (264, 356), (446, 378),
                           (532, 264), (567, 359)]
        grass_test = [(548, 385), (487, 342), (443, 310), (378, 291), (88, 278)]

        PATCH_SIZE = 20
        image = cv2.imread(filepath, 0)
        plt.imshow(image, cmap='gray')
        plt.show()
        print(image.shape)
        import numpy as np
        grass_test = [(548, 385), (487, 342), (443, 310), (378, 291), (88, 278)]
        sky_test = [(41, 92), (41, 198), (151, 217), (230, 143), (176, 62)]
        grass_patches = []
        xs = []
        x = []
        y = []
        contrast = []
        dissimilarity = []
        homogeneity = []
        asm = []
        energy = []
        correlation = []
        for loc in grass_test:
            grass_patches = (image[loc[1]:loc[1] + PATCH_SIZE, loc[0]:loc[0] + PATCH_SIZE])
            x = []
            # glcm = greycomatrix(grass_patches, [5], [0], 256, symmetric=True, normed=True)
            glcm = greycomatrix(grass_patches, [1, 2], [0, np.pi / 2], 256, symmetric=True, normed=True)
            xs.append(greycoprops(glcm, 'contrast')[0, 0])
            xs.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            xs.append(greycoprops(glcm, 'homogeneity')[0, 0])
            xs.append(greycoprops(glcm, 'ASM')[0, 0])
            xs.append(greycoprops(glcm, 'energy')[0, 0])
            xs.append(greycoprops(glcm, 'correlation')[0, 0])
            x.append(xs)
            y.append([1])
            contrast.append(greycoprops(glcm, 'contrast')[0, 0])
            dissimilarity.append(greycoprops(glcm, 'dissimilarity')[0, 0])
            homogeneity.append(greycoprops(glcm, 'homogeneity')[0, 0])
            asm.append(greycoprops(glcm, 'ASM')[0, 0])
            energy.append(greycoprops(glcm, 'energy')[0, 0])
            correlation.append(greycoprops(glcm, 'correlation')[0, 0])
        # print( len(x[0]))
        # print (x[0])
        # print (len(x))
        # print (y)

        cont = sum(contrast) / len(contrast)
        dissim = sum(dissimilarity) / len(dissimilarity)
        homogen = sum(homogeneity) / len(homogeneity)
        asmval = sum(asm) / len(asm)
        energ = sum(energy) / len(energy)
        correl = sum(correlation) / len(correlation)
        v = (cont+dissim+homogen+energ+correl)/5
        if v>100:
            lblclass = 1
        else:
            lblclass = 0
        rsltdict = { "contrast":cont,
                     'dissimilarity':dissim,
                     'homogeneity': homogen,
                     'energy':energ,
                     'ASM':asmval,
                     'correlation':correl,
                     'lblclass':lblclass
                     }

        print(rsltdict)
        os.remove("media/" + filename)
        #os.remove(filename)
        return rsltdict
