import torch
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchsummary import summary
import os
import glob

from .model import predict_image, show_train_images


class Question5:
    # def showDataAugmentation(self, imgPath):
    #     if imgPath == None:
    #         print('Please load the image.')
    #     else:
    #         imgRotation = self.showRandomRotation(imgPath)
    #         imgResized = self.showRandomResizedCrop(imgPath)
    #         imgFlipped = self.showRandomHorizontalFlip(imgPath)
    #         result = self.getConcatH(imgRotation, imgResized, imgFlipped)
    #         result.show()
    #         # result.save('Q5/augmentation.png')


    def showRandomRotation(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomRotation(degrees=(0, 180))    # rotated degree from 0 to 180
        img = transfrom(img)
        # img.show()
        return img

    def showRandomVertivalFlip(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomVerticalFlip(p=1) # p is prob of upside down
        img = transfrom(img)
        # img.show()
        return img

    def showRandomResizedCrop(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomResizedCrop(size=img.size, scale=(0.05, 0.95)) # random cropped size is 0.05x to 0.99x
        img = transfrom(img)
        # img.show()
        return img

    def showRandomHorizontalFlip(self, imgPath):
        img = Image.open(imgPath)
        transfrom = transforms.RandomHorizontalFlip(p=0.5)   # filp rate is 1/2
        img = transfrom(img)
        # img.show()
        return img

    # mix the images horizontally
    def getConcatH(self, img1, img2, img3):
        concatenated = Image.new('RGB', (img1.width + img2.width + img3.width, img1.height))
        concatenated.paste(img1, (0, 0))
        concatenated.paste(img2, (img1.width, 0))
        concatenated.paste(img3, (img1.width + img2.width, 0))
        return concatenated

    def showDataAugmentation(self):
        dirName = "/Users/eddie/Desktop/CvDl_HW1_Dataset/Q5_image/Q5_1"
        filePaths = glob.glob(os.path.join(dirName, "*.png"))
        fig, axes = plt.subplots(3, 3)
        counter = 0
        for i in range(0, 3):
            for j in range(0, 3):
                img = filePaths[counter]
                num = counter % 3
                if num == 0:
                    showimg = self.showRandomRotation(img)
                elif num == 1:
                    showimg = self.showRandomVertivalFlip(img)
                elif num == 2:
                    showimg = self.showRandomHorizontalFlip(img)
                axes[i, j].imshow(showimg)
                axes[i, j].set_title(img[39:-4])
                axes[i, j].set_axis_off()
                counter += 1
        plt.show()

    def showModelStructure(self):
        model = torchvision.models.vgg19_bn(num_classes = 10)
        # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # if device == 'cuda:0':
        #     model = torch.load('Q5/model_vgg19.pth')
        # else:
        #     model = torch.load('Q5/model_vgg19.pth', map_location ='cpu')

        summary(model, (3, 224, 224))   # show model structure

    def makeAccuracyAndLoss(self):
        imgAcc = cv2.imread('Q5/accuracy.png')
        imgLoss = cv2.imread('Q5/loss.png')
        # result = np.concatenate((imgAcc, imgLoss), axis=0)  # concat two pictures together
        img = cv2.imread("Q5/result.png")
        print(img.shape)
        img = cv2.resize(img,(1000,600))
        cv2.namedWindow('Accuracy & Loss')
        cv2.imshow('Accuracy & Loss', img)

        # window size depends on resized image 

        cv2.resizeWindow('Accuracy & Loss', img.shape[1], img.shape[0])
        
        # exit
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # cv2.imwrite('Q5/result.png', result)

    def showInference(self, imgPath):
        if imgPath == None:
            print('Please load the image.')
        else:
            eval, label, class_names = predict_image(imgPath=imgPath)
            print(eval)
            plt.bar(class_names,np.array(eval[0].tolist()))
            plt.xticks(rotation=45)
            plt.savefig('Q5/prob.png')
            plt.close()
            # plt.show(img)
            img = cv2.imread("Q5/prob.png")

            cv2.namedWindow('prob of each class')
            cv2.imshow('prob of each class', img)

            cv2.resizeWindow('prob of each class', img.shape[1], img.shape[0])
            
            # exit
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        print(class_names)
        # print(label)
        return class_names     



if __name__ == '__main__':
    print('This is Q5')
    print('Do run run this file')