import cv2
import numpy as np

class Question4:
    def showKeypoints(self, image1Path):
        if image1Path == None:
            print('Please load the image.')

        else:
            imgGray = cv2.imread(image1Path, 0) # read in gray scale
            sift = cv2.xfeatures2d.SIFT_create()
            keypoints = sift.detect(imgGray, None)

            # keypoints, _ = sift.detectAndCompute(imgGray, None)
            imgGray = cv2.drawKeypoints(imgGray, keypoints=keypoints, outImage=None, color=(0, 255, 0)) # draw green keypoints
            
            resize_width = 600
            resize_height = 600

            resized_image = cv2.resize(imgGray,(resize_width,resize_height))
            cv2.namedWindow('1.4 Keypoints', cv2.WINDOW_NORMAL)
            print(resized_image.shape)
            cv2.imshow('1.4 Keypoints', resized_image)

            # window size depends on resized image 
            cv2.resizeWindow('1.4 Keypoints', resized_image.shape[0], resized_image.shape[1])
            
            # exit
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def showMatchedKeypoints(self, image1Path, image2Path):
        if image1Path == None or image2Path == None:
            print('Please load the images.')

        else:
            img1Gray = cv2.imread(image1Path, 0)    # read in gray scale
            img2Gray = cv2.imread(image2Path, 0)    # read in gray scale
            sift = cv2.xfeatures2d.SIFT_create()

            # find the keypoints and descriptors with SIFT
            keypoints1, descriptors1 = sift.detectAndCompute(img1Gray, None)
            keypoints2, descriptors2 = sift.detectAndCompute(img2Gray, None)

            # FLANN parameters
            FLANN_INDEX_KDTREE = 1
            indexParams = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            searchParams = dict(checks=50)   # or pass empty dictionary

            flann = cv2.FlannBasedMatcher(indexParams, searchParams)
            matches = flann.knnMatch(descriptors1, descriptors2, k=2)

            # Need to draw only good matches, so create a mask
            matchesMask = np.zeros((len(matches), 2), np.int8)  # initialise the matrix
            
            # ratio test as per Lowe's paper
            for i, (m, n) in enumerate(matches):
                if m.distance < 0.75 * n.distance:
                    # matchesMask[i]=[1, 0]
                    matchesMask[i][0] = 1


            matchesMask = matchesMask.tolist()  # change type to list

            # draw the matches
            # matched = cv2.drawMatchesKnn(img1=img1Gray, keypoints1=keypoints1, img2=img2Gray, keypoints2=keypoints2, matches1to2=matches, 
            #                                 outImg=None, matchColor=(0, 255, 255), singlePointColor=(0, 255, 0), matchesMask=matchesMask)

            matched = cv2.drawMatchesKnn(img1=img1Gray, keypoints1=keypoints1, img2=img2Gray, keypoints2=keypoints2, matches1to2=matches, 
                                            outImg=None, flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS , singlePointColor=(0, 255, 0), matchesMask=matchesMask)
            
            matched_resized = cv2.resize(matched, (1200, 600))
            cv2.imshow('1.4 Matched keypoints', matched_resized) # show the result
            cv2.resizeWindow('1.4 Keypoints', matched_resized.shape[0], matched_resized.shape[1])
            

            # exit
            cv2.waitKey(0)
            cv2.destroyAllWindows()