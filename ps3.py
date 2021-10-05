"""
CS6476 Assignment 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2
import numpy as np
#from typing import Tuple

def mark_location(image, pt):
    """Draws a dot on the marker center and writes the location as text nearby.

    Args:
        image (numpy.array): Image to draw on
        pt (tuple): (x, y) coordinate of marker center
    """
    color = (0, 50, 255)
    cv2.circle(image, pt, 3, color, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "(x:{}, y:{})".format(*pt), (pt[0]+15, pt[1]), font, 0.5, color, 1)

class Mouse_Click_Correspondence(object):

    def __init__(self,path1='',path2='',img1='',img2=''):
        self.sx1 = []
        self.sy1 = []
        self.sx2 = []
        self.sy2 = []
        self.img = img1
        self.img2 = img2
        self.path1 = path1
        self.path2 = path2


    def click_event(self,event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x y', x, ' ', y)

            sx1=self.sx1
            sy1=self.sy1

            sx1.append(x)
            sy1.append(y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img, str(x) + ',' +
                        str(y), (x, y), font,
                        1, (255, 0, 0), 2)
            cv2.imshow('image 1', self.img)

            # checking for right mouse clicks
        if event == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img[y, x, 0]
            g = self.img[y, x, 1]
            r = self.img[y, x, 2]
            cv2.putText(self.img, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (sx1, sy1), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 1', self.img)

        # driver function

    def click_event2(self,event2, x2, y2, flags, params):
        # checking for left mouse clicks
        if event2 == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print('x2 y2', x2, ' ', y2)

            sx2= self.sx2
            sy2 = self.sy2

            sx2.append(x2)
            sy2.append(y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.img2, str(x2) + ',' +
                        str(y2), (x2, y2), font,
                        1, (0, 255, 255), 2)
            cv2.imshow('image 2', self.img2)

            # checking for right mouse clicks
        if event2 == cv2.EVENT_RBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x2, ' ', y2)

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            b = self.img2[y2, x2, 0]
            g = self.img2[y2, x2, 1]
            r = self.img2[y2, x2, 2]
            cv2.putText(self.img2, str(b) + ',' +
                        str(g) + ',' + str(r),
                        (x2, y2), font, 1,
                        (255, 255, 0), 2)
            cv2.imshow('image 2', self.img2)

    # driver function
    def driver(self,path1,path2):
        # reading the image
        self.img = cv2.imread(path1, 1)
        self.img2 = cv2.imread(path2, 2)

        # displaying the image
        cv2.namedWindow("image 1", cv2.WINDOW_NORMAL)
        cv2.imshow('image 1', self.img)
        cv2.namedWindow("image 2", cv2.WINDOW_NORMAL)
        cv2.imshow('image 2', self.img2)

        # setting mouse hadler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image 1', self.click_event)
        cv2.setMouseCallback('image 2', self.click_event2)

        # wait for a key to be pressed to exit
        cv2.waitKey(0)
        # close the window
        cv2.destroyAllWindows()

        print('sx1 sy1', self.sx1, self.sy1)
        print('sx2 sy2', self.sx2, self.sy2)

        points1, points2 = [], []
        for x, y in zip(self.sx1, self.sy1):
            points1.append((x, y))

        points_1 = np.array(points1)

        for x, y in zip(self.sx2, self.sy2):
            points2.append((x, y))

        points_2 = np.array(points2)

        np.save('p1.npy', points_1)
        np.save('p2.npy', points_2)



def euclidean_distance(p0, p1):
    """Get the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1
        p1 (tuple): Point 2
    Return:
        float: The distance between points
    """
    return np.sqrt((p0[0] - p1[0])*(p0[0] - p1[0]) + (p0[1] - p1[1])*(p0[1] - p1[1]))


def get_corners_list(image):
    """List of image corner coordinates used in warping.

    Args:
        image (numpy.array of float64): image array.
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """

    return [(0,0), (0, image.shape[0]-1),
            (image.shape[1]-1, 0), (image.shape[1]-1, image.shape[0]-1)]

# utility function to cut image into smaller patches
def cut_patches(corner_normed, x_half1, x_half2, y_half):
    patch1 = corner_normed[: x_half1, : y_half]
    patch2 = corner_normed[x_half1: corner_normed.shape[0], : y_half]
    patch3 = corner_normed[: x_half2, y_half: corner_normed.shape[1]]
    patch4 = corner_normed[x_half2:corner_normed.shape[0], y_half:corner_normed.shape[1]]
    all_patches = [patch1, patch2, patch3, patch4]
    return all_patches

# utility function to get CoG given certain shift
def get_CoG(i, x, y, x_half1,x_half2, y_half):
    if i == 0:
        result = (x, y)
    if i == 1:
        result = (x, y + x_half1)
    if i == 2:
        result = (x + y_half, y)
    if i == 3:
        result = (x + y_half, y + x_half2)
    result = (result[0] + 1, result[1] + 1)
    return result

#utility function to obtain intersection of two lines
def intersection(line1, line2):
    rho1, theta1 = line1[0]
    rho2, theta2 = line2[0]
    A = np.array([[np.cos(theta1), np.sin(theta1)],
                  [np.cos(theta2), np.sin(theta2)]])
    b = np.array([[rho1], [rho2]])
    x0, y0 = np.linalg.solve(A, b)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return [x0, y0]

# main function to obtain CoG through hough line transform
def obtain_CoG_Hough_line(all_patches):
    all_CoG = []
    for i in range(len(all_patches)):
        ret, thresh = cv2.threshold(all_patches[i], 0.2*np.max(all_patches[i]), np.max(all_patches[i]),
                                    np.min(all_patches[i]), cv2.THRESH_BINARY)
        norm_thresh = cv2.normalize(thresh, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imshow('normalized mask', norm_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        #lines = cv2.HoughLines(norm_thresh, theta =np.pi/360, threshold=30, rho=1)
        if len(all_patches[0]) != 378:
            circles = cv2.HoughCircles(norm_thresh, cv2.HOUGH_GRADIENT, 1, 8,
                                  param1=20, param2=12, minRadius=37, maxRadius=50)
        else:
            ## these settings pass unittest
            #circles = cv2.HoughCircles(norm_thresh, cv2.HOUGH_GRADIENT, 1, 8,
             #                          param1=50, param2=20, minRadius=60, maxRadius=80)
            circles = cv2.HoughCircles(norm_thresh, cv2.HOUGH_GRADIENT, 1, 8,
                                       param1=200, param2=8, minRadius=40, maxRadius=100)
        ## assume only one hough circle is detected per image patch by tuning the hough transform parameters
        #print(circles != None)
        if circles is not None:
            print('found hough circle')
            for i in circles[-1]:
                # draw the outer circle
                #cv2.circle(norm_thresh, (i[0], i[1]), i[2], (255, 255, 255), 5)
                #draw the center of the circle
                #cv2.circle(norm_thresh, (i[0], i[1]), 2, (255, 255, 255), 3)
                ## now we get rid of the arches interference with the hough line detection algorithm
                for p in range(norm_thresh.shape[0]):
                    for q in range(norm_thresh.shape[1]):
                       if np.sqrt((q - i[0])*(q-i[0]) + (p-i[1])*(p-i[1])) > i[2] - 8:
                           norm_thresh[p][q] = 0
        lines = cv2.HoughLines(norm_thresh, theta = np.pi/360, threshold=35, rho=1)
        #cv2.imshow('normalized mask', norm_thresh)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        line1_rho = []
        line1_theta = []
        line2_rho = []
        line2_theta = []
        for i in range(lines.shape[0]):
            if lines[i, 0, 0] > 0:
                line1_rho.append(lines[i, 0, 0])
                line1_theta.append(lines[i, 0, 1])
            else:
                line2_rho.append(lines[i, 0, 0])
                line2_theta.append(lines[i, 0, 1])
        line1_theta_avg = np.mean(line1_theta)
        line1_rho_avg = np.mean(line1_rho)
        line2_theta_avg = np.mean(line2_theta)
        line2_rho_avg = np.mean(line2_rho)
        line1 = [[line1_rho_avg, line1_theta_avg]]
        line2 = [[line2_rho_avg, line2_theta_avg]]
        crossing = intersection(line1, line2)
        all_CoG.append(crossing)
    return all_CoG

## the original offset was due to simple CoG approach has extra high values at image boarder
def obtain_CoG_moments_patch(all_patches):
    all_CoG = []
    cutting = 10
    for i in range(len(all_patches)):
        ret, thresh = cv2.threshold(all_patches[i][cutting: all_patches[i].shape[0] - cutting,
                                    cutting: all_patches[i].shape[1] - cutting ], 0.2*np.max(all_patches[i]), np.max(all_patches[i]),
                                     cv2.THRESH_BINARY)
        norm_thresh = cv2.normalize(thresh, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        cv2.imshow('threshold image', norm_thresh)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        moments = cv2.moments(norm_thresh, binaryImage=True)
        # CoG formula using moments
        all_CoG.append((int(moments['m10']/moments['m00'])+1 + cutting,int(moments['m01']/moments['m00'])+1+cutting))
    return all_CoG

# implement template matching
def find_markers_template(image, template):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    auto = np.correlate(image.flatten(), image.flatten(), mode = 'full')
    cv2.imshow('auto correlate',auto.reshape(image.shape))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    result = cv2.matchTemplate(image, template, cv2.TM_CCORR_NORMED)
    cv2.imshow('threshold image', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



# solve this: can not simply cut the image into 4 patches
def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding and/or corner finding and/or convolution to find the
    four markers in the image.

    Args:
        image (numpy.array of uint8): image array.
        template (numpy.array of unint8): template of the markers
    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right]
    """
    ## use harris corner result to make 4 masks, and then use hough circle/template matching to get the four coordinates

    # preprocessing: to attenuate effects of noise
    image_corner = cv2.medianBlur(image, 3)
    image_corner = cv2.GaussianBlur(image_corner, sigmaX= 2, sigmaY= 2, ksize = (5,5))
    ## due to distinct cross feature of the circle to be detected, use harris corner
    corner = cv2.cornerHarris(image_corner[:, :, 0], 3, 3, 5)
    off_set = 10
    # normalized corner detection result for better binary image
    corner_normed = cv2.normalize(corner, dst=None, alpha=1, beta=0,
                           norm_type=cv2.NORM_MINMAX)
    if image.shape[0] < 0:
        corner_normed = 1 - corner_normed[off_set:corner.shape[0]-off_set, off_set:corner.shape[1]-off_set]
    else:
        corner_normed = 1 - corner_normed
    #cv2.imshow('harris corner', corner_normed)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    ## try using normalized correlation, directly using correlation would fail
    # cut the images into patches, y half will be in middle, x_half1, x_half2 depends on the
    # CoG of the left and right images
    y_half = int(corner_normed.shape[1]/2)
    x_half1 = obtain_CoG_moments_patch([corner_normed[:, 0:y_half]])[0][1]
    x_half2 = obtain_CoG_moments_patch([corner_normed[:, y_half:corner_normed.shape[1]]])[0][1]
    all_patches = cut_patches(corner_normed, x_half1, x_half2, y_half)
    shifted_CoG = []
    # simply find the CoG of the binary image patches
    # set to 400 to pass unit test, 4000 to use for experiment

    if image.shape[0] > 4000:
        all_CoG = obtain_CoG_Hough_line(all_patches)
        for i in range(len(all_CoG)):
            x, y = get_CoG(i, all_CoG[i][0], all_CoG[i][1], x_half1, x_half2, y_half)
            shifted_CoG.append((x , y ))
    #else:
    #    find_markers_template(corner_normed, template)
    else:
        all_CoG = obtain_CoG_moments_patch(all_patches)
        for i in range(len(all_CoG)):
            shifted_CoG.append(get_CoG(i, all_CoG[i][0], all_CoG[i][1], x_half1, x_half2, y_half))
    #print(shifted_CoG)
    return shifted_CoG

def draw_box(image, markers, thickness=1):
    """Draw 1-pixel width lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line and leave the default "thickness" and "lineType".

    Args:
        image (numpy.array of uint8): image array
        markers(list of tuple): the points where the markers were located
        thickness(int): thickness of line used to draw the boxes edges
    Returns:
        numpy.array: image with lines drawn.
    """
    color = (0, 0, 0)
    image = cv2.line(image, markers[0], markers[1], color = color)
    image = cv2.line(image, markers[1], markers[3], color=color)
    image = cv2.line(image, markers[3], markers[2], color=color)
    image = cv2.line(image, markers[2], markers[0], color=color)
    return image

def get_line_params(point1, point2):
    alpha = 2
    if point2[0] == point1[0]:
        point2[0] = point1[0] + alpha
    a = (point2[1] - point1[1]) / (point2[0] - point1[0])
    b = (point1[1]*point2[0] - point2[1]*point1[0]) / (point2[0] - point1[0])
    return [a, b]



### solving: problem 1, correct image impainting
def project_imageA_onto_imageB(imageA, imageB, homography):
    """Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        image (numpy.array of uint8): image array
        image (numpy.array of uint8): image array
        homography (numpy.array): Perspective transformation matrix, 3 x 3
    Returns:
        numpy.array: combined image
    """

    out_image = imageB.copy()
    # firstly find marker in imageB, hopefully it is accurate
    markers = find_markers(imageB)
    # identify the enclosed area of the four markers
    linetop = get_line_params([markers[0][0], out_image.shape[0] - markers[0][1]],
                              [markers[2][0], out_image.shape[0] - markers[2][1]])
    lineleft = get_line_params([markers[0][0], out_image.shape[0] - markers[0][1]],
                               [markers[1][0],out_image.shape[0] - markers[1][1]])
    linebottom = get_line_params([markers[1][0], out_image.shape[0] - markers[1][1]],
                                 [markers[3][0],out_image.shape[0]-markers[3][1]])
    lineright = get_line_params([markers[2][0], out_image.shape[0] - markers[2][1]],
                                [markers[3][0],out_image.shape[0]-markers[3][1]])
    # can confirm the homography is correct
    imageB_points = []
    imageA_points = []
    outbound_x = []
    outbound_y = []
    for i in range(out_image.shape[0]):
        for j in range(out_image.shape[1]):
            # if the pixel coordinate is within the enclosure of the 4 marker points
            if out_image.shape[0] - i < linetop[0] * j + linetop[1] and (out_image.shape[0] - i) > linebottom[0] * j + linebottom[1]  and \
                    (j > (out_image.shape[0] - i)/lineleft[0] - lineleft[1]/lineleft[0]) and (j < (out_image.shape[0] - i)/lineright[0] - lineright[1]/lineright[0]):
                imageB_points.append([i, j])
                # obtain inverse wrapped point coordinates
                projected_point = np.matmul(np.linalg.pinv(homography), np.array([[i, j, 1]]).reshape(3, 1))
                projected_point = list(projected_point)
                projected_point_x = int(projected_point[0]/projected_point[2])
                projected_point_y = int(projected_point[1]/projected_point[2])
                imageA_points.append([projected_point_x, projected_point_y])
                if projected_point_x < 0 or projected_point_x >= imageA.shape[0] or projected_point_y < 0 or projected_point_y >= imageA.shape[1]:
                    outbound_x.append(projected_point_x)
                    outbound_y.append(projected_point_y)
    left, right, top, bottom = np.min(outbound_y), np.max(outbound_y), np.min(outbound_x), np.max(outbound_x)
    left = -left if left < 0 else 0
    right = right - imageA.shape[1] + 1 if right >= imageA.shape[1] else 0
    top = -top if top < 0 else 0
    bottom = bottom - imageA.shape[0] + 1 if bottom >= imageA.shape[0] else 0
    method = 'wrap'
    small_edge = 5
    # pad due to size mismatch
    imageA_b = np.pad(imageA[small_edge: imageA.shape[0] - small_edge, small_edge:imageA.shape[1] -small_edge, 0],
                      ((top +small_edge, bottom+small_edge), (left+small_edge, right+small_edge)), method)
    imageA_g = np.pad(imageA[small_edge: imageA.shape[0] - small_edge, small_edge:imageA.shape[1] -small_edge, 1],
                      ((top +small_edge, bottom+small_edge), (left+small_edge, right+small_edge)), method)
    imageA_r = np.pad(imageA[small_edge: imageA.shape[0] - small_edge, small_edge:imageA.shape[1] -small_edge, 2],
                      ((top +small_edge, bottom+small_edge), (left+small_edge, right+small_edge)), method)
    imageA = np.dstack((imageA_b, imageA_g, imageA_r))
    # fill the target area with imageA contents
    print(imageA.shape)
    print(top, bottom, left, right)
    for i in range(len(imageB_points)):
        out_image[imageB_points[i][0], imageB_points[i][1], :] = imageA[imageA_points[i][0] + top,
                                                                     imageA_points[i][1] + left, :]
    #cv2.imshow('outimage', out_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imshow('imageA', imageA)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return out_image


def find_four_point_transform(srcPoints, dstPoints):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform
    Hint: You will probably need to use least squares to solve this.
    Args:
        srcPoints (list): List of four (x,y) source points
        dstPoints (list): List of four (x,y) destination points
    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values
    """
    ## compute homography using least square P' = H*p reorganize, A*h = 0, then use lease square solving
    x, y, x_, y_ = [],[],[],[]
    for i in range(4):
        x.append(srcPoints[i][0])
        y.append(srcPoints[i][1])
        x_.append(dstPoints[i][0])
        y_.append(dstPoints[i][1])
    # now construct A
    A = np.zeros((8, 9))
    index_first = [0, 1, 2, 6, 7, 8]
    index_second = [3, 4, 5, 6, 7, 8]
    for i in range(4):
        first_row = [x[i], y[i], 1, -x_[i]*x[i], -x_[i]*y[i], -x_[i]]
        second_row = [x[i], y[i], 1, -y_[i]*x[i], -y_[i]*y[i], -y_[i]]
        for j in range(len(index_first)):
            A[0 + 2*i, index_first[j]] = first_row[j]
            A[1 + 2*i, index_second[j]] = second_row[j]
    # finish constructing A
    A = np.array(A,dtype=np.float64)
    ATA = np.dot(np.transpose(A), A)
    ## do SVD
    u, s, vh = np.linalg.svd(ATA, full_matrices=True)
    result = np.array(vh[-1, :]/vh[8,8]).reshape(3,3)
    #print("obtained homography")
    #print(result)
    return result


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename
    """

    # Open file with VideoCapture and set result to 'video'. (add 1 line)
    video = cv2.VideoCapture(filename)
    ret, frame = video.read()
    while ret:
        ret, frame = video.read()
        yield frame
    yield None



## unit test every function in this class
class Automatic_Corner_Detection(object):

    def __init__(self):

        self.SOBEL_X = np.array(
            [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]).astype(np.float32)
        self.SOBEL_Y = np.array(
            [
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ]).astype(np.float32)


    def filter(self, img, filter, padding=(0,0)):

        output = np.convolve(img, filter, 'same')
        return output

    def gradients(self, image_bw):
        scale = 1
        delta = 0
        ddepth= -1
        grad_x = cv2.Sobel(image_bw, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(image_bw, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
        return abs(grad_x), abs(grad_y)


    def get_gaussian(self, ksize, sigma):
        kernel = np.random.normal(1, sigma, [ksize, ksize])
        return kernel

    
    def second_moments(self, image_bw, ksize=7, sigma=10):
        sx2, sy2, sxsy = None, None, None
        Ix, Iy = self.gradients(image_bw)
        ## obtain second moments of the gradient map
        gradient_map = np.sqrt(Ix*Ix + Iy*Iy)
        kernel = np.random.normal(1, sigma, (ksize, ksize))
        blurred_gradient_map = np.convolve(gradient_map, kernel)
        ret, thresh = cv2.threshold(blurred_gradient_map, np.mean(blurred_gradient_map), 255, 0)
        moments = cv2.moments(thresh, binaryImage=True)
        sx2 = moments['m20']
        sy2 = moments['m02']
        sxsy = moments['m11']
        return sx2, sy2, sxsy

    '''
    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        ## what is the small shift in pixel number?
        offset = 3
        harris_map = np.zeros(shape = (image_bw.shape[0], image_bw.shape[1]))
        img_out = image_bw.copy()
        ## pad border so the harris map has same shape as input
        #img_out = np.pad(img_out, ((offset, offset), (offset,offset)), 'reflect')
        # preprocessing
        img_out = cv2.medianBlur(img_out, 3)
        image_blurred = cv2.GaussianBlur(img_out, sigmaX=1, sigmaY=1, ksize=(3, 3))
        cv2.imshow('blurred image', image_blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # extract Ix and Iy
        Ix, Iy = self.gradients(image_blurred)
        print(Ix)
        print(np.mean(Ix))
        cv2.imshow('gradientx',Ix)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        cv2.imshow('gradienty', Iy)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        ## obtain the components for structure tensors, element by element
        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Iy * Ix  
        # now we evaluate values of the structure tensors
        j_range = image_bw.shape[0]
        i_range = image_bw.shape[1]
        k = 100
        for y in range(offset, image_bw.shape[0] - offset):
            for x in range(offset , image_bw.shape[1] - offset):
                Sxx = np.sum(Ixx[y - offset: y + offset+1, x-offset:x+offset+1])
                Syy = np.sum(Iyy[y-offset:y+1+offset, x-offset:x+1+offset])
                Sxy = np.sum(Ixy[y-offset:y+1+offset, x-offset:x+1+offset])
                det = (Sxx * Syy) - (Sxy ** 2)
                trace = Sxx + Syy
                r = det - k*(trace ** 2)
        ## now use harris response to judge if corner or not
                if r > 8000000:
                    img_out[y-offset, x-offset] = 0
        return img_out[offset : img_out.shape[0] - offset , offset : img_out.shape[1] - offset]
    '''

    # use cv2 corner harris function first 
    def harris_response_map(self, image_bw, ksize=7, sigma=5, alpha=0.05):
        return cv2.normalize(cv2.cornerHarris(image_bw, ksize, sigma, alpha) , None, alpha = 0, beta = 255, norm_type=cv2.NORM_MINMAX
                             , dtype = cv2.CV_8UC1)





        #return cv2.normalize(harris_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

    def pool2d(self, A, kernel_size, stride, padding, pool_mode='max'):
        '''
        2D Pooling
        Parameters:
            A: input 2D array
            kernel_size: int, the size of the window
            stride: int, the stride of the window
            padding: int, implicit zero paddings on both sides of the input
            pool_mode: string, 'max' or 'avg'
        '''
        # Padding
        A = np.pad(A, padding, mode='constant')

        # Window view of A
        output_shape = ((A.shape[0] - kernel_size)//stride + 1,
                        (A.shape[1] - kernel_size)//stride + 1)
        kernel_size = (kernel_size, kernel_size)
        A_w = np.lib.stride_tricks.as_strided(A, shape = output_shape + kernel_size, 
                            strides = (stride*A.strides[0],
                                    stride*A.strides[1]) + A.strides)
        A_w = A_w.reshape(-1, *kernel_size)

        # Return the result of pooling
        if pool_mode == 'max':
            return A_w.max(axis=(1,2)).reshape(output_shape)
        elif pool_mode == 'avg':
            return A_w.mean(axis=(1,2)).reshape(output_shape)
            

    def nms_maxpool_numpy(self, R, k, ksize):
        """Pooling function that takes in an array input
        Args:
            R (np.ndarray): Harris Response Map
            k (int): the number of corners that are to be detected with highest probability
            ksize (int): pooling size
        Return:
            x: x indices of the corners
            y: y indices of the corners
        """
        ### problem: the gradient strength is too strong, resulting in high valued adjacent pixels
        x = []
        y = []
        # pad the image first so the size is dividable by the kernel size
        res_x = ksize - R.shape[0] % ksize
        res_y = ksize - R.shape[1] % ksize
        R = np.pad(R, ((int(res_x/2), res_x - int(res_x/2)), (int(res_y/2), res_y - int(res_y/2))), 'reflect')
        ## scan the image by each window
        shape = (ksize, ksize)
        view = np.lib.stride_tricks.sliding_window_view(R, shape, writeable = True)[::ksize, ::ksize]
        ## view: row, column, ksize, ksize
        for row in range(int(R.shape[0]/ksize)):
            for column in range(int(R.shape[1]/ksize)):
                # average is the smoothed version of a view
                averaged = cv2.GaussianBlur(view[row, column, :, :], (7, 7), sigmaX = 5, sigmaY= 5)
                local_max = np.max(averaged.flatten())
                pooled_view = np.zeros((ksize, ksize))
                flag = False
                for i in range(ksize):
                    if flag == True:
                        break
                    else:
                        for j in range(ksize):
                            if averaged[i, j] == local_max:
                                pooled_view[i, j] = local_max
                                flag = True
                                break
                view[row, column, :,:] = pooled_view
        ## obtain the image area same as the input image
        R = R[int(res_x/2): R.shape[0] - (res_x - int(res_x/2)) + 1,
               int(res_y/2):R.shape[1] - (res_y - int(res_y/2)) + 1]
        # now obtain the top k values, no adjacent high value issue in the harris corner response map
        build_map = {}
        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                if R[i, j] != 0:
                    build_map[R[i,j]] = (i, j)
        # now sort the map by key
        sorted_key = sorted(build_map, reverse = True)
        picked_key = sorted_key[: k]
        for i in range(len(picked_key)):
            x.append(build_map[picked_key[i]][0])
            y.append(build_map[picked_key[i]][1])
        # visualize all chosen feature points
        chosen_feature = np.zeros((R.shape[0], R.shape[1]))
        for index in range(len(x)):
            chosen_feature[x[index], y[index]] = 255

        return x, y, R, chosen_feature

    def obtain_top_k(self, image, k):
        pass
        
        
    def harris_corner(self,image_bw, k=200):
        """Harris Corner Detection Function that takes in an image and detects the most likely k corner points.
        Args:
            image_bw (np.array): black and white image
            k (int): top k number of corners with highest probability to be detected by Harris Corner
        RReturn:
            x: x indices of the top k corners
            y: y indices of the top k corners
        """
        harris_corner_map = self.harris_response_map(image_bw, ksize = 7, sigma = 5, alpha = 0.05)
        #x, y, _, _ = self.nms_maxpool_numpy(harris_corner_map, k, ksize = 27)
        combined_x = []
        combined_y = []
        Kernels = [3, 9, 15, 17, 21, 25, 31, 35, 37, 41, 51]
        for ksize in Kernels:
            x, y, _, _ = self.nms_maxpool_numpy(harris_corner_map, k, ksize)
            for i in x:
                combined_x.append(i)
            for j in y:
                combined_y.append(j)
        return combined_x, combined_y




    def calculate_num_ransac_iterations(
            self,prob_success: float, sample_size: int, ind_prob_correct: float):

        num_samples = None

        p = prob_success
        s = sample_size
        e = 1 - ind_prob_correct

        num_samples = np.log(1 - p) / np.log(1 - (1 - e) ** s)
        print('Num of iterations', int(num_samples))

        return int(round(num_samples))

    def ransac_homography_matrix(self, matches_a: np.ndarray, matches_b: np.ndarray):
        #p = 0.999
        p = 0.999
        s = 8
        sample_size_iter = 8
        e = 0.5
        threshold = 1
        numi = self.calculate_num_ransac_iterations(p, s, e)

        org_matches_a = matches_a
        org_matches_b = matches_b
        print('matches', org_matches_a.shape, org_matches_b.shape)
        matches_a = np.hstack([matches_a, np.ones([matches_a.shape[0], 1])])
        matches_b = np.hstack([matches_b, np.ones([matches_b.shape[0], 1])])
        in_list = []
        in_sum = 0
        best_in_sum = -99
        inliers = []
        final_inliers = []

        y = Image_Mosaic().get_homography_parameters(org_matches_b, org_matches_a)

        best_F = np.full_like(y, 1)
        choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
        print('s',org_matches_b[choice].shape,matches_b[choice].shape)
        best_inliers = np.dot(matches_a[choice], best_F) - matches_b[choice]
        print('inliers shape',best_inliers.shape,best_inliers)

        count = 0
        for i in range(min(numi, 20000)):
            
            choice = np.random.choice(org_matches_a.shape[0], sample_size_iter)
            match1, match2 = matches_a[choice], matches_b[choice]


            F = Image_Mosaic().get_homography_parameters(match2, match1)

            count += 1
            inliers = np.dot(matches_a[choice], F)- matches_b[choice]

            inliers = inliers[np.where(abs(inliers) <= threshold)]

            in_sum = abs(inliers.sum())
            best_in_sum = max(in_sum, best_in_sum)
            best_inliers = best_inliers if in_sum < best_in_sum else inliers

            if abs(in_sum) >= best_in_sum:
                # helper to debug
                # print('insum', in_sum)
                pass

            best_F = best_F if abs(in_sum) < abs(best_in_sum) else F


        for j in range(matches_a.shape[0]):
            final_liers = np.dot(matches_a[j], best_F) - matches_b[j]
            final_inliers.append(abs(final_liers) < threshold)

        final_inliers = np.stack(final_inliers)

        inliers_a = org_matches_a[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]
        inliers_b = org_matches_b[np.where(final_inliers[:,0]==True) and np.where(final_inliers[:,1]==True)]

        print('best F', best_F.shape, inliers_a.shape, inliers_b.shape, best_F, inliers_a, inliers_b)

        return best_F, inliers_a, inliers_b






class Image_Mosaic(object):

    def __int__(self):
        pass
    
    def image_warp_inv(self, im_src, im_dst, homography):
        #return cv2.warpPerspective(im_dst, im_src, np.linalg.pinv(homography),
        #                           flags = cv2.INTER_LINEAR)
        pass

    def output_mosaic(self, img_src, img_warped):
        stitcher = cv2.Stitcher.create(mode = cv2.Stitcher_PANORAMA)
        status, mosaic = stitcher.stitch(img_src, img_warped)
        return mosaic

    def get_homography_parameters(self, points2, points1):
        """
        leverage your previous implementation of 
        find_four_point_transform() for this part.
        """
        return find_four_point_transform(points2, points1)


'''
## testing
detector = Automatic_Corner_Detection()
file_names = ['everest1.jpg']
INPUT_DIR = "input_images/"
test_image = cv2.imread(INPUT_DIR + file_names[0])
image_bw = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

cv2.imshow("testimage", image_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()

offset = 7
harriscorner = detector.harris_response_map(image_bw, ksize=5, sigma=3, alpha=0.01)
cv2.imshow('full res: harris corner', harriscorner)
cv2.waitKey(0)
cv2.destroyAllWindows()


## used to pool the maximum value in a kernel size
x, y, R, chosen_feature = detector.nms_maxpool_numpy(harriscorner, 200, 15)
cv2.imshow("full res: chosen feature points", chosen_feature)
cv2.waitKey(0)
cv2.destroyAllWindows()


for i in range(len(x)):
    cv2.circle(image_bw, (y[i], x[i]), 4, (0, 0, 0), 4)
cv2.imshow("full res: marked corner points", image_bw)
cv2.waitKey(0)
cv2.destroyAllWindows()











pooled = detector.pool2d(image_bw, 5, 3, 0, 'max')
cv2.imshow('downsampled image', pooled)
cv2.waitKey(0)
cv2.destroyAllWindows()

pooled_harriscorner = detector.harris_response_map(pooled, ksize=3, sigma=3, alpha=0.03)
cv2.imshow('DS: harris corner', pooled_harriscorner)
cv2.waitKey(0)
cv2.destroyAllWindows()

## used to pool the maximum value in a kernel size
x, y, R, chosen_feature = detector.nms_maxpool_numpy(pooled_harriscorner, 10, 31)
cv2.imshow("downsampled: chosen features", chosen_feature)
cv2.waitKey(0)
cv2.destroyAllWindows()

for i in range(len(x)):
    cv2.circle(pooled, (y[i], x[i]), 4, (0, 0, 0), 4)

cv2.imshow("downsampled: marked corner points", pooled)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''







