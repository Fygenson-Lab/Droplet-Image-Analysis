import os
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import cv2 as cv
from joblib import Parallel, delayed

plt.style.use('sarah_plt_style.mplstyle')

def find_pixels_in_circle(cir, x_dim, y_dim):
    #returns a list of points inside a circle with a given center point and radius
    #does not include any points which are off the edge of the image
    points = []
    for y in range(-1 *  cir.r, +cir.r+1):
        x = (cir.r**2 - y**2)**0.5

        x1 = int(-x + cir.cx)
        x2 = int(+x + cir.cx)
    
        y = y + cir.cy
        for x in range(x1, x2):
            if x < x_dim and y < y_dim: #don't keep a point if it falls outside the image size
                points.append([x,y])
    return points 

def find_average_brightness(xdim, ydim, cir, image_to_get_values):
    points = find_pixels_in_circle(cir, xdim, ydim)
    values = [image_to_get_values[point[1], point[0]] for point in points]
    return np.mean(values) 

find_average_brightness = np.vectorize(find_average_brightness, excluded=[0,1,3])

def find_best_r(xdim, ydim, og_cir, image, r_min, r_max):
    if r_min == 'min':
        r_min = (og_cir.r - 1) * -1
    r_array = range(og_cir.r + r_min,og_cir.r + r_max) #range of radi to consider 
    cir_array = [circle([og_cir.cx, og_cir.cy, i]) for i in r_array]
    brightness = find_average_brightness(xdim, ydim, cir_array, image)

    dir = np.gradient(brightness, r_array) #get the derivative of how the average brightness changes in the circle with changeling radius
    try:
        best_r = r_array[list(dir).index(np.min(dir))] #select the radius witch gives the minimum derivative
    except:
        best_r = og_cir.r

    return circle([og_cir.cx, og_cir.cy, best_r])

find_best_r = np.vectorize(find_best_r, excluded=[0,1,3])

def custom_maximize(fun, guess, image, max_fun_calls = None, bounds = None):
    best_par = guess
    best_val = fun(best_par, image)
    function_calls = 1
    improved = True #will stop the loop when takeing a step does not improve the result
    stop = False #will stop the loop when reach max number of function calls
    tested_parameters = []
    
    while improved and not stop:
        improved = False
        for step in [best_par - 1, best_par + 1]: #try taking either a step forward or a step backwards in one each dimension
            if step >= bounds[0] and step <= bounds[1]: #if the proposed step is inside the given bounds
                test_par = best_par
                test_par = step
                if test_par not in tested_parameters: #if we haven't previously tested this set of parameters
                    tested_parameters.append(test_par)
                    test_val = fun(test_par, image)
                    function_calls += 1
                    if test_val > best_val:
                        best_val = test_val
                        best_par = test_par
                        improved = True
        if max_fun_calls is not None and function_calls >= max_fun_calls:
            stop = True #stop the for loop if we reach the max number of function calls
            break
    print('Total function calls: {}'.format(function_calls))

    return best_par

class circle:
    def __init__(self, par):
        self.cx, self.cy, self.r = par

class droplet_image:
    def binerize(self, cutoff):
        #convert to an 8 bit image (open cv requires 8, not 16 bit images)
        img_8bit = (self.image/256).astype('uint8')
        #binerize the 8 bit image
        _, bin_image = cv.threshold(img_8bit, cutoff, 1, cv.THRESH_BINARY)
        return bin_image

    def find_raw_circles(self, image_to_search):
        #find circles using a Hough transform
        circles = cv.HoughCircles(image_to_search, cv.HOUGH_GRADIENT, dp =1, minDist = 50, param1=1, param2=1, maxRadius = 50)
        try:
            circles = np.uint16(circles)
        except:
            return []
        return np.array([circle(i) for i in circles[0,:]])
    
    def get_circles(self):
        raw_circles = self.find_raw_circles(self.bin_image)
        try:
            filtered_circles = raw_circles[np.where(find_average_brightness(self.xdim, self.ydim, raw_circles, self.bin_image) > 0.7)[0]]
            gas_circles = find_best_r(self.xdim, self.ydim, filtered_circles, self.image, -5, 35)
            condensate_circles = find_best_r(self.xdim, self.ydim, filtered_circles, self.image, 'min', -5)
        except:
            return [], []

        return list(gas_circles), list(condensate_circles)

    def __init__(self, array, cutoff):
        self.image = array
        self.ydim, self.xdim = self.image.shape
        self.bin_image = self.binerize(cutoff)
        self.gas_circles, self.condensate_circles = self.get_circles()
        self.gas_radi, self.condensate_radi = [cir.r for cir in self.gas_circles], [cir.r for cir in self.condensate_circles]
        self.num_gas = len(self.gas_radi)

###get images###
desired__folder = input('Folder with images to analyze: ') #ask user for location of the images to analyze (must be in the Images folder)
os.chdir('{}\Images\\{}'.format(os.getcwd(), desired__folder)) #change the current working dircetory to be the folder of images we want to analyze
image_dir = os.getcwd()

image_files = sorted(os.listdir(image_dir)) #get the image files
images = [plt.imread('{}\{}'.format(image_dir, image_file)) for image_file in image_files] #read the images from the files
num_images = len(images)

initial_guess = 3
cutoff_bounds = (0,10)
best_fit_array =  Parallel(n_jobs = -1, verbose = 10)(delayed(custom_maximize)(lambda par, image : droplet_image(image, par).num_gas, initial_guess, images[i], bounds = cutoff_bounds) for i in range(num_images))
best_droplet_array = Parallel(n_jobs= -1, verbose= 10)(delayed(droplet_image)(images[i], best_fit_array[i]) for i in range(num_images))

fig, axs = plt.subplots(1, 3, figsize = (40,20))
for i in range(num_images):
    axs[i].imshow(best_droplet_array[i].image, cmap = 'gray')
    for cir in best_droplet_array[i].gas_circles:
        axs[i].add_patch(Circle((cir.cx, cir.cy), cir.r, color = 'g', fill = False, lw = 2))
        axs[i].add_patch(Circle((cir.cx, cir.cy), 2, color = 'b'))
    for cir in best_droplet_array[i].condensate_circles:
        axs[i].add_patch(Circle((cir.cx, cir.cy), cir.r, color = 'r', fill = False, lw = 2))
plt.show()