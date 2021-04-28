import numpy as np
import math



class Filtering:
    image = None
    filter_name = None
    filter_size = None
    alpha_d = None
    order = None

    def __init__(self, image, filter_name, filter_size, alpha_d=2, order = 1.5 ):
        """initializes the variables of spatial filtering on an input image
        takes as input:
        image: the noisy input image
        filter_name: the name of the mask to use
        filter_size: integer value of the size of the mask
        alpha_d: parameter of the alpha trimmed mean filter
        order: parameter of the order for contra harmonic"""

        self.image = image
        if filter_name == 'median':
            self.filter = self.get_median_filter
        elif filter_name == 'min':
            self.filter = self.get_min_filter
        elif filter_name == 'max':
            self.filter = self.get_max_filter
        elif filter_name == 'alpha_trimmed':
            self.filter = self.get_alpha_filter
        elif filter_name == 'arithmetic_mean':
            self.filter = self.get_arithmetic_mean_filter
        elif filter_name == 'geometric_mean':
            self.filter = self.get_geo_mean_filter
        elif filter_name == 'contra_harmonic':
            self.filter = self.get_contra_harmonic_filter
        self.filter_name = filter_name
        self.filter_size = filter_size
        self.alpha_d = alpha_d
        self.order = order

    def get_median_filter(self, kernel): # done
        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        returns the median value in the current kernel
        """
        sortk = sorted(kernel)
        medval = sortk[int(len(kernel)/2)]
        return medval

    def get_min_filter(self, kernel): # done
        """Computes the minimum filter
        takes as input:
        ikernel: a list/array of intensity values
        returns the minimum value in the current kernel"""
        if kernel[int(len(kernel)/2)] == 255:
            minval = min(kernel)
            return minval
        else:
            return kernel[int(len(kernel)/2)]

    def get_max_filter(self, kernel): # done
        """Computes the maximum filter
        takes as input:
        kernel: a list/array of intensity values
        returns the maximum value in the current kernel"""
        if kernel[int(len(kernel)/2)] == 0:
            maxval = max(kernel)
            return maxval
        else:
            return kernel[int(len(kernel)/2)]

    def get_alpha_filter(self, kernel, alpha_d):
        """Computes the median filter
        takes as input:
        kernel: a list/array of intensity values
        alpha_d: clip off parameter for the alpha trimmed filter
        returns the alpha trimmed mean value in the current kernel"""
        sortk = sorted(kernel)
        d = int(alpha_d/2)
        del sortk[0:d]
        del sortk[len(sortk) - d:len(sortk)]
        sum = 0
        for x in range(len(sortk)):
            sum += sortk[x]
        alpha = 1/(len(kernel)-(2*d)) * sum

        return alpha

    def get_arithmetic_mean_filter(self, kernel):
        """Computes the arithmetic mean filter
        takes as input:
        kernel: a list/array of intensity values
        returns the arithmetic mean value in the current kernel"""
        sum = 0
        for x in range(len(kernel)):
            sum += kernel[x]
        arthmean = (1/len(kernel)) * sum

        return arthmean

    def get_geo_mean_filter(self, kernel): # done
        """Computes the geometric mean filter
                    takes as input:
                        kernel: a list/array of intensity values
                        returns the geometric mean value in the current kernel"""
        prod = 1
        for x in range(len(kernel)):
            if kernel[x] != 0:
                prod = prod * kernel[x]
        geomean = pow(prod, (1/len(kernel)))

        return geomean

    def get_contra_harmonic_filter(self, kernel, order): # done
        """Computes the harmonic filter
                        takes as input:
        kernel: a list/array of intensity values
        order: order paramter for the
        returns the harmonic mean value in the current kernel"""
        sumt = 0
        sumb = 0
        oval = -1 * order
        if order < 0:
            for x in range(len(kernel)):
                sumt += pow(kernel[x], -oval - 1)
                sumb += pow(kernel[x], -oval)
            contra = sumb/sumt
        else:
            for x in range(len(kernel)):
                sumt += pow(kernel[x], order + 1)
                sumb += pow(kernel[x], order)
            contra = sumt/sumb

        return contra

    def filtering(self):
        """performs filtering on an image containing gaussian or salt & pepper noise
        returns the denoised image
        ----------------------------------------------------------
        Note: Here when we perform filtering we are not doing convolution.
        For every pixel in the image, we select a neighborhood of values defined by the kernal and apply a mathematical
        operation for all the elements with in the kernel. For example, mean, median and etc.

        Steps:
        1. add the necesssary zero padding to the noisy image, that way we have sufficient values to perform the operati
        ons on the pixels at the image corners. The number of rows and columns of zero padding is defined by the kernel size
        2. Iterate through the image and every pixel (i,j) gather the neighbors defined by the kernel into a list (or any data structure)
        3. Pass these values to one of the filters that will compute the necessary mathematical operations (mean, median, etc.)
        4. Save the results at (i,j) in the ouput image.
        5. return the output image
        """
        "///////////////////////////////// 1"
        R, C = self.image.shape
        sizenum = self.filter_size
        sizenum = int(.5 * sizenum - .5)
        pad_image = np.zeros((R + (2*sizenum), C + (2*sizenum)))
        pad_newimage = np.zeros((R + (2*sizenum), C + (2*sizenum)))

        iimag = np.zeros((R, C))
        Rp , Cp = pad_image.shape
        #print(self.image.shape, " ", pad_image.shape, " ", sizenum)
        kernel = []
        for x in range(R):
            for y in range(C):
                pad_image[x+sizenum][y+sizenum] = self.image[x][y]
        "///////////////////////////////// 2"
        for x in range(sizenum+1,Rp - sizenum):
            for y in range(sizenum+1,Cp - sizenum):
                kernel.clear()
                #print(x, y)
                for xk in range(-sizenum,sizenum+1):
                    for yk in range(-sizenum,sizenum+1):
                        kernel.append(pad_image[x+xk][y+yk])
                """ used when i thought size was fixed
                kernel.append(pad_image[x-1][y-1])
                kernel.append(pad_image[x-1][y])
                kernel.append(pad_image[x-1][y+1])
                kernel.append(pad_image[x][y-1])
                kernel.append(pad_image[x][y])
                kernel.append(pad_image[x][y+1])
                kernel.append(pad_image[x+1][y-1])
                kernel.append(pad_image[x+1][y])
                kernel.append(pad_image[x+1][y+1])
                """
                # trail ############################################
                "///////////////////////////////// 3"
                if self.filter_name == 'alpha_trimmed':
                    Fvalue = self.filter(kernel, self.alpha_d)
                elif self.filter_name == 'contra_harmonic':
                    Fvalue = self.filter(kernel, self.order)
                else:
                    Fvalue = self.filter(kernel)
                "///////////////////////////////// 4"
                pad_newimage[x][y] = Fvalue
        "///////////////////////////////// 5"

        for x1 in range(R):
            for y1 in range(C):
                iimag[x1][y1] = pad_newimage[x1+sizenum][y1+sizenum]
        return iimag

