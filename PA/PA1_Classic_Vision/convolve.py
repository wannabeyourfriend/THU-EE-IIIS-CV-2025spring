import numpy as np
from utils import read_img, write_img

def padding(img, padding_size, type):
    """
        The function you need to implement for Q1 a).
        Inputs:
            img: array(float)
            padding_size: int
            type: str, zeroPadding/replicatePadding
        Outputs:
            padding_img: array(float)
    """
    h, w = img.shape
    padding_img = np.zeros((h + 2 * padding_size, w + 2 * padding_size))
    if type=="zeroPadding":
        # Used in PyTorch style is  nn.ZeroPad2d(padding=(left, right, top, bottom))
        padding_img[padding_size:h+padding_size, padding_size:w+padding_size] = img
        return padding_img
    elif type=="replicatePadding":
        # Used in PyTorch style is nn.ReplicationPad2d(padding=(left, right, top, bottom))
        padding_img[padding_size:h+padding_size, padding_size:w+padding_size] = img
        padding_img[:padding_size, padding_size:w+padding_size] = img[0:1, :]  # Top
        padding_img[h+padding_size:, padding_size:w+padding_size] = img[-1:, :]  # Bottom
        padding_img[padding_size:h+padding_size, :padding_size] = img[:, 0:1]  # Left
        padding_img[padding_size:h+padding_size, w+padding_size:] = img[:, -1:]  # Right
        padding_img[:padding_size, :padding_size] = img[0, 0]  # Top-left
        padding_img[:padding_size, w+padding_size:] = img[0, -1]  # Top-right
        padding_img[h+padding_size:, :padding_size] = img[-1, 0]  # Bottom-left
        padding_img[h+padding_size:, w+padding_size:] = img[-1, -1]  #
        return padding_img
    # Else padding methods can be ConstantPad2d and ReflectionPad2d 


def convol_with_Toeplitz_matrix(img, kernel):
    """
        The function you need to implement for Q1 b).
        Inputs:
            img: array(float) 6*6
            kernel: array(float) 3*3
        Outputs:
            output: array(float)
    """
    # Get dimensions
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    # Pad the input image with zeros
    padding_size = (k_h - 1) // 2
    padded_img = padding(img, padding_size, "zeroPadding")
    
    # Flatten and flip the kernel
    kernel_flipped = np.flip(kernel.flatten())
    
    # Create indices for the Toeplitz matrix
    row_indices = np.arange(h*w)[:, None]
    
    # Create base column indices for each row
    base_col = (row_indices // w) * (w + k_w - 1) + (row_indices % w)
    
    # Create offset indices for the kernel
    kernel_offsets = np.arange(k_h)[:, None] * (w + k_w - 1) + np.arange(k_w)
    kernel_offsets = kernel_offsets.flatten()
    
    # Combine base columns with kernel offsets
    col_indices = base_col + kernel_offsets[None, :]
    
    # Create Toeplitz matrix using advanced indexing
    toeplitz_matrix = np.zeros((h*w, (h+k_h-1)*(w+k_w-1)))
    toeplitz_matrix[row_indices, col_indices] = kernel_flipped[None, :]
    
    # Perform convolution
    result_flat = np.dot(toeplitz_matrix, padded_img.flatten())
    
    # Reshape result back to 2D
    output = result_flat.reshape(h, w)
    
    return output


def convolve(img, kernel):
    """
        The function you need to implement for Q1 c).
        Inputs:
            img: array(float)
            kernel: array(float)
        Outputs:
            output: array(float)
    """
    h, w = img.shape
    k_h, k_w = kernel.shape
    
    out_h, out_w = h - k_h + 1, w - k_w + 1
    
    i, j = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing='ij')
    i = i.reshape(-1, 1)
    j = j.reshape(-1, 1)
    
    di, dj = np.meshgrid(np.arange(k_h), np.arange(k_w), indexing='ij')
    di = di.reshape(1, -1)
    dj = dj.reshape(1, -1)
    
    i_pos = i + di
    j_pos = j + dj
    
    windows = img[i_pos, j_pos]
    
    kernel_flipped = np.flip(np.flip(kernel, 0), 1).reshape(-1)
    output = np.dot(windows, kernel_flipped).reshape(out_h, out_w)
    
    return output


def Gaussian_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    gaussian_kernel = np.array([[1/16,1/8,1/16],[1/8,1/4,1/8],[1/16,1/8,1/16]])
    output = convolve(padding_img, gaussian_kernel)
    return np.clip(output, 0, 1)

def Sobel_filter_x(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    output = convolve(padding_img, sobel_kernel_x)
    return np.clip(output, 0, 1)

def Sobel_filter_y(img):
    padding_img = padding(img, 1, "replicatePadding")
    sobel_kernel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    output = convolve(padding_img, sobel_kernel_y)
    return np.clip(output, 0, 1)

def Sharpening_filter(img):
    padding_img = padding(img, 1, "replicatePadding")
    sharpening_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    output = convolve(padding_img, sharpening_kernel)
    return np.clip(output, 0, 1)

if __name__=="__main__":

    np.random.seed(111)
    input_array=np.random.rand(6,6)
    input_kernel=np.random.rand(3,3)


    # task1: padding
    zero_pad =  padding(input_array,1,"zeroPadding")
    np.savetxt("result/Convolve/Convolve_zero_pad.txt",zero_pad)

    replicate_pad = padding(input_array,1,"replicatePadding")
    np.savetxt("result/Convolve/Convolve_replicate_pad.txt",replicate_pad)


    #task 2: convolution with Toeplitz matrix
    result_1 = convol_with_Toeplitz_matrix(input_array, input_kernel)
    np.savetxt("result/Convolve/Convolve_result_1.txt", result_1)

    #task 3: convolution with sliding-window
    result_2 = convolve(input_array, input_kernel)
    np.savetxt("result/Convolve/Convolve_result_2.txt", result_2)

    #task 4/5: Gaussian filter and Sobel filter
    input_img = read_img("lenna.png")/255

    img_gadient_x = Sobel_filter_x(input_img)
    img_gadient_y = Sobel_filter_y(input_img)
    img_blur = Gaussian_filter(input_img)
    img_sharpening = Sharpening_filter(input_img)

    write_img("result/Convolve/Convolve_img_gadient_x.png", img_gadient_x*255)
    write_img("result/Convolve/Convolve_img_gadient_y.png", img_gadient_y*255)
    write_img("result/Convolve/Convolve_img_blur.png", img_blur*255)
    write_img("result/Convolve/Convolve_img_sharpening.png", img_sharpening*255)




    