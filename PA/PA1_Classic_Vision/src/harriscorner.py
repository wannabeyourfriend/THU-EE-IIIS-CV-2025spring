import numpy as np
from utils import read_img, draw_corner
from convolve import convolve, Sobel_filter_x, Sobel_filter_y, padding


def corner_response_function(input_img, window_size, alpha, threshold):
    """
        The function you need to implement for Q3.
        Inputs:
            input_img: array(float)
            window_size: int
            alpha: float
            threshold: float
        Outputs:
            corner_list: array
    """

    # please solve the corner_response_function of each window,
    # and keep windows with theta > threshold.
    # you can use several functions from HM1_Convolve to get 
    # I_xx, I_yy, I_xy as well as the convolution result.
    # for detials of corner_response_function, please refer to the slides.
    
    # 直接使用Sobel滤波器计算梯度，内部已经包含了padding
    I_x = Sobel_filter_x(input_img)
    I_y = Sobel_filter_y(input_img)
    
    # Compute products of derivatives
    I_xx = I_x * I_x
    I_yy = I_y * I_y
    I_xy = I_x * I_y
    
    # 对导数积进行填充，以便使用窗口卷积
    pad_size = window_size // 2
    I_xx_padded = padding(I_xx, pad_size, "replicatePadding")
    I_yy_padded = padding(I_yy, pad_size, "replicatePadding")
    I_xy_padded = padding(I_xy, pad_size, "replicatePadding")
    
    # Generate Gaussian window for better weighting
    def gaussian_window(size, sigma=1.0):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g/g.sum()
    
    # Use Gaussian window instead of uniform window for better results
    window = gaussian_window(window_size, sigma=1.0)
    
    # Apply window to the derivative products using convolution
    S_xx = convolve(I_xx_padded, window)
    S_yy = convolve(I_yy_padded, window)
    S_xy = convolve(I_xy_padded, window)
    
    # Calculate the Harris corner response function R
    # R = det(M) - alpha * (trace(M))^2
    # where M is the structure tensor [[S_xx, S_xy], [S_xy, S_yy]]
    det_M = S_xx * S_yy - S_xy * S_xy  # determinant
    trace_M = S_xx + S_yy  # trace
    R = det_M - alpha * (trace_M ** 2)
    
    # 对R图像进行填充，便于后续局部最大值检测
    R_padded = padding(R, 1, "replicatePadding")
    
    # Normalize R values to [0,1] range for easier thresholding
    R_min = np.min(R)
    R_max = np.max(R)
    if R_max > R_min:
        R_normalized = (R - R_min) / (R_max - R_min)
    else:
        R_normalized = R
    
    # Find corner points where R is above threshold
    corner_list = []
    height, width = R.shape
    
    for i in range(height):
        for j in range(width):
            if R_normalized[i, j] > threshold / 100.0:
                # 使用填充后的R来检查局部最大值，避免边界问题
                # 注意索引需要+1，因为R_padded比R多出一圈填充
                neighborhood = R_padded[i:i+3, j:j+3]
                if R_padded[i+1, j+1] == np.max(neighborhood):
                    corner_list.append([i, j, R[i, j]])
    
    # 如果检测到的角点太少，使用自适应阈值
    if len(corner_list) < 10:
        print(f"Only {len(corner_list)} corners detected. Using adaptive threshold.")
        # 获取响应值排名前N的点
        indices = np.argsort(R.flatten())[-100:]  # 获取前100个点
        corner_list = []
        
        for idx in indices:
            i, j = idx // width, idx % width
            # 使用填充后的R进行局部最大值检测
            if i > 0 and j > 0 and i < height-1 and j < width-1:
                neighborhood = R_padded[i:i+3, j:j+3]
                if R_padded[i+1, j+1] == np.max(neighborhood):
                    corner_list.append([i, j, R[i, j]])
    
    return np.array(corner_list) # array, each row contains information about one corner, namely (index of row, index of col, theta)


if __name__=="__main__":
    # Load the input images
    input_img = read_img("hand_writting.png")/255.

    # You can adjust the parameters to fit your own implementation 
    window_size = 7
    alpha = 0.033
    threshold = 35  # Lower threshold value to detect more corners

    corner_list = corner_response_function(input_img, window_size, alpha, threshold)

    # Check if corners were detected
    if len(corner_list) == 0:
        print("No corners detected. Please try adjusting the parameters.")
        exit()

    # NMS
    corner_list_sorted = sorted(corner_list, key=lambda x: x[2], reverse=True)
    NML_selected = [] 
    NML_selected.append(corner_list_sorted[0][:-1])
    dis = 10
    for i in corner_list_sorted:
        for j in NML_selected:
            if(abs(i[0] - j[0]) <= dis and abs(i[1] - j[1]) <= dis):
                break
        else:
            NML_selected.append(i[:-1])

    # save results
    draw_corner("hand_writting.png", "result/HarrisCorner/harriscorner.png", NML_selected)