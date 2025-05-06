## **2 Low Level Vision**

### 1.2 Edge/corner detection

> **Edge Definition:** An edge is defined as a region in the image where there is **a “significant” change in the pixel intensity values (or having high contrast)** along one direction in the image, and almost no changes in the pixel intensity values (or low contrast) along its orthogonal direction.

So what can cause am edge:

**Depth discontinuity, Surface color discontinuity, Surface orientation discontinuity, Illumination discontinuity**

**Criteria for Optimal Edge Detection**

- High precision: make sure all detected edges are true edges (via minimizing FP).

- High recall: make sure all edges can be detected (via minimizing FN).

- Good localization: minimize the distance between the detected edge and the ground truth
  edge

- Single response constraint: minimize redundant responses

  ![image-20250226172606980](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250226172606980.png)



见下面两种著名的算法Canny Edge detecting和Harris Detector

## Canny Edge detecting

### Step 1 Gaussion Denoising

- Smooth the image using Gaussian kernel to suppress high frequency noise and avoid noise from being misclassified as an edge.
- It improves the reliability of subsequent gradient calculation and reduces the pseudo-edge points caused by noise.

### Step 2 Gradient filtering

- **Gradient calculation**: Calculate the gradient in $G_x$ and $Gy$ direction by **Sobel** operator.
  - Sobel operator:

​	![img](https://i-blog.csdnimg.cn/blog_migrate/5219827a9fd4a6c18667773602203b71.png)

### Step 3 NMS

- Perform non-maximal value suppression, comparing neighboring pixel magnitudes along the gradient direction and retaining only local maxima
- **Line point filtering**: suppress the non-extremely large value points and get the candidate edge line points with single pixel width.

Bilinear interpolation is needed in NMS.



### Step 4 Line fitting

#### Least Squares Method

Given a set of data points $(x_i,y_i)$, our goal is to find a line $y=mx+b$ such that the sum of the squares of the perpendicular distances from all points to the line is minimized.

1. Error function definition:

$$
E = \sum_{i=1}^n (y_i - mx_i - b)^2
$$

2. To minimize the error E, take the partial derivatives of the parameters m and b and make them zero:

$$
   \frac{\partial E}{\partial m} = -2\sum_{i=1}^n x_i(y_i - mx_i - b) = 0\\\\
   \frac{\partial E}{\partial b} = -2\sum_{i=1}^n (y_i - mx_i - b) = 0
$$


The problem can be rewritten in matrix form:
$$
E = (Y - XB)^T(Y - XB)
$$
Derive E with respect to B and make it zero:
$$
\frac{\partial E}{\partial B} = -2X^TY + 2X^TXB = 0
$$

Obtain the Normal equation:
$$
X^TXB = X^TY
$$

The final solution is:
$$
B = (X^TX)^{-1}X^TY
$$

##### Outliers

Very poor robustness to outliers, not ideal for line fitting

#### SVD method



**RANSAC**-methods

In the process of line-fitting, we often encounter the need to exclude the effect of outliers on the model.



## Harris Detector

## Idea

Harris corner detector

## Window Function

The choices for the window function are

### Corner Response Function

We've got $$\mathbb{E}_{(x,y)}(u,v) = \lambda_1 u^2 + \lambda_2v^2$$

If one of $\lambda_i, i = 1,2$ is particularly large or small, it means that the **energy function** will vary particularly dramatically in one direction or another.



![image-20250312153013966](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250312153013966.png)



To quickly estimate whether a feature point is an edge or a corner, we introduce a fast approximation:
$$
\theta = \frac{1}{2}(\lambda_1\lambda_2 - 2\alpha(\lambda_1 + \lambda_2)^2) + \frac{1}{2}(\lambda_1 \lambda_2 - 2t)
$$
The origin of this equation: for a CORNER point such as the green area of Figure Lake, we have the following equation holding true
$$
\frac{1}{k} < \frac{\lambda_1}{\lambda_2} < k \Leftrightarrow 
$$




Finally, summarize the overall flow of detect:



为了快速估计一个特征点是否是edge或corner，我们引入一个fast approximation：
$$
\theta = \frac{1}{2}(\lambda_1\lambda_2 - 2\alpha(\lambda_1 + \lambda_2)^2)+ \frac{1}{2}(\lambda_1 \lambda_2 - 2t)
$$
这个式子的由来：对于一个corner点如图湖绿色区域，我们有如下的式子成立
$$
\frac{1}{k} < \frac{\lambda_1}{\lambda_2} < k \Leftrightarrow 
$$




最后总结detect的整体流程：

![image-20250312154647217](C:\Users\35551\AppData\Roaming\Typora\typora-user-images\image-20250312154647217.png)







#### from traditional methods to modern deep learning

- Equivariance and Invariance

我们利用Harris detector来寻找图片中的角点，我们希望对于作用在图像上的transformation，这个算法是等变的。

所谓的等变性(a),不变性(b)，数学上为:
$$
\text{If } X \in V \text{ and } f: V \rightarrow V \text{ is a function, and T is a transformation operating X, }\\
f(TX) = Tf(X) \quad (a)\\
f(TX) = f(X)\quad (b)\\
$$

- 判断一个算法是否是等变的/不变的？
  - 首先对于一个分类问题，显然希望是不变的（从image到语义的映射）
  - 对于