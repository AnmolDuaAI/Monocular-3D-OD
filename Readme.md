# M3D-RPN: Monocular 3D Region Proposal Network for Object Detection

- Our framework is comprised of three key components.
    - multi-class 3D region proposal network
    - details of depth-aware convolution and our collective network architecture
    -  postoptimization algorithm for increased 3D→2D consistency

- The core foundation of our proposed framework is based upon the principles of the region proposal network (RPN) first proposed in Faster R-CNN , tailored for 3D.

- Anchor Definations:
    - To simultaneously predict both the 2D and 3D boxes, each anchor template is defined using parameters of both spaces: [w, h]2D ,zP and [w, h, l, θ]3D.
    - For placing an anchor and defining the full 2D / 3D box, a shared center pixel location [x, y]P must be specified.
    - We encode the depth parameter zP by projecting the 3D center location [x, y, z]3D in camera coordinates into the image given a known projection matrix P ∈ R3×4
    - The θ3D represents the observation viewing angle
    - Compared to the Y-axis rotation in the camera coordinate system, the observation angle accounts for the relative orientation of the object with respect to the camera viewing angle rather than the Bird’s Eye View (BEV) of the ground plane. Therefore, the viewing angle is intuitively more meaningful to estimate when dealing with image features.
    - We encode the remaining 3D dimensions [w, h, l]3D as given in the camera coordinate system.

    - The mean statistic for each zP and [w, h, l, θ]3D is precomputed for each anchor individually, which acts as strong prior to ease the difficultly in estimating 3D parameters.
    - Specifically, for each anchor we use the statistics across all matching ground truths which have ≥ 0.5 intersection over union (IoU) with the bounding box of the corresponding [w, h]2D anchor.

- 3D Detection:
    - Our model predicts output feature maps per anchor for c, [tx, ty, tw, th]2D, [tx, ty, tz]P, [tw, th, tl,tθ]3D.
    - Let us denote na the number of anchors, nc the number of classes, and h × w the feature map resolution.
    - total number of box outputs is denoted nb = w × h × na, spanned at each pixel location [x, y]P ∈ R w×h per anchor
    - The first output c represents the shared classification prediction of size na × nc × h × w , whereas each other output has size na × h × w
    - The outputs of [tx, ty, tw, th]2D represent the 2D bounding box transformation, which we collectively refer to as b2D.
    - the bounding box transformation is applied to an anchor with [w, h]2D as fig1.jpg. where xP and yP denote spatial center location of each box.
    - The following 7 outputs represent transformations denoting the projected center [tx, ty, tz]P, dimensions [tw, th, tl]3D and orientation tθ3D , which we collectively refer to as b3D. fig2.png
    - As described, we estimate the projected 3D center rather than camera coordinates to better cope with the convolutional features based exclusively in the image space. Therefore, during inference we back-project the projected 3D center location from the image space  to camera coordinates 3D by using the inverse.
- Loss Defination
    - The network loss of our framework is formed as a multi-task learning problem composed of classification Lc and a box regression loss for 2D and 3D, respectfully denoted as Lb2D and Lb3D .
    - For each generated box, we check if there exists a ground truth with at least ≥ 0.5 IoU
        - If yes then we use the best matched ground truth for each generated box to define a target with τ class index, 2D box ˆb2D, and 3D box ˆb3D
        -  Otherwise, τ is assigned to the catch-all background class and bounding box regression is ignored.
        - A softmax-based multinomial logistic loss is used to supervise for Lc.
        - We use a negative logistic loss applied to the IoU between the matched ground truth box ˆb2D and the transformed b02D 0for Lb2D
        - The remaining 3D bounding box parameters are each optimized using a Smooth L1 [16] regression loss applied to the transformations b3D and the ground truth transformations gˆ3D
        - Hence, the overall multi-task network loss L, including regularization weights λ1 and λ2, is denoted as:
        L = Lc + λ1 Lb2D + λ2 Lb3D

- Depth Aware Convolution
    - We expect that low-level features in the early layers of a network can reasonably be shared and are otherwise invariant to depth or object scale
    - However, we intuitively expect that high-level features related to 3D scene understanding are dependent on depth when a fixed camera view is assumed.
    - The depth-aware convolution layer can be loosely summarized as regular 2D convolution where a set of discretized depths are able to learn non-shared weights and features.
    - We introduce a hyperparameter b denoting the number of row-wise bins to separate a feature map into, where each learns a unique kernel k.
    - Depth-aware kernels enable the network to develop location specific features and biases for each bin region, ideally to exploit the geometric consistency of a fixed viewpoint within urban scenes
    - An obvious drawback to using depth-aware convolution is the increase of memory footprint for a given layer by ×b. However, the total theoretical FLOPS to perform convolution remains consistent regardless of whether kernels are shared

- Network Architecture
    - The backbone of our network uses DenseNet-121.
    - We remove the final pooling layer to keep the network stride at 16, then dilate each convolutional layer in the last DenseBlock by a factor of 2 to obtain a greater field-of-view.
    - We connect two parallel paths at the end of the backbone network.
        - The first path uses regular convolution where kernels are shared spatially, which we refer to as global.
        - The second path exclusively uses depth-aware convolution and is referred to as local.
    - For each path, we append a proposal feature extraction layer using its respective convolution operation to generate Fglobal and Flocal. Each feature extraction layer generates 512 features using a 3 × 3 kernel with 1 padding and is followed by a ReLU non-linear activation.
    - We then connect the 12 outputs to each F corresponding to c, [tx, ty, tw, th]2D, [tx, ty, tz]P, [tw, th, tl, tθ]3D.
    - Each output uses a 1 × 1 kernel and are collectively denoted as O-global and O-local.
    - To leverage the depth-aware and spatialinvariant strengths, we fuse each output using a learned attention α (after sigmoid) applied for i = 1 . . . 12 as follows

- Post 3D -> 2D Optimization
    - We optimize the orientation parameter θ in a simple but effective post-processing algorithm
    - The proposed optimization algorithm takes as input both the 2D and 3D box estimations b2D, [x, y, z]P, and [w, h, l, θ]3D, as well as a step size σ, termination β, and decay γ parameters.
    - The algorithm then iteratively steps through θ and compares the projected 3D boxes with b2D using a L1 loss.

- Implementation Details
    - 