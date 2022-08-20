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
    - 

