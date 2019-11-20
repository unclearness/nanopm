# NanoPM, single header only PatchMatch
**NanoPM** is a single header-only implementation of PatchMatch algorithm written in C++. Could be used for variety of applications.

| |PatchMatch (350 ms.)|BruteForce as Ground Truth (18 min.)|
|---|---|---|
|NNF (Nearest Neighbor Field) |![](https://raw.github.com/wiki/unclearness/nanopm/images/art_nnf_20191120.gif)|![](data/scenes2005/Art/nnf_bruteforce.jpg)|
|Distance|![](https://raw.github.com/wiki/unclearness/nanopm/images/art_dist_20191120.gif)|![](data/scenes2005/Art/distance_bruteforce.jpg)|

## Example applications

- dense optical flow

|src|dst|optical flow|
|---|---|---|
|![](data/scenes2005/Art/view1.png)|![](data/scenes2005/Art/view5.png)|![](data/scenes2005/Art/nnf.jpg)|

- image synthesis (WIP)
- inpainting (WIP)

# Optional Dependencies
You can include optional dependencies in nanopm.h (mainly for I/O) but it will no longer be "single header-only".
- stb (default ON)
    https://github.com/nothings/stb
    - Image I/O
- OpenCV (default OFF)
    - cv::Mat_ as Image class. Image I/O
- OpenMP (default OFF)
    (if supported by your compiler)
    - Multi-thread accelaration


# Reference
- Barnes, Connelly, et al. "PatchMatch: A randomized correspondence algorithm for structural image editing." ACM Transactions on Graphics (ToG). Vol. 28. No. 3. ACM, 2009.

# Data
Borrowed Middlebury Stereo Datasets. Original data is from
http://vision.middlebury.edu/stereo/data/scenes2005/ThirdSize/zip-2views/ALL-2views.zip
- D. Scharstein and C. Pal. Learning conditional random fields for stereo.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007.
- H. Hirschmüller and D. Scharstein. Evaluation of cost functions for stereo matching.
In IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR 2007), Minneapolis, MN, June 2007.

