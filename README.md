# NanoPM, single header only PatchMatch
**NanoPM** is a header-only implementation of PatchMatch algorithm written in C++.

# Dependencies
## Mandatory
- None
## Optional
- OpenCV
    - cv::Mat_ as Image class. Image I/O
- stb
    https://github.com/nothings/stb
    - Image I/O
- LodePNG
    https://github.com/lvandeve/lodepng
    - .png I/O particularly for 16bit writing that is not supported by stb
- tinycolormap
    https://github.com/yuki-koyama/tinycolormap
    - Colorization.
- OpenMP
    (if supported by your compiler)
    - Multi-thread accelaration


# Reference
- Barnes, Connelly, et al. "PatchMatch: A randomized correspondence algorithm for structural image editing." ACM Transactions on Graphics (ToG). Vol. 28. No. 3. ACM, 2009.
