# Region-Growing-CUDA-
C++ with CUDA implementation of the Region Growing algorithm.
The Region Growing algorithm implemented grows regions selecting pixels that are similar to one given seed on the image, resulting in a single region approach.
The files can be compiled using the shell script provided.

Region Growing techniques start with a pixel from a potential region and try to grow it adding adjacent pixels while all the considered pixels are similar
by a criterion.
The idea behind this algorithm is to use gray-scale levels images, but the implementation that we’ll see goes beyond by using RGB images.
The starting pixel can either be the first non-labeled pixel in the image or a set of seed pixels.
The proposed version of the algorithm is a single-region implementation, in which a single seed is given as input and the similarity criterion, that tipically
is a sort of statistical test based on variance or mean of the regions, is instead based on the euclidean distance between the RGB colors of seed and the one
of pixels which must be less or equal than an input threshold.
As a result of this particular implementation, the output image will only have RGB values different from black for the pixels that belong to the region
found that satisfy the threshold.

Given as inputs the image, the threshold and the seed coordinates, the goal is to obtain a matrix of the same size as that of the image, we’ll call it
the mask matrix, where the values can be either 0 or 1, meaning that pixel didn’t (or did) satisfy the homogeneity criterion

## Sequential Algorithm Pseudocode

<div align=center>
<img src="https://github.com/MicheleZito/Region-Growing-CUDA-/blob/main/images/alg_1.png" height="400" />
</div>

## Parallel CUDA Algorithm Pseudocode

<div align=center>
<img src="https://github.com/MicheleZito/Region-Growing-CUDA-/blob/main/images/alg_2_cuda.png" height="600" />
</div>

## Examples of Outputs

<div align=center>
<img src="https://github.com/MicheleZito/Region-Growing-CUDA-/blob/main/images/img_1.png" height="400" />
<img src="https://github.com/MicheleZito/Region-Growing-CUDA-/blob/main/images/img_2.png" height="200" />
</div>
