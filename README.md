# Automatic Stitching of Two Images
## Problem
1. Take two views in Sogang University.
2. Develop a ORB + Ransac + homography algorithm to create a panorama image from the two inputs.
3. Apply the algorithm to get a result.
4. Take another set of two views in Sogang University
5. Produce output

- I wanted to create a panorama by combining the two images by calculating the homography matrix using the ORB and Ransac algorithms.
   
## Directory Structure
- `midterm_automatic_stitching_of_two_images.py`: Main program code file
- `midterm_automatic_stitching_of_two_images.ipynb`: If needed, you can refer to this notebook for additional insights or experimentation.
- `case1/image_left.jpg`: Left image file
- `case1/image_right.jpg`: Right image file
- `case1/result_forward.png`: Resultant image using Forward Mapping
- `case1/result_backward_wo_interpolation.png`: Resultant image using Backward Mapping (Without Interpolation)
- `case1/result_backward_w_interpolation.png`: Resultant image using Backward Mapping (With Interpolation)
- The `case2` folder follows a structure similar to that of the `case1` folder.

## How to Use

To run the project, follow these steps:

1. Execute `midterm_automatic_stitching_of_two_images.py`.
2. Use the `--input_left_dir`, `--input_right_dir` and `--output_dir` options to specify the paths of the image files.

Example:

```bash
python midterm_automatic_stitching_of_two_images.py --input_left_dir=path_to_left_image --input_right_dir=path_to_right_image --output_dir=path_to_output_directory
python midterm_automatic_stitching_of_two_images.py --input_left_dir=image_left.jpg --input_right_dir=image_right.jpg --output_dir=./results
```

## Results

