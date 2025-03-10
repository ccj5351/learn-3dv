# A Simple Introduction to 3D Gaussian Splatting

> see: https://medium.com/towards-data-science/a-python-engineers-introduction-to-3d-gaussian-splatting-part-1-e133b0449fc6  
> see: https://github.com/dcaustin33/intro_to_gaussian_splatting  

To begin, we use COLMAP, a software that extracts points consistently seen across multiple images using Structure from Motion (SfM). SfM essentially identifies points (e.g., the top right edge of a doorway) found in more than 1 picture. By matching these points across different images, we can estimate the depth of each point in 3D space. This closely emulates how human stereo vision works, where depth is perceived by comparing slightly different views from each eye. Thus, SfM generates a set of 3D points, each with x, y, and z coordinates, from the common points found in multiple images giving us the “structure” of the scene.

In this tutorial we will use a prebuilt COLMAP scan that is available for [download here](https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip) (Apache 2.0 license). Specifically we will be using the Treehill folder within the downloaded dataset.

