## Task 2-	Histogram Of Oriented Gradients

Matriculation Number: 100004430
Date Of submission: 09-07-2025
Introduction:
Image histogram is a type of histogram which reflects the intercity distribution of an image plotting the number of pixels for each intensity values. The number of each intensity value is called a frequency. The histogram shows the number of pixels for every intensity value, ranging from 0 to 255. Each of these 256 values is called a bin in histogram terminology. There are two types of histogram process i.e. 1. AHE (Adaptive Histogram Equalization) – low contrast to high contrast 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) – Use clipping value to increase the impurity.

Let’s discuss about the HOG Feature descriptor, after that we can conclude about HOG.
What is feature descriptor?
A feature descriptor is a representation of an image or an image patch that simplifies the image by extracting useful information and throwing away extraneous information.
Typically, a feature descriptor converts an image of size width x height x 3 (channels ) to a feature vector / array of length n. In the case of the HOG feature descriptor, the input image is of size 64 x 128 x 3 and the output feature vector is of length 3780.

In the HOG feature descriptor, the distribution ( histograms ) of directions of gradients ( oriented gradients ) are used as features. Gradients ( x and y derivatives ) of an image are useful because the magnitude of gradients is large around edges and corners ( regions of abrupt intensity changes ) and we know that edges and corners pack in a lot more information about object shape than flat regions.
Methodology and steps:
•	Gradient computation: Calculates the gradient magnitude and direction at each pixel.
•	Orientation binning: Divides the image into small cells and bins the gradient directions.
•	Block normalization: Groups cells into blocks and normalizes the histograms to improve invariance to illumination.
•	Feature vector: Concatenates all normalized histograms into a single feature vector.
Gradient computation:
For a grayscale image I(x,y)I(x,y), the gradient at each pixel is computed in two directions:
Horizontal gradient (Gx):
Gx=I(x+1,y)−I(x−1,y)Gx=I(x+1,y)−I(x−1,y)
Vertical gradient (Gy):
Gy=I(x,y+1)−I(x,y−1)Gy=I(x,y+1)−I(x,y−1)
These are approximated using convolution kernels, like the Sobel operator:
Once you have GxGx and GyGy, you compute:
Gradient Magnitude:
M=Gx2+Gy2M=Gx2+Gy2
Gradient Orientation (Angle):
θ=tan⁡−1(GyGx)θ=tan−1(GxGy)
Orientation Binding:
 
The image was divided into cells of size 8×8 pixels.
For each cell, we computed a histogram of gradient directions (using 9 bins from 0° to 180°).
Each small plot represents the distribution of edge directions within that cell.
This is a key step in HOG feature extraction, where local edge patterns are captured and later normalized across blocks for robustness.\

Block Normalization:
A block is a group of adjacent cells (typically 2×2).Each cell has a histogram of gradient orientations (9 bins). These histograms are flattened into a single vector and then normalized using L2 norm-
 
This normalization improves robustness to lighting and contrast changes.

Feature Vector:
 
This is the histogram representation for a 3780 feature vector. The feature vector has 3780 elements, which is typical for a 64×128 image patch using standard HOG settings.Each value represents the strength of a particular gradient orientation in a specific block.This vector is used as input to machine learning models like SVMs for tasks such as object detection or image classification.
Sobel Operator:
The Sobel operator is a popular edge detection technique used in image processing and computer vision. It helps identify regions in an image where intensity changes sharply—these are typically edges or boundaries of objects.
________________________________________
What Does the Sobel Operator Do?
It calculates the gradient of image intensity at each pixel, which tells us:
How strong the change in brightness is (magnitude).
In which direction the change occurs (orientation).
________________________________________
How It Works
The Sobel operator uses convolution kernels to approximate derivatives:
One kernel detects horizontal edges (changes along the x-axis).
Another detects vertical edges (changes along the y-axis).

Why Use Sobel Instead of Simple Derivatives?
•	It smooths the image slightly (reduces noise) while computing the gradient.
•	It’s more robust than basic derivative filters like Prewitt or Roberts.
________________________________________
Applications
•	Edge detection
•	Feature extraction (e.g., for object recognition)
•	Image segmentation
•	Motion detection

Implementation:
•	Platform: Python
•	IDE: Spyder
•	Libraries: Opencv, Matplotlib
•	Input: Normal jpg image
•	Output: Image containing grey scale, Gradient magnitude, Gradient orientation image
Code:
 
Code File:
 

Actual Image:
 

Resultant Image:
 

HOG Usage and its implementation:
Python script uses OpenCV and Matplotlib to perform pedestrian detection in an image using the Histogram of Oriented Gradients (HOG) descriptor.
This code is useful for:
•	Surveillance systems: Detecting people in security footage.
•	Autonomous vehicles: Identifying pedestrians for safety.
•	Smart cities: Monitoring pedestrian traffic.
•	Robotics: Helping robots navigate environments with humans.
Code snippet:
 

Actual Image:

 


After running the code:
 
Code file:
 

Conclusion:
Conclusion on HOG (Histogram of Oriented Gradients)
The Histogram of Oriented Gradients (HOG) is a powerful feature descriptor used primarily for object detection, especially pedestrian detection. Here's a concise conclusion:
________________________________________
Key Takeaways:
Gradient-Based: HOG captures edge and gradient structures that are characteristic of local shapes in an image.
Robust to Illumination: By focusing on gradient orientation rather than intensity, HOG is less sensitive to lighting changes.
Effective for Detection: When combined with a classifier like SVM, HOG is highly effective for detecting humans and other objects.
Widely Used: It has been a foundational technique in computer vision, especially before deep learning became dominant.

Reference Used:
AI Platform: Copilot
Lecture notes
Research Paper: https://learnopencv.com/histogram-of-oriented-gradients/
GitHub Repo: https://github.com/spmallick/learnopencv/

## Task 3-	Object Detection Process
Matriculation Number: 100004430
Date Of submission: 15-07-2025
Introduction:
This report presents the implementation of an object detection algorithm based on the sliding window technique. The objective of the task is to detect the presence and locations of an object (represented in a given grayscale image, referred to as imageA) within a larger scene (imageB). The detection process leverages shape-based descriptors, specifically chain code representations, to analyse contour patterns and identify similar regions in the scene.
The sliding window approach systematically scans imageB using a fixed-size window, comparing each region with the object in imageA. This method ensures localized analysis while maintaining a manageable computational load. The detection is performed using several input parameters including the number of chaincode directions, grid dimension, window size, and sliding steps along both axes.
This document outlines the methodology, implementation details, example inputs and outputs, and an evaluation of the detection performance across various scenarios.
Theoretical Aspect:
We can discuss step by step the theoretical aspect of this topic. Lets discuss about few the important definition for object detection-
Chaincode:
Chaincode is a method used to represent the boundary or contour of a binary object in a digital image using a sequence of directional moves. It encodes the shape of the object by tracing its boundary pixel by pixel, assigning a direction (typically from 0 to 7 in 8-connectivity) to each step taken. This representation is compact, efficient, and retains the geometric structure of the object, making it useful for shape analysis and pattern recognition.

Example: In 8-directional Freeman Chaincode, direction values are assigned as follows:
0 = right, 1 = top-right, 2 = up, ..., 7 = bottom-right.

Contour:
A contour refers to the boundary or outline of an object in an image. In image processing, contours are curves joining all continuous points along a boundary that share the same intensity or color. Detecting contours helps isolate and analyze shapes within an image, especially in binary images. They are fundamental in object segmentation, shape analysis, and recognition tasks.

Example: Contours are typically extracted using edge detection followed by boundary tracing algorithms.


Sliding Window Algorithm:
The sliding window algorithm is a widely used technique in computer vision and pattern recognition for processing and analyzing subsets of an image or data structure. A window of fixed size moves across the input image step-by-step (horizontally and vertically), allowing localized examination of content.

Purpose: It enables region-based operations such as object detection, template matching, or feature extraction.
Parameters include:
•	Window size: the dimensions of the region to analyze at a time.
•	Step size: the number of pixels the window shifts between evaluations.

Object Detection:
Object detection is the process of identifying and locating instances of objects (such as shapes, symbols, or patterns) within an image or video. It combines classification (what the object is) with localization (where the object is). In classical approaches, techniques like template matching, sliding window search, and feature descriptors are used, whereas modern approaches often involve deep learning.

In this project: Object detection is implemented by comparing the chaincode of a reference object (imageA) with regions in a target scene (imageB) using the sliding window technique.
Methodology:
Here we are discussing about the methodology to detect and object using two images- 
1. Pattern Recognition and Shape Matching
The task implies recognizing an object (imageA) within another image (imageB), which is a classic pattern recognition problem. The use of chaincode suggests a shape-based representation, meaning the algorithm compares the structure of objects rather than their pixel values directly.
•	Derived Concepts:
o	Shape descriptors (e.g., chaincode, Fourier descriptors)
o	Similarity metrics (edit distance, correlation)
o	Template matching using shape representation
________________________________________
2. Feature Extraction
By converting the contour into a chaincode, we are effectively extracting features from the object. This process reduces dimensionality and focuses on the most informative part of the image (the boundary or edge).
•	Derived Concepts:
o	Feature vectors
o	Invariant features (rotation, scale, translation)
o	Grid-based spatial sampling (from the mention of grid dimensions)
________________________________________
3. Region-Based Analysis using Sliding Windows
The sliding window algorithm enables localized inspection of the larger image. Each sub-window is treated as an independent candidate for matching.
•	Derived Concepts:
o	Multi-scale detection (if windows of different sizes are used)
o	Overlapping vs. non-overlapping windows
o	Trade-offs between accuracy and computational efficiency
________________________________________
4. Parameterization in Computer Vision Algorithms
The mention of number of chaincode directions, grid dimensions, window dimensions, and step sizes highlights how the algorithm can be tuned or optimized based on different scenarios.
•	Derived Concepts:
o	Hyperparameter tuning
o	Sensitivity analysis
o	Impact of spatial resolution on detection accuracy
________________________________________
5. Binary Image Processing
The task assumes the input images are grayscale, but the actual contour extraction often requires binary thresholding to identify object boundaries.
•	Derived Concepts:
o	Image thresholding (global, adaptive)
o	Morphological operations (e.g., dilation, erosion to clean up binary masks)
o	Edge detection as a precursor to contour extraction
________________________________________
6. Object Detection Pipeline
The overall structure aligns with a standard computer vision pipeline:
1.	Input: Grayscale images (imageA, imageB)
2.	Preprocessing: Thresholding to binary
3.	Feature Extraction: Chaincode from contours
4.	Search: Sliding window
5.	Matching: Chaincode comparison
6.	Output: List of object matches and locations
•	Derived Concepts:
o	Modularity of vision systems
o	Potential to integrate machine learning or CNNs in future iterations
________________________________________
7. Performance Metrics (Implied)
Although not explicitly mentioned, any detection algorithm can be evaluated using common metrics like:
•	Precision and Recall
•	Detection rate
•	False positives/negatives
•	Computational cost

Implementation:
•	Platform: Python
•	IDE: Spyder
•	Libraries: Opencv, Matplotlib
•	Input: Normal jpg image
•	Output: Image containing grey scale, Object detected image


Code File:
 
Code Snippet:
 

 

 
Example Use Cases
Example Type	ImageA	ImageB
Basic Shape	One Black Square	Grid with many square
Letters Character	Icon logo	Full screenshot of a mobile screen
Industrial Part	One gear	Chain Of gears
Letters	Isolated A	Screenshot of Group of A

Output with plot visualization:
Use case 1:
Image A
 

Image B

 
Output Image with result:
  

 

Output Explanation:
•	The output is a visual plot showing:
•	Green rectangles around detected regions in imageB.
•	Blue text showing the match score.
•	Lower scores indicate better matches.
•	The plot helps visually verify if the object from imageA was correctly found in imageB.

Use Case-2:

Image C
 
Image D
 

Output Image with result:
 

 


Output Explanation:
•	The output is a visual plot showing:
•	Green rectangles around detected regions in imageD.
•	Blue text showing the match score.
•	Lower scores indicate better matches.
•	The plot helps visually verify if the object from imageC was correctly found in imageD.

Use Case-3

Image E
 

Image F
 

Output Image with result:
 
Output Explanation:
•	The output is a visual plot showing:
•	Green rectangles around detected regions in imageF.
•	Blue text showing the match score.
•	Lower scores indicate better matches.
•	The plot helps visually verify if the object from imageE was correctly found in imageF.

Summary
This logic:
•	Efficiently scans the image in a grid-like fashion.
•	Extracts patches for further analysis (e.g., chaincode comparison).
•	Ensures no patch goes out of bounds.
•	Is flexible: you can change window_size and step_size to control granularity.
Conclusion:
This project demonstrates a successful application of chaincode-based object detection using the sliding window method. By encoding shape contours into a directional representation, the system effectively locates object instances in complex scenes. The modular design allows flexibility in parameter tuning—such as grid size, chaincode resolution, and step size—making it adaptable to a variety of image contexts.
While the method is efficient for simple binary patterns and rigid shapes, future improvements may include:
Incorporation of rotation and scale invariance in matching,
Use of dynamic time warping (DTW) or other shape-matching algorithms,
Optimization of window search through image pyramids or coarse-to-fine strategies.
The results across various test cases (basic shapes, letters, logos, and natural objects) show promising accuracy, validating the effectiveness of this technique in contour-based object detection tasks.





























