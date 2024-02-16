# Lane Detection System
## Niv Doron 322547282 && Shelly Doron
The Lane Detection System is designed to identify and highlight lane lines on the road in real-time video streams. This system processes video frames to detect lane lines, potential crosswalks, and significant lane changes, enhancing driver awareness and safety. Below is an overview of its features and functionalities.

## Features

### Lane Line Detection
- **Edge Detection**: Applies the Canny Edge Detector to identify edges in the filtered frames, capturing the structure that could indicate lane lines.
- **Region of Interest Selection**: Masks the video frame to focus on the probable area where lane lines are present, reducing computation on irrelevant areas.
- **Hough Transform**: Detects lines within the specified region of interest by employing the Probabilistic Hough Line Transform, a technique to identify linear patterns.
- **Lane Line Categorization**: Segregates detected lines into left and right lane lines based on their slope and intercept, filtering out horizontal lines and noise.
- **Averaging and Extrapolation**: Averages the positions of detected lines for each lane and extrapolates to full lane lines for a clearer visualization of the lane boundaries.

### Crosswalk Detection
- Detects potential crosswalks using contour detection on the edge-detected frames, based on the geometric properties and area of detected shapes.

### Lane Change Detection
- Monitors for significant changes in lane line orientation to detect potential lane changes or deviations, enhancing the system's utility in dynamic driving scenarios.
### Vehicle Detection
- **Basic Vehicle Detection**: Utilizes shape and contour analysis to identify potential vehicles within the frame. This rudimentary detection aims to enhance situational awareness but lacks the precision of neural network-based systems.

### Curve Adjustment in Lane Detection
- **Adapts to Road Curvature**: Dynamically adjusts detected lane lines to accurately reflect road curvature, ensuring relevant lane boundary visualization even on curved paths.

### Visual Enhancements
- **Lane Line Drawing**: Draws the detected lane lines onto the original video frames, providing real-time feedback on lane positioning.
- **Crosswalk Highlighting**: Highlights detected crosswalks in the video frames to alert the driver.
- **Directional Indicators**: Displays directional arrows to indicate significant lane changes or suggested maneuvers based on the detected lane information.
- **Text Annotations**: Adds text annotations to the video frames for additional context, such as warnings or lane change suggestions.

### History-Based Smoothing
- Employs a history mechanism to smooth lane detection over time, reducing the impact of transient noise and ensuring more stable lane line visualization.

## Implementation Details
This system is implemented using Python, leveraging OpenCV for image processing and video handling. The code structure is modular, with each feature encapsulated in functions for clarity and maintainability. Key components include:

- Edge detection (`canny_detector`) and region of interest selection (`region_selection`) to prepare frames for line detection.
- Line detection using the Hough Transform (`hough_transform`) and line categorization (`average_slope_intercept`) to identify potential lane lines.
- Functions for drawing detected lane lines (`draw_lane_lines`), crosswalks (`draw_crosswalks`), and directional indicators (`draw_arrow`) on video frames.
- A function to process each frame (`process_frame`), applying the above features to detect and visualize lane lines and other road markings in real-time.
- A history mechanism (`LaneLineHistory` class) to average detection over time for smoother results.

---

## Car Detection Feature

### Vehicle Detection and Proximity Alert
- **Vehicle Detection**: The system includes functionality to detect vehicles within the video frame. This is achieved by analyzing each frame for shapes and contours that match the characteristics of a vehicle. Specific regions of the frame can be targeted to optimize detection.
- **Wheel Detection**: An additional layer of validation is applied by checking for circular shapes within detected contours, indicative of wheels, to further confirm the presence of vehicles.
- **Proximity Visualization**: Once a potential vehicle is detected, the system draws a bounding box around it and marks it as "Vehicle" to alert the driver, enhancing situational awareness.

### Dynamic Lane Adjustment
- **Curve Detection**: The lane detection algorithm is capable of identifying not only straight lane lines but also curves. It dynamically adjusts as the vehicle navigates through curved paths, providing accurate lane boundary visualization at all times.
- **Lane Line Correction**: Based on the detected curvature of the lane lines, the system corrects the visualization of the lane lines to match the actual road geometry. This feature is particularly useful in scenarios with winding roads or when changing lanes.

### Limitations
- **Accuracy**: It's important to note that the vehicle detection feature does not operate with the same level of precision as systems based on neural networks or deep learning models. The current implementation relies on traditional computer vision techniques, which, while effective in many scenarios, may not achieve the high accuracy and robustness of AI-based systems.
- **Model Testing**: Due to constraints, we have not been able to test and build a more accurate model for car detection using neural networks. This limitation affects the system's ability to reliably identify vehicles under varying conditions and reduces its overall detection accuracy.

---
## Dynamic Lane Adjustment
- **Curve Detection**: The lane detection algorithm is capable of identifying not only straight lane lines but also curves. It dynamically adjusts as the vehicle navigates through curved paths, providing accurate lane boundary visualization at all times.
- **Lane Line Correction**: Based on the detected curvature of the lane lines, the system corrects the visualization of the lane lines to match the actual road geometry. This feature is particularly useful in scenarios with winding roads or when changing lanes.

---
## Crosswalk Detection

### Overview
The Lane Detection System extends its functionality beyond lane recognition to include the detection of crosswalks. This feature aims to enhance pedestrian safety by alerting drivers to potential crosswalks ahead. Crosswalk detection is performed using edge detection and contour analysis to identify characteristic patterns on the road surface.

### Implementation
- **Edge Detection**: The system applies the Canny Edge Detector to the video frames to highlight edges, which form the basis for identifying crosswalk patterns.
- **Contour Detection**: Following edge detection, the system searches for contours within the frames. These contours are analyzed to identify shapes and patterns that match those of crosswalks.
- **Geometric Analysis**: To differentiate crosswalks from other road markings, the system examines the geometric properties of detected contours, such as shape, size, and orientation. Crosswalks typically exhibit regular, parallel lines that can be distinguished from other markings.
- **Area Filtering**: The detected contours are filtered based on their area. This step ensures that only significant shapes are considered, reducing false positives from small or irrelevant markings.

### Visualization
- **Highlighting Crosswalks**: Once a potential crosswalk is detected, the system draws contours around it on the video frame. This visual cue alerts the driver to the presence of a crosswalk, enhancing pedestrian safety.
- **Alerts and Annotations**: In addition to visual highlighting, the system can provide text annotations or auditory alerts to further draw the driver's attention to the crosswalk.

### Dynamic Adjustment
The crosswalk detection feature is designed to be dynamic, adjusting to various lighting conditions, road textures, and crosswalk designs. It employs adaptive thresholding and contour analysis to maintain effectiveness across different environments.

### Integration with Lane Detection
Crosswalk detection complements the lane detection functionality by providing a more comprehensive understanding of the road environment. By recognizing both lane boundaries and pedestrian crossings, the system offers a holistic view of the road, contributing to safer driving practices.

This feature's implementation uses OpenCV for image processing, relying on its robust functions for edge detection, contour finding, and geometric analysis. The integration of crosswalk detection into the Lane Detection System underscores the system's versatility and its potential to serve as a foundational technology for advanced driver-assistance systems (ADAS).


## Getting Started

Welcome to the Lane Detection System! This system is designed to enhance driving safety by providing real-time lane, vehicle, and crosswalk detection in video streams. To get started with this system, there are a few steps you'll need to follow to ensure everything runs smoothly.

### Requirements

Before running the system, it's essential to install the necessary Python libraries and dependencies. These are outlined in the `requirements.txt` file included with the system. To install these requirements, please run the following command in your terminal:

```bash
pip install -r requirements.txt
```

This command will automatically install all the required libraries, such as OpenCV, NumPy, and Matplotlib, to ensure the Lane Detection System runs without any issues.

### Running the System

After installing the requirements, you can start the system by running the main script. The system processes the video file specified as `car_driving_video.mp4` and applies lane detection, vehicle detection, and crosswalk highlighting techniques to the video frames.

### Output

The result of the lane detection process is saved in an output video file named `output.avi`. This file will be generated in the same directory as the script and contains the original video frames with lane lines, detected vehicles, and crosswalks highlighted. You can view this video to see the system's detection capabilities in action.

We hope you find this Lane Detection System useful for enhancing driving safety and awareness. If you encounter any issues or have suggestions for improvement, please feel free to contribute or reach out to us.