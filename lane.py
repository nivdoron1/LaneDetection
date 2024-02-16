# Importing some useful packages

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import LaneLineHistory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# Main video processing loop

video_file = 'car_driving_short.mp4'
cap = cv2.VideoCapture(video_file)
left_image = cv2.imread('arrow.png', cv2.IMREAD_UNCHANGED)
left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
left_image[left_image > 240] = 0

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Get frame size for video writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
previous_left_lane = None
previous_right_lane = None
frame_counter = 0
lane_line_history = LaneLineHistory.LaneLineHistory()
rho = 1
theta = np.pi / 180
threshold = 30
minLineLength = 10
maxLineGap = 150
min_area = 1800  # Minimum area of the contour to be considered
max_area = 50000  # Maximum area of the contour to be considered
aspect_ratio_range = (0.8, 1.2)  # Acceptable aspect ratio range for contours
kernel_size = 13


def list_images(images, cols=2, rows=5, cmap=None):
    """
    Display a list of images in a single figure with matplotlib.
        Parameters:
            images: List of np.arrays compatible with plt.imshow.
            cols (Default = 2): Number of columns in the figure.
            rows (Default = 5): Number of rows in the figure.
            cmap (Default = None): Used to display gray images.
    """
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        # Use gray scale color map if there is only one channel
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def RGB_color_selection(image):
    """
    Apply color selection to RGB images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # White color mask
    lower_threshold = np.uint8([200, 200, 200])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([0, 0, 200])
    upper_threshold = np.uint8([255, 255, 255])
    yellow_mask = cv2.inRange(image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def HSV_color_selection(image):
    """
    Apply color selection to the HSV images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSV
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # White color mask
    lower_threshold = np.uint8([0, 0, 210])
    upper_threshold = np.uint8([255, 30, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([0, 0, 200])
    upper_threshold = np.uint8([30, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def HSL_color_selection(image):
    """
    Apply color selection to the HSL images to blackout everything except for white and yellow lane lines.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    # Convert the input image to HSL
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # White color mask
    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Yellow color mask
    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    # Combine white and yellow masks
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def gray_scale(image):
    """
    Convert images to gray scale.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def canny_detector(image, low_threshold=50, high_threshold=150):
    """
    Apply Canny Edge Detection algorithm to the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
            low_threshold (Default = 50).
            high_threshold (Default = 150).
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def region_selection(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: An np.array compatible with plt.imshow.
    """
    mask = np.zeros_like(image)
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.1, rows * 0.95]
    top_left = [cols * 0.4, rows * 0.6]
    bottom_right = [cols * 0.9, rows * 0.95]
    top_right = [cols * 0.6, rows * 0.6]
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def hough_transform(image):
    """
    Determine and cut the region of interest in the input image.
        Parameters:
            image: The output of a Canny transform.
    """
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines, slope_threshold=0.3):
    """
    Find the slope and intercept of the left and right lanes of each image.
        Parameters:
            lines: The output lines from Hough Transform.
            :param slope_threshold:
    """
    left_lines = []
    left_weights = []
    right_lines = []
    right_weights = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if slope < 0:
                if slope > -1.1 or slope < -2.5:
                    continue
                left_lines.append((slope, intercept))
                left_weights.append(length)
            else:
                if slope < 0.45 or slope > 0.9:
                    continue
                right_lines.append((slope, intercept))
                right_weights.append(length)

    left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
    right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None
    if left_lane is not None and right_lane is not None:
        left_slope = left_lane[0]
        right_slope = right_lane[0]
        if abs(left_slope - right_slope) > slope_threshold:
            left_lane = None

    return get_average_history(left_lane, right_lane)


def fix_slope_lines(left_slope, right_slope, left_lane, right_lane, slope_threshold=0.3):
    if abs(left_slope - right_slope) > slope_threshold:
        left_lane = None  # Discard left lane if slope difference exceeds threshold
        left_lane, right_lane = get_average_history(left_lane, right_lane)
        if abs(left_slope - right_slope) > slope_threshold:
            right_lane = None
        else:
            return left_lane, right_lane


def pixel_points(y1, y2, line):
    """
    Converts the slope and intercept of each line into pixel points.
        Parameters:
            y1: y-value of the line's starting point.
            y2: y-value of the line's end point.
            line: The slope and intercept of the line.
    """
    if line is None:
        return None
    slope, intercept = line
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    y1 = int(y1)
    y2 = int(y2)
    return (x1, y1), (x2, y2)


def lane_lines(image, lines):
    """
    Create full length lines from pixel points.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    y2 = y1 * 0.8
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)


def draw_lane_lines(image, lines, color=None, thickness=12):
    """
    Draw lines onto the input image.
        Parameters:
            image: The input test image.
            lines: The output lines from Hough Transform.
            color (Default = red): Line color.
            thickness (Default = 12): Line thickness.
    """
    if color is None:
        color = [0, 0, 255]
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def draw_crosswalks(frame, edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height = frame.shape[0]
    max_y_from_bottom = height - 110

    # Filter contours and draw them
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Check if the contour is within the desired height
        if y + h >= max_y_from_bottom:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if len(approx) >= 4 and min_area <= area <= max_area:
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 3)
                center_message_counter[0] = 12


def draw_text(width, height, text):
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 255), 2)


def significant_change(previous_lane, current_lane, min_slope_threshold=89, max_slope_threshold=91):
    """
    Determine if there's a significant change in lane position.
    """
    if previous_lane[0] and current_lane[0]:
        prev_slope, prev_intercept = previous_lane[0]
        curr_slope, curr_intercept = current_lane[0]

        # Calculate the change in slope and intercept
        slope_change = abs(curr_slope - prev_slope)
        return min_slope_threshold < slope_change < max_slope_threshold
    return False


def process_frame(frame):
    """
    Process a single video frame to detect lane lines.
        Parameters:
            frame: A single video frame.
    """
    color_select = HSL_color_selection(frame)
    gray = gray_scale(color_select)
    smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    my_lines = lane_lines(frame, hough)
    draw_crosswalks(frame, edges)
    draw_text_arrow()
    result = draw_lane_lines(frame, my_lines)
    return result, my_lines


def get_average_history(left_line, right_line):
    # If a line is not detected, use the average of the last 24 lines
    if left_line is None or right_line is None:
        avg_left_line, avg_right_line = lane_line_history.get_average_line()
        if left_line is None and avg_left_line is not None:
            left_line = avg_left_line
        if right_line is None and avg_right_line is not None:
            right_line = avg_right_line
    # Update the history
    lane_line_history.add_line(left_line, right_line)
    return left_line, right_line


def draw_text_arrow():
    if left_message_counter[0] == 24 or right_message_counter[0] == 24:
        lane_line_history.reset_history()

    if left_message_counter[0] > 0:
        draw_text(frame_width, frame_height, "Left")
        left_message_counter[0] -= 1

    if right_message_counter[0] > 0:
        draw_text(frame_width, frame_height, "Right")
        right_message_counter[0] -= 1

    if center_message_counter[0] > 0:
        center_message_counter[0] -= 1
        draw_text(width=frame_width, height=frame_height, text="Crosswalk")


# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))
left_message_counter = [0]  # Use list to pass by reference
right_message_counter = [0]
center_message_counter = [0]
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Process the frame
        processed_frame, lines = process_frame(frame)
        # Conditionally draw text based on message counters right before displaying or writing the frame

        left_line, right_line = lines

        if frame_counter > 0:  # Skip comparison for the first frame
            if left_line and previous_left_lane:
                # Compare slopes and intercepts for left lane
                if significant_change(previous_left_lane, left_line):
                    right_message_counter[0] = 24  # Set to display message for 24 frames for left lane change
                    print(left_message_counter[0])
                    logging.info("Significant movement detected in right lane.")

            if right_line and previous_right_lane:
                # Compare slopes and intercepts for right lane
                if significant_change(previous_right_lane, right_line):
                    left_message_counter[0] = 24  # Set to display message for 24 frames for right lane change
                    print(right_message_counter[0])
                    logging.info("Significant movement detected in left lane.")

        previous_left_lane, previous_right_lane = left_line, right_line
        frame_counter += 1
        # Write the frame into the output file
        out.write(processed_frame)

        # Display the resulting frame
        cv2.imshow('Frame', processed_frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

# Closes all the frames
cv2.destroyAllWindows()
