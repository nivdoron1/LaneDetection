# Importing some useful packages

import cv2
import matplotlib.pyplot as plt
import numpy as np
import logging
import LaneLineHistory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

video_file = 'car_driving_video.mp4'
cap = cv2.VideoCapture(video_file)
left_arrow = cv2.imread('left-arrow.png')
right_arrow = cv2.imread('right-arrow.png')

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
    Displays a list of images in a grid format.
    :param images: List of images to display.
    :param cols: Number of columns in the grid.
    :param rows: Number of rows in the grid.
    :param cmap: Colormap to use for displaying images.
    """
    plt.figure(figsize=(10, 11))
    for i, image in enumerate(images):
        plt.subplot(rows, cols, i + 1)
        cmap = 'gray' if len(image.shape) == 2 else cmap
        plt.imshow(image, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout(pad=0, h_pad=0, w_pad=0)
    plt.show()


def HSL_color_selection(image):
    """
    Filters the image to only include white and yellow colors using HSL color space.
    :param image: The input image.
    :return: The image with only white and yellow colors.
    """
    converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    lower_threshold = np.uint8([0, 200, 0])
    upper_threshold = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    lower_threshold = np.uint8([10, 0, 100])
    upper_threshold = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted_image, lower_threshold, upper_threshold)

    mask = cv2.bitwise_or(white_mask, yellow_mask)
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image


def canny_detector(image, low_threshold=50, high_threshold=150):
    """
    Applies the Canny edge detector to an image.
    :param image: The input image.
    :param low_threshold: The low threshold for the hysteresis procedure.
    :param high_threshold: The high threshold for the hysteresis procedure.
    :return: A binary image showing the detected edges.
    """
    return cv2.Canny(image, low_threshold, high_threshold)


def region_selection(image):
    """
    Applies a mask to the input image to focus on the region of interest.
    :param image: The input image.
    :return: The image with mask applied.
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
    Detects lines in an image using the Probabilistic Hough Transform.
    :param image: The input image after edge detection.
    :return: An array of lines detected.
    """
    return cv2.HoughLinesP(image, rho=rho, theta=theta, threshold=threshold,
                           minLineLength=minLineLength, maxLineGap=maxLineGap)


def average_slope_intercept(lines, slope_threshold=0.3):
    """
    Averages and categorizes lines into two groups: left and right lanes.
    :param lines: Lines detected by the Hough transform.
    :param slope_threshold: Slope threshold to filter out nearly horizontal lines.
    :return: The average lines for the left and right lanes.
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


def pixel_points(y1, y2, line):
    """
    Converts a line represented by slope and intercept into pixel points.
    :param y1: The y-coordinate of the first point.
    :param y2: The y-coordinate of the second point.
    :param line: The line represented as (slope, intercept).
    :return: Two points that define the line in the image.
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
    Generates full-length lines from the segmented lines detected by Hough Transform.
    :param image: The input image.
    :param lines: The lines detected by the Hough Transform.
    :return: Endpoints of the averaged left and right lane lines.
    """
    left_lane, right_lane = average_slope_intercept(lines)
    y1 = image.shape[0]
    if left_message_counter[0] > 0 or right_message_counter[0] > 0:
        y2 = y1
    else:
        y2 = y1 * 0.8
    return pixel_points(y1, y2, left_lane), pixel_points(y1, y2, right_lane)


def draw_lane_lines(image, lines, color=None, thickness=12):
    """
    Draws lines on an image.
    :param image: The input image.
    :param lines: Lines to draw.
    :param color: Line color.
    :param thickness: Line thickness.
    :return: Image with lines drawn.
    """
    if color is None:
        color = [0, 0, 255]
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, *line, color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 1.0, 0.0)


def has_wheels_within_contour(contour, frame):
    """
    Determines if there are circular shapes within a given contour that might represent wheels.

    :param contour: A single contour to analyze.
    :param frame: The frame from which the contour was extracted.
    :return: A boolean value indicating the presence of circular shapes within the contour.
    """
    x, y, w, h = cv2.boundingRect(contour)
    roi = frame[y:y + h, x:x + w]

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                               param1=100, param2=30, minRadius=10, maxRadius=40)

    return circles is not None


def filter_contours_for_vehicles(contours, frame_height, frame, min_area=5000, max_area=30000,
                                 aspect_ratio_range=(0.8, 3.5), min_y_position=0.1, max_y_position=0.8):
    """
     Filters detected contours to identify potential vehicles based on area, aspect ratio, and position in the frame.

     :param contours: Contours detected in the frame.
     :param frame_height: The height of the frame, used for position-based filtering.
     :param frame: The current video frame.
     :param min_area: Minimum area of a contour to be considered a potential vehicle.
     :param max_area: Maximum area of a contour to be considered a potential vehicle.
     :param aspect_ratio_range: Acceptable aspect ratio range for contours to be considered.
     :param min_y_position: Minimum relative y position (as a fraction of frame height) for a contour to be considered.
     :param max_y_position: Maximum relative y position (as a fraction of frame height) for a contour to be considered.
     :return: A list of contours that are likely to represent vehicles.
     """
    vehicle_contours = []
    for contour in contours:
        if has_wheels_within_contour(contour, frame):
            continue  # Skip contours where wheels are detected
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h
        relative_y_position = y / frame_height
        if (min_area < area < max_area and
                aspect_ratio_range[0] < aspect_ratio < aspect_ratio_range[1] and
                min_y_position < relative_y_position < max_y_position):
            vehicle_contours.append(contour)
    return vehicle_contours


def draw_vehicle_proximity(frame, contours):
    """
    Draws bounding boxes around detected vehicles and labels them.

    :param frame: The video frame on which to draw the bounding boxes.
    :param contours: The contours that represent detected vehicles.
    """
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Vehicle", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


def vehicle_detection(frame):
    """
    Processes the lower half of the frame to detect vehicles, applies morphological transformations,
    finds contours, filters them for potential vehicles, and draws bounding boxes around them.

    :param frame: The current video frame to process.
    :return: The video frame with vehicle detections drawn on the lower half.
    """
    frame_height = frame.shape[0]
    lower_half_frame = frame[frame_height // 2:, :]

    hsv = cv2.cvtColor(lower_half_frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 168], dtype=np.uint8)
    upper_white = np.array([172, 111, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_white, upper_white)

    result = cv2.bitwise_and(lower_half_frame, lower_half_frame, mask=mask)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    vehicle_contours = filter_contours_for_vehicles(contours, lower_half_frame.shape[0], frame, min_y_position=0.2)

    draw_vehicle_proximity(lower_half_frame, vehicle_contours)

    frame[frame_height // 2:, :] = lower_half_frame

    return frame


def draw_crosswalks(frame, edges):
    """
    Detects and draws crosswalks on the frame based on edges.
    :param frame: The current video frame.
    :param edges: The edge-detected image.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    height = frame.shape[0]
    max_y_from_bottom = height - 110

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if y + h >= max_y_from_bottom:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(contour)
            if len(approx) >= 4 and min_area <= area <= max_area:
                cv2.drawContours(frame, [contour], 0, (0, 255, 0), 3)
                center_message_counter[0] = 12


def draw_text(width, text, top_margin=50):
    """
    Draws text centered on the frame.
    :param width: The width of the frame.
    :param text: The text to draw.
    :param top_margin: The top margin from the frame's top edge.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = (width - text_size[0]) // 2  # Center the text horizontally
    text_y = top_margin + text_size[1]  # Position the text top_margin pixels from the top
    cv2.putText(frame, text, (text_x, text_y), font, 1, (0, 255, 255), 2)


def draw_arrow(frame, arrow_img, position):
    """
    Overlays an arrow image onto the frame at the specified position.
    :param frame: The video frame.
    :param arrow_img: The arrow image to overlay.
    :param position: A tuple (x, y) representing the top-left corner where the arrow image will be placed.
    """
    y1, y2 = position[1], position[1] + arrow_img.shape[0]
    x1, x2 = position[0], position[0] + arrow_img.shape[1]

    roi = frame[y1:y2, x1:x2]
    result = cv2.addWeighted(roi, 1, arrow_img, 1, 0)
    frame[y1:y2, x1:x2] = result


def significant_change(previous_lane, current_lane, min_slope_threshold=89, max_slope_threshold=91):
    """
    Determines if there is a significant change between the previous and current lane detections.
    :param previous_lane: The previous lane detection.
    :param current_lane: The current lane detection.
    :param min_slope_threshold: Minimum slope change threshold.
    :param max_slope_threshold: Maximum slope change threshold.
    :return: Boolean indicating if there is a significant change.
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
    Processes a video frame and applies lane detection and drawing routines.
    :param frame: The video frame to process.
    :return: The processed frame with lane markings.
    """
    color_select = HSL_color_selection(frame)
    gray = cv2.cvtColor(color_select, cv2.COLOR_RGB2GRAY)
    smooth = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
    edges = canny_detector(smooth)
    region = region_selection(edges)
    hough = hough_transform(region)
    lines = lane_lines(frame, hough)
    draw_crosswalks(frame, edges)
    draw_text_arrow()
    result = draw_lane_lines(frame, lines)
    return result, lines


def get_average_history(left_line, right_line):
    """
    Averages line history to smooth the lane detection over time.
    :param left_line: The current left lane line.
    :param right_line: The current right lane line.
    :return: The averaged left and right lane lines.
    """
    if left_line is None or right_line is None:
        avg_left_line, avg_right_line = lane_line_history.get_average_line()
        if left_line is None and avg_left_line is not None:
            left_line = avg_left_line
        if right_line is None and avg_right_line is not None:
            right_line = avg_right_line
    lane_line_history.add_line(left_line, right_line)
    return left_line, right_line


def draw_text_arrow():
    """
    Determines the text and arrow direction to display based on the message counters.
    """
    if left_message_counter[0] == 85 or right_message_counter[0] == 85:
        lane_line_history.reset_history()
    arrow_position_x = frame_width // 2 - right_arrow.shape[1] // 2
    arrow_position_y = frame_height // 2 - right_arrow.shape[0] // 2
    if left_message_counter[0] > 0:
        draw_arrow(frame, left_arrow, (arrow_position_x, arrow_position_y))
        draw_text(width=frame_width, text="Left")
        left_message_counter[0] -= 1

    if right_message_counter[0] > 0:
        draw_arrow(frame, right_arrow, (arrow_position_x, arrow_position_y))
        draw_text(width=frame_width, text="Right")
        right_message_counter[0] -= 1

    if center_message_counter[0] > 0:
        center_message_counter[0] -= 1
        draw_text(width=frame_width, text="Crosswalk")


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_width, frame_height))
left_message_counter = [0]  # Use list to pass by reference
right_message_counter = [0]
center_message_counter = [0]
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        processed_frame, lines = process_frame(frame)

        left_line, right_line = lines
        processed_frame = vehicle_detection(processed_frame)
        if frame_counter > 0:
            if left_line and previous_left_lane:
                if significant_change(previous_left_lane, left_line):
                    right_message_counter[0] = 85
                    print(left_message_counter[0])
                    logging.info("Significant movement detected in right lane.")

            if right_line and previous_right_lane:
                if significant_change(previous_right_lane, right_line):
                    left_message_counter[0] = 85
                    print(right_message_counter[0])
                    logging.info("Significant movement detected in left lane.")

        previous_left_lane, previous_right_lane = left_line, right_line
        frame_counter += 1
        out.write(processed_frame)

        cv2.imshow('Frame', processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()

cv2.destroyAllWindows()
