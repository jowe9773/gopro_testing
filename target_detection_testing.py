import imageio
import random
import cv2
import numpy as np
from functions import File_Functions

ff = File_Functions()

# Function to average frames to reduce noise
def average_frames(video_path, num_frames=700):
    reader = imageio.get_reader(video_path)
    avg_frame = None
    
    for i, frame in enumerate(reader):
        if i >= num_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert from RGB to BGR
        if avg_frame is None:
            avg_frame = np.float32(frame)
        else:
            cv2.accumulateWeighted(frame, avg_frame, 1.0 / num_frames)
    
    avg_frame = cv2.convertScaleAbs(avg_frame)
    return avg_frame

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)

def clahe_normalization(image, clipLimit = 2.0, tileGridSize = (8,8)):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cl1 = clahe.apply(gray)
    return cl1

# Function to apply Gaussian blur
def apply_gaussian_blur(image, kernel_size=9):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

# Function to identify contours and draw on the original image
def find_and_draw_contours(original_image, edges, aspect_ratio_thresh=0.15, min_area=4000, max_area=7000):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    unfiltered_img = original_image.copy()
    
    result_image = original_image.copy()

    # Filter contours by aspect ratio and bounding box area
    for contour in contours:
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        bounding_box_area = w * h
        box = cv2.minAreaRect(contour)
        box_area = box[1][0] * box[1][1]  # width * height

        cv2.drawContours(unfiltered_img, [contour], -1, color, 2)

        if abs(aspect_ratio - 1) < aspect_ratio_thresh and min_area <= bounding_box_area <= max_area and bounding_box_area*0.5 < box_area:

            # Get the four points of the rectangle
            box_points = cv2.boxPoints(box)
            box_points = np.int64(box_points)  # Convert to integer

            # Draw contour in multicolor
            cv2.drawContours(result_image, [contour], -1, color, 2)

            # Draw bounding box in blue
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw the rotated rectangle in blue
            cv2.drawContours(result_image, [box_points], 0, (255, 0, 255), 2)

            # Draw center dot
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot with radius 5

    return unfiltered_img, result_image

def find_and_draw_lines(original_image, edges, rho=1, theta=np.pi/180, threshold=100, minLineLength=50, maxLineGap=10):
    #find lines
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, minLineLength, maxLineGap)
    
    #copy original image
    line_image = original_image.copy()

    # Draw the lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return line_image


# Function to scale down the frame for display
def scale_down_frame(frame, max_width=800, max_height=600):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_frame

def draw_grid_of_frames(frames, n, window_name='Grid Window'):
    """
    Draws an nxn grid of frames in a window.
    
    Parameters:
        frames (list of np.ndarray): List of frames (images) to display in the grid.
        n (int): Number of rows and columns in the grid (nxn).
        window_name (str): Name of the window to display the grid.
    """
    # Check if we have enough frames to fill the grid
    if len(frames) < n * n:
        raise ValueError(f"Not enough frames to fill an {n}x{n} grid. Need at least {n*n} frames.")

    # Convert all frames to color if they are grayscale
    frames = [cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if len(frame.shape) == 2 else frame for frame in frames]
    scaled_frames = []

    # Scale down the frames for display
    for i, frame in enumerate(frames):
        scaled_frame = scale_down_frame(frame, 1260/n, 945/n)
        scaled_frames.append(scaled_frame)

    # Get the size of the first frame
    frame_height, frame_width = scaled_frames[0].shape[:2]
    
    # Create an empty grid with dimensions to fit all frames
    grid_height = frame_height * n
    grid_width = frame_width * n
    grid = np.zeros((grid_height, grid_width, 3), dtype=np.uint8)
    
    # Place each frame into the grid
    for i in range(n):
        for j in range(n):
            index = i * n + j
            if index < len(scaled_frames):
                # Compute position to place the frame
                y_start = i * frame_height
                y_end = y_start + frame_height
                x_start = j * frame_width
                x_end = x_start + frame_width
                
                # Place the frame into the grid
                grid[y_start:y_end, x_start:x_end] = scaled_frames[index]
    
    # Display the grid
    cv2.imshow(window_name, grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the detection on your video
if __name__ == "__main__":

    video_path = ff.load_fn("Select a Video")
    
    average = average_frames(video_path, 300)

    normalized = clahe_normalization(average)

    gamma_corrected1point25 = adjust_gamma(normalized, gamma=1.25)

    gaussian_blurred = apply_gaussian_blur(gamma_corrected1point25, 9)

    gaussian_edges = cv2.Canny(gaussian_blurred, 50, 100)

    dilated_edges = cv2.dilate(gaussian_edges, (5,5), iterations = 1)

    contours, contours_large = find_and_draw_contours(average, dilated_edges)

    draw_grid_of_frames([contours_large], 1)
