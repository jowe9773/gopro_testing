import imageio
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

# Function to normalize the brightness across the entire image
def normalize_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    normalized_image = cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    return normalized_image

def clahe_normalization(image):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cl1 = clahe.apply(gray)
    return cl1

# Function to apply Gaussian blur
def apply_gaussian_blur(image, kernel_size=9):
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    return blurred_image

# Function to apply Canny edge detection
def apply_canny_edge(image):
    edges = cv2.Canny(image, 50, 100)
    return edges

# Function to identify contours and draw on the original image
def find_and_draw_contours(original_image, edges, aspect_ratio_thresh=0.15, min_area=4000, max_area=7000):
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    result_image = original_image.copy()
    
    # Filter contours by aspect ratio and bounding box area
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        bounding_box_area = w * h
        box = cv2.minAreaRect(contour)
        box_area = box[1][0] * box[1][1]  # width * height
        
        if abs(aspect_ratio - 1) < aspect_ratio_thresh and min_area <= bounding_box_area <= max_area and bounding_box_area*0.9 < box_area:
            # Get the four points of the rectangle
            box_points = cv2.boxPoints(box)
            box_points = np.int64(box_points)  # Convert to integer

            # Draw contour in green
            cv2.drawContours(result_image, [contour], -1, (0, 255, 0), 2)
            
            # Draw bounding box in blue
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Draw the rotated rectangle in blue
            cv2.drawContours(result_image, [box_points], 0, (255, 0, 255), 2)
            
            # Draw center dot
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            cv2.circle(result_image, (center_x, center_y), 5, (0, 0, 255), -1)  # Red dot with radius 5
    
    return result_image


# Function to scale down the frame for display
def scale_down_frame(frame, max_width=800, max_height=600):
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    scaled_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return scaled_frame

# Main function to process a single frame from video and display results
def main(video_path):
    # Average frames to get a single representative frame
    avg_frame = average_frames(video_path, num_frames=300)
    normalized_frame = clahe_normalization(avg_frame)
    blurred_frame = apply_gaussian_blur(normalized_frame, kernel_size=9)
    edges = apply_canny_edge(blurred_frame)
    
    # Find and draw contours based on bounding box area and aspect ratio
    result_image = find_and_draw_contours(avg_frame, edges, aspect_ratio_thresh=0.15, min_area=4000, max_area=7000)
    
    # Scale down image to fit the screen
    scaled_image = scale_down_frame(result_image, max_width=800, max_height=600)
    
    # Display the result image
    cv2.imshow("Contours and Bounding Boxes", scaled_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Run the detection on your video
video_path = ff.load_fn("Select a Video")
main(video_path)
