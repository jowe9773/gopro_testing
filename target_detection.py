import cv2
import imageio
import numpy as np
from functions import File_Functions

def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    return gray, edges

def find_and_draw_filtered_contours(frame, edges, min_length, max_length, min_aspect_ratio, max_aspect_ratio):
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours within the length range and aspect ratio range on a copy of the original frame
    contours_frame = frame.copy()
    for contour in contours:
        # Compute perimeter (length) of contour
        perimeter = cv2.arcLength(contour, True)
        if min_length <= perimeter <= max_length:
            # Compute bounding rectangle and aspect ratio
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / float(h) if h != 0 else 0

            if min_aspect_ratio <= aspect_ratio <= max_aspect_ratio:
                # Draw contour
                cv2.drawContours(contours_frame, [contour], -1, (0, 255, 0), 3)
    
    return contours_frame

# Instantiate file functions
ff = File_Functions()

# Open the video
video_fn = ff.load_fn("Choose a video")

# Load the video using imageio
video = imageio.get_reader(video_fn)

# Set scale factor for resizing
scale_factor = 0.5

# Define length and aspect ratio ranges for filtering contours
min_contour_length = 200  # Minimum contour perimeter
max_contour_length = 1000  # Maximum contour perimeter
min_aspect_ratio = .8  # Minimum aspect ratio (width/height)
max_aspect_ratio = 1.2  # Maximum aspect ratio (width/height)

for frame in video:
    # Convert frame from RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # Process the frame to get edges
    gray, edges = process_frame(frame_bgr)
    
    # Find and draw filtered contours
    filtered_contours_frame = find_and_draw_filtered_contours(frame_bgr, edges, min_contour_length, max_contour_length, min_aspect_ratio, max_aspect_ratio)

    # Resize the images
    edges_resized = cv2.resize(edges, (int(edges.shape[1] * scale_factor), int(edges.shape[0] * scale_factor)))
    contours_resized = cv2.resize(filtered_contours_frame, (int(filtered_contours_frame.shape[1] * scale_factor), int(filtered_contours_frame.shape[0] * scale_factor)))

    # Convert grayscale edge detection result to BGR for stacking
    edges_bgr = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)

    # Stack images horizontally (side by side)
    combined_frame = np.hstack((edges_bgr, contours_resized))

    # Resize the final display window for better viewing
    display_resized = cv2.resize(combined_frame, 
                                 (int(combined_frame.shape[1] * scale_factor), 
                                  int(combined_frame.shape[0] * scale_factor)))

    # Show the multi-paned image in a resizable window
    cv2.imshow('Edge Detection and Filtered Contours', display_resized)
    cv2.namedWindow('Edge Detection and Filtered Contours', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Edge Detection and Filtered Contours', int(display_resized.shape[1]), int(display_resized.shape[0]))

    # Exit if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
