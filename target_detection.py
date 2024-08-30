import cv2
import imageio
import numpy as np
from functions import File_Functions

def process_frame(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    return edges

# Instantiate file functions
ff = File_Functions()

# Open the video
video_fn = ff.load_fn("Choose a video")

# Load the video using imageio
video = imageio.get_reader(video_fn)

# Set scale factor for resizing
scale_factor = 0.5

for frame in video:
    # Process the frame
    edges = process_frame(frame)

    # Resize the frame
    frame_resized = cv2.resize(edges, (int(edges.shape[1] * scale_factor), int(edges.shape[0] * scale_factor)))

    # Display the result
    cv2.imshow('Edge Detection', frame_resized)
    cv2.namedWindow('Edge Detection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Edge Detection', int(frame_resized.shape[1]), int(frame_resized.shape[0]))

    # Exit if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()