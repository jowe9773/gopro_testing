#testing.py

#LOAD NECCESARY PACKAGES AND MODULES
import concurrent
import csv
import tkinter as tk
from tkinter import filedialog
import tempfile
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.io import wavfile
import cv2
import numpy as np



#FILE MANAGERS
def load_dn(purpose):
    """this function opens a tkinter GUI for selecting a 
    directory and returns the full path to the directory 
    once selected
    
    'purpose' -- provides expanatory text in the GUI
    that tells the user what directory to select"""

    root = tk.Tk()
    root.withdraw()
    directory_name = filedialog.askdirectory(title = purpose)

    return directory_name

def load_fn(purpose):
    """this function opens a tkinter GUI for selecting a 
    file and returns the full path to the file 
    once selected
    
    'purpose' -- provides expanatory text in the GUI
    that tells the user what file to select"""

    root = tk.Tk()
    root.withdraw()
    filename = filedialog.askopenfilename(title = purpose)

    return filename

def import_gcps():
    """module for importing ground control points as lists"""

    gcps_fn = load_fn("GCPs file") #load filename/path of the GCPs

    gcps_rw_list = [] #make list for real world coordinates of GCPs
    gcps_image_list = [] #make list for image coordinates of GCPs

    #Read csv file into a list of real world and a list of image gcp coordinates
    with open(gcps_fn, 'r', newline='') as csvfile:
        # Create a CSV reader object
        csv_reader = csv.reader(csvfile)

        # Skip the header row
        next(csv_reader)

        # Iterate over each row in the CSV file
        for row in csv_reader:
            # Each row is a list where each element represents a column value
            gcps_image_list.append(row[1:3])
            gcps_rw_list.append(row[3:5])

            gcps = [gcps_rw_list, gcps_image_list]

    return gcps

#AUDIO MANAGERS
def extract_audio(video_path):

    print(video_path)
    clip = mp.VideoFileClip(video_path)
    audio = clip.audio
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
        temp_audio_path = temp_audio_file.name
    audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
    rate, audio_data = wavfile.read(temp_audio_path)
    return rate, audio_data

def find_time_offset(rate1, audio1, rate2, audio2):
    # Ensure the sample rates are the same
    if rate1 != rate2:
        raise ValueError("Sample rates of the two audio tracks do not match")

    # Convert stereo to mono if necessary
    if len(audio1.shape) == 2:
        audio1 = audio1.mean(axis=1)
    if len(audio2.shape) == 2:
        audio2 = audio2.mean(axis=1)

    # Normalize audio data to avoid overflow
    audio1 = audio1 / np.max(np.abs(audio1))
    audio2 = audio2 / np.max(np.abs(audio2))

    # Compute cross-correlation
    correlation = correlate(audio1, audio2)
    lag = np.argmax(correlation) - len(audio2) + 1

    # Calculate the time offset in seconds
    time_offset = lag / rate1

    return time_offset


#   ORTHORECTIFICATION MANAGERS
def find_homography(cam, gcps):
    """Method for finding homography matrix."""

    #adjust the ground control points so that they are within the frame of the camera, which starts at (0,0) for each camera
    for count, i in enumerate(gcps[0]):
        i[0] = float(i[0]) - 2438 * (cam-1)
        i[1] = (float(i[1])*-1) + 2000

    #convert the image and destination coordinates to numpy array with float32
    src_pts = np.array(gcps[1])
    src_pts = np.float32(src_pts[:, np.newaxis, :])

    dst_pts = np.array(gcps[0])
    dst_pts = np.float32(dst_pts[:, np.newaxis, :])

    #now we can find homography matrix
    h_matrix = cv2.findHomography(src_pts, dst_pts)

    return h_matrix[0]


def orthomosaic_video_umat(videos, gcps_list, offsets_list, output_dn, outname, start_time_s, length_s, compress_by, out_speed, final_shape = (2438,4000)):
    """ method for orthorecifying videos"""

    compressed_shape = tuple([int(s / compress_by) for s in final_shape])
    output_shape = (int(4 * compressed_shape[0]), compressed_shape[1])

    #choose a place to store output video
    output_fn = output_dn + "\\" + outname + ".mp4"

    #Find homography matricies for each camera
    matrix = find_homography(1, gcps_list[0])
    matrix1 = find_homography(2, gcps_list[1])
    matrix2 = find_homography(3, gcps_list[2])
    matrix3 = find_homography(4, gcps_list[3])

    #create a capture object for each video
    cap = cv2.VideoCapture(videos[0])
    cap1 = cv2.VideoCapture(videos[1])
    cap2 = cv2.VideoCapture(videos[2])
    cap3 = cv2.VideoCapture(videos[3])

    #find frame rate of first video
    fps = cap.get(cv2.CAP_PROP_FPS)

    #select a fourcc to encode the video as
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object to save the output video
    out = cv2.VideoWriter(output_fn, fourcc, fps*out_speed, output_shape)

    #set up start time and count to point to where to start and when to end processing
    start_time = start_time_s *1000
    count = 0

    #start reading frames
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    #set start time for each video (with time offset to align videos in time)
    cap.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[0]))
    cap1.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[1]))
    cap2.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[2]))
    cap3.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[3]))

    while ret and ret1 and ret2 and ret3 and count <= length_s:

        # Convert frames to UMat
        uframe = cv2.UMat(frame)
        uframe1 = cv2.UMat(frame1)
        uframe2 = cv2.UMat(frame2)
        uframe3 = cv2.UMat(frame3)

        

    def process_frame(frame, matrix, compressed_shape, final_shape):
        corrected_frame = cv2.warpPerspective(frame, matrix, final_shape)
        corrected_frame = cv2.resize(corrected_frame, compressed_shape)
        return corrected_frame

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_frame, uframe, matrix, compressed_shape, final_shape),
            executor.submit(process_frame, uframe1, matrix1, compressed_shape, final_shape),
            executor.submit(process_frame, uframe2, matrix2, compressed_shape, final_shape),
            executor.submit(process_frame, uframe3, matrix3, compressed_shape, final_shape)
        ]

        corrected_frames = [f.result() for f in futures]



        # Convert UMat back to numpy array for concatenation
        corrected_frame = corrected_frames[0].get()
        corrected_frame1 = corrected_frames[1].get()
        corrected_frame2 = corrected_frames[2].get()
        corrected_frame3 = corrected_frames[3].get()

        #merge corrected frames
        merged = cv2.hconcat([corrected_frame, corrected_frame1, corrected_frame2, corrected_frame3])

        #write the merged frame to new video
        out.write(merged)

    #move to next frame
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    count = count + 1/fps
    print(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


def orthomosaic_video_original(videos, gcps_list, offsets_list, output_dn, outname, start_time_s, length_s, compress_by, out_speed, final_shape = (2438,4000)):
    """ method for orthorecifying videos"""

    compressed_shape = tuple([int(s / compress_by) for s in final_shape])
    output_shape = (int(4 * compressed_shape[0]), compressed_shape[1])

    #choose a place to store output video
    output_fn = output_dn + "\\" + outname + ".mp4"

    #Find homography matricies for each camera
    matrix = find_homography(1, gcps_list[0])
    matrix1 = find_homography(2, gcps_list[1])
    matrix2 = find_homography(3, gcps_list[2])
    matrix3 = find_homography(4, gcps_list[3])

    #create a capture object for each video
    cap = cv2.VideoCapture(videos[0])
    cap1 = cv2.VideoCapture(videos[1])
    cap2 = cv2.VideoCapture(videos[2])
    cap3 = cv2.VideoCapture(videos[3])

    #find frame rate of first video
    fps = cap.get(cv2.CAP_PROP_FPS)

    #select a fourcc to encode the video as
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Create VideoWriter object to save the output video
    out = cv2.VideoWriter(output_fn, fourcc, fps*out_speed, output_shape)

    #set up start time and count to point to where to start and when to end processing
    start_time = start_time_s *1000
    count = 0

    #start reading frames
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    #set start time for each video (with time offset to align videos in time)
    cap.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[0]))
    cap1.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[1]))
    cap2.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[2]))
    cap3.set(cv2.CAP_PROP_POS_MSEC,(start_time + offsets_list[3]))

    # correct frames with warpPerspective
    corrected_frame = cv2.warpPerspective(frame, matrix, final_shape)
    corrected_frame = cv2.resize(corrected_frame, compressed_shape)
    corrected_frame1 = cv2.warpPerspective(frame1, matrix1, final_shape)
    corrected_frame1 = cv2.resize(corrected_frame1, compressed_shape)
    corrected_frame2 = cv2.warpPerspective(frame2, matrix2, final_shape)
    corrected_frame2 = cv2.resize(corrected_frame2, compressed_shape)
    corrected_frame3 = cv2.warpPerspective(frame3, matrix3, final_shape)
    corrected_frame3 = cv2.resize(corrected_frame3, compressed_shape)

    #merge corrected frames
    merged = cv2.hconcat([corrected_frame, corrected_frame1, corrected_frame2, corrected_frame3])

    #write the merged frame to new video
    out.write(merged)

    #move to next frame
    ret, frame = cap.read()
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    ret3, frame3 = cap3.read()

    count = count + 1/fps
    print(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Release video capture and writer objects
    cap.release()
    out.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()
