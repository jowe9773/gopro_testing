#functions.py

'''A file that holds all of the functions used in the process of orthomosaicing the video files.'''

#import neccesary packages and modules
import csv
import tkinter as tk
from tkinter import filedialog
import tempfile
import sys
import moviepy.editor as mp
from scipy.signal import correlate
from scipy.io import wavfile
import cv2
import numpy as np



class File_Functions():
    '''Class containing methods for user handling of files.'''

    def __init__(self):
        print("Initialized File_Funtions")

    def load_dn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        directory and returns the full path to the directory 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what directory to select"""

        root = tk.Tk()
        root.withdraw()
        directory_name = filedialog.askdirectory(title = purpose)

        return directory_name

    def load_fn(self, purpose):
        """this function opens a tkinter GUI for selecting a 
        file and returns the full path to the file 
        once selected
        
        'purpose' -- provides expanatory text in the GUI
        that tells the user what file to select"""

        root = tk.Tk()
        root.withdraw()
        filename = filedialog.askopenfilename(title = purpose)

        return filename

    def import_gcps(self, gcps_fn):
        """module for importing ground control points as lists"""

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
    
class Audio_Functions():
    def __init__(self):
        print("Initialized Audio_Functions")

    def extract_audio(self, video_path):
        """this method extracts audio from an mp4 and saves it as a tempoorary wav file."""

        print(video_path)
        clip = mp.VideoFileClip(video_path)
        audio = clip.audio
        # Save audio to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_audio_file:
            temp_audio_path = temp_audio_file.name
        audio.write_audiofile(temp_audio_path, codec='pcm_s16le')
        rate, audio = wavfile.read(temp_audio_path)
        clip.reader.close()
        return rate, audio

    def find_time_offset(self, rate1, audio1, rate2, audio2):
        """This function compares two audiofiles and lines up the wave patterns to match in time.
        Retuns the time offset."""

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

        # Calculate the time offset in milliseconds
        time_offset = lag / rate1 * 1000

        return time_offset

class Video_Functions():
    def __init__(self):
        print("Initialized Video_Functions.")

    def find_homography(self, cam, gcps):
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

    def frame_to_umat_frame(self, frame):
        uframe = cv2.UMat(frame)
        return uframe

    def orthomosaicing(self, captures, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION):
            #Describe shape
        final_shape = [2438, 4000]
        compressed_shape = (int(final_shape[0]/COMPRESSION), int(final_shape[1]/COMPRESSION))
        output_shape = (compressed_shape[0]*4, compressed_shape[1])
        print("out Shape: ", output_shape)

        #Find frame rates for the videos and ensure that they match
        frame_rates = []
        for i, cap in enumerate(captures):
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_rates.append(fps)

        fps_check = all(x == frame_rates[0] for x in frame_rates) #ensures matching fps for all videos

        if fps_check is True:
            print("All captures have same FPS.")

        else:
            print("FPS from all captures do not match. Check and try again.")
            sys.exit()

        #Get video writer set up
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')     #select a fourcc to encode the video

        out = cv2.VideoWriter(out_vid_dn + "/" + OUT_NAME, fourcc, frame_rates[0]*SPEED, output_shape)    # create VideoWriter object to save the output video

        #Set start frames and counter to end video after the specified length of time
        start_time = START_TIME * 1000
        count = 0   

        for i, cap in enumerate(captures):
            cap.set(cv2.CAP_PROP_POS_MSEC, start_time + time_offsets[i])

        while True and count <= LENGTH:
            corrected_frames = []
            for i, cap in enumerate(captures):
                ret, frame = cap.read()
                if frame is None or frame.size == 0:
                    print(f"Warning: Frame {i} is empty or could not be read.")
                    continue  # Skip processing this frame\

                corrected_frame = cv2.warpPerspective(frame, homo_mats[i], final_shape)
                corrected_frame = cv2.resize(corrected_frame, compressed_shape)
                corrected_frames.append(corrected_frame)

            merged = cv2.hconcat(corrected_frames)  #merge the four corrected frames together
            out.write(merged)   #write the merged frame to new video

            count += 1/frame_rates[0]
            print(count)

        # Release video capture and writer objects
        cap.release()
        out.release()

        # Close all OpenCV windows
        cv2.destroyAllWindows()
