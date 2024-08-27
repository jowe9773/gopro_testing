#video_processing_tools.py
"""This module contains the tools to orthorectify and merge video from multiple overhead cameras"""
import cv2
import cProfile
import concurrent.futures
from orthomosaic_tools import OrthomosaicTools
from file_managers import FileManagers


fm = FileManagers()
ot = OrthomosaicTools()

output_path = fm.load_dn("Choose a directory to store video in")

def orthomosaic_video(videos, gcps_list, offsets_list, output_dn, outname, start_time_s, length_s, compress_by, out_speed, final_shape = (2438,4000)):
    """ method for orthorecifying videos"""

    compressed_shape = tuple([int(s / compress_by) for s in final_shape])
    output_shape = (int(4 * compressed_shape[0]), compressed_shape[1])

    #instantiate the orthomosaic tools module
    ot = OrthomosaicTools()

    #choose a place to store output video
    output_fn = output_dn + "\\" + outname + ".mp4"

    #Find homography matricies for each camera
    matrix = ot.find_homography(1, gcps_list[0])
    matrix1 = ot.find_homography(2, gcps_list[1])
    matrix2 = ot.find_homography(3, gcps_list[2])
    matrix3 = ot.find_homography(4, gcps_list[3])

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

            """# correct frames with warpPerspective
            corrected_frame = cv2.warpPerspective(uframe, matrix, final_shape)
            corrected_frame = cv2.resize(corrected_frame, compressed_shape)
            corrected_frame1 = cv2.warpPerspective(uframe1, matrix1, final_shape)
            corrected_frame1 = cv2.resize(corrected_frame1, compressed_shape)
            corrected_frame2 = cv2.warpPerspective(uframe2, matrix2, final_shape)
            corrected_frame2 = cv2.resize(corrected_frame2, compressed_shape)
            corrected_frame3 = cv2.warpPerspective(uframe3, matrix3, final_shape)
            corrected_frame3 = cv2.resize(corrected_frame3, compressed_shape)"""

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



#select videos and corresponding GCPS file
video = fm.load_fn("Video")
video1 = fm.load_fn("Video1")
video2 = fm.load_fn("Video2")
video3 = fm.load_fn("Video3")


gcps = fm.import_gcps()
gcps1= fm.import_gcps()
gcps2 = fm.import_gcps()
gcps3 = fm.import_gcps()

#create lists to send to the function we made
videos = [video, video1, video2, video3]
print(videos)

gcps_list = [gcps, gcps1, gcps2, gcps3]
print(gcps_list)

#time offsets between the videos (in ms)
rate1, audio1  = ot.extract_audio(videos[0])
rate2, audio2 = ot.extract_audio(videos[1])
rate3, audio3 = ot.extract_audio(videos[2])
rate4, audio4 = ot.extract_audio(videos[3])

offset2 = ot.find_time_offset(rate1, audio1, rate2, audio2)
offset3 = ot.find_time_offset(rate2, audio2, rate3, audio3) + offset2
offset4 = ot.find_time_offset(rate3, audio3, rate4, audio4) + offset3


offsets = [0, offset2*-1000, offset3*-1000, offset4*-1000]
print(offsets)

#choose start time for first video and length to process
start = 56
length = 3

#choose compression and output speed
compress = 4
speed = 1

#output video file information

outn = "test"

cProfile.run('orthomosaic_video(videos, gcps_list, offsets, output_path, outn, start_time_s = start, length_s = length, compress_by = compress, out_speed = speed)')