#learning_async.py

"import packages and modules"
import sys
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
import asyncio
from functions import File_Functions, Video_Functions, Audio_Functions

#Asyncronous function for parallel processing of frames and async merging
async def process_and_merge_frames(captures, homo_mats, final_shape, compressed_shape, out_writer):
    loop = asyncio.get_event_loop()
    futures = []
    count = 0 

    with ProcessPoolExecutor() as executor:
        while True:
            tasks = []
            
            for i, cap in enumerate(captures):
                ret, frame = cap.read()
                if frame is None or frame.size == 0:
                    print(f"Warning: Frame {i} is empty or could not be read.")
                    continue  # Skip processing this frame

                # Submit frame processing tasks to the executor
                task = loop.run_in_executor(
                    executor,
                    vf.process_frame,
                    frame,
                    homo_mats[i],
                    final_shape,
                    compressed_shape
                )
                tasks.append(task)

            # Wait for all tasks to complete
            processed_frames = await asyncio.gather(*tasks)

            # Merge frames and write to output
            merged = cv2.hconcat(processed_frames)  # Merge the corrected frames together
            out_writer.write(merged)  # Write the merged frame to new video

            # Update count and break condition
            count += 1 / frame_rates[0]
            if count > LENGTH:
                break

            print(count)

    # Release video capture and writer objects
    for cap in captures:
        cap.release()
    out_writer.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == "__main__":

    #instantiate classes
    ff = File_Functions()
    af = Audio_Functions()
    vf = Video_Functions()

    #load video files:
    vid1 = ff.load_fn("Select video from camera 1")
    vid2 = ff.load_fn("Select video from camera 2")
    vid3 = ff.load_fn("Select video from camera 3")
    vid4 = ff.load_fn("Select video from camera 4")

    videos = [vid1, vid2, vid3, vid4]

    #load gcps files:
    gcps1 = ff.load_fn("Select gcps file for camera 1")
    gcps2 = ff.load_fn("Select gcps file for camera 2")
    gcps3 = ff.load_fn("Select gcps file for camera 3")
    gcps4 = ff.load_fn("Select gcps file for camera 4")

    gcpss = [gcps1, gcps2, gcps3, gcps4]

    targets = []
    for i, gcps in enumerate(gcpss):
        gcps = ff.import_gcps(gcps)
        targets.append(gcps)

    #choose an output location for the final video
    out_vid_dn = ff.load_dn("Select output location for the final video")

    #define other important variables
    COMPRESSION = 5
    SPEED = 1
    START_TIME = 56
    LENGTH = 10
    OUT_NAME = "test.mp4"

    #Generate time offsets
    rate_data = []
    audio_data = []
    for i, vid in enumerate(videos):
        rate, audio = af.extract_audio(vid)
        rate_data.append(rate)
        audio_data.append(audio)

    time_offsets = []
    for i, audio in enumerate(audio_data):
        offset = af.find_time_offset(rate_data[i], audio_data[i], rate_data[0], audio_data[0])
        time_offsets.append(offset)
        print(f"Offset between video {1} and video {i+2} found")

    print(time_offsets)


    #Generate homography matricies
    homo_mats = []
    for i, vid in enumerate(videos):
        homography = vf.find_homography(i+1, targets[i])
        homo_mats.append(homography)

    #Describe shape
    final_shape = [2438, 4000]
    compressed_shape = (int(final_shape[0]/COMPRESSION), int(final_shape[1]/COMPRESSION))
    output_shape = (compressed_shape[0]*4, compressed_shape[1])
    print("out Shape: ", output_shape)

    #Open capture for each video stream
    captures = []
    for i, vid in enumerate(videos):
        cap = cv2.VideoCapture(vid)
        captures.append(cap)

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
      

    for i, cap in enumerate(captures):
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time + time_offsets[i])

    # Start the asynchronous processing
    asyncio.run(process_and_merge_frames(captures, homo_mats, final_shape, compressed_shape, out))
