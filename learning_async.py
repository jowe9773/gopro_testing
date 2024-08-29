#learning_async.py

"import packages and modules"
import cv2
from functions import File_Functions, Video_Functions, Audio_Functions


if __name__ == "__main__":

    #instantiate classes
    ff = File_Functions()
    af = Audio_Functions()
    vf = Video_Functions()

    #define important variables
    COMPRESSION = 5
    SPEED = 1
    START_TIME = 56
    LENGTH = 10
    OUT_NAME = "test.mp4"

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


    #Open capture for each video stream
    captures = []
    for i, vid in enumerate(videos):
        cap = cv2.VideoCapture(vid)
        captures.append(cap)

    vf.orthomosaicing(captures, time_offsets, homo_mats, out_vid_dn, OUT_NAME, SPEED, START_TIME, LENGTH, COMPRESSION)