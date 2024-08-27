#merge_videos.py

"""place to figure out merging videos together"""

#import neccesary packages and modules
from file_managers import FileManagers
from orthomosaic_tools import OrthomosaicTools


#instantiate file managers and orthomosaic tools
fm = FileManagers()
ot = OrthomosaicTools()

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
length = 90

#choose compression and output speed
compress = 4
speed = 1

#output video file information
output_path = fm.load_dn("Choose a directory to store video in")
outn = "test"

#run function!
ot.orthomosaic_video(videos, gcps_list, offsets, output_path, outn, start_time_s = start, length_s = length, compress_by = compress, out_speed = speed)
