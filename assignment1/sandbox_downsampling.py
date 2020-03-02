from downsampler import Downsampler
video_file = "D:/Videos/part2.MOV"
output_file = "D:/Videos/part2_downsampled.mp4"
downsampler = Downsampler()
downsampler.downsample(video_file,output_file, 2.25)