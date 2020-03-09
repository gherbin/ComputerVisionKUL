from downsampler import Downsampler
video_file = "D:/Videos/part3.MOV"
output_file = "D:/Videos/part3-1_downsampled.mp4"
downsampler = Downsampler()
downsampler.downsample(video_file,output_file, 2.25)