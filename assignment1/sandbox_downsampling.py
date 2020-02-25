from downsampler import Downsampler
video_file = "D:/Videos/downsampled.mov"
output_file = "D:/Videos/downsampled2.mp4"
downsampler = Downsampler()
downsampler.downsample(video_file,output_file, 2.25)