import subprocess as sp

cmd='ffprobe -i input.avi -show_entries format=duration -v quiet -of csv="p=0"'
dur=sp.check_output(cmd,shell=True)
print("dur - ",dur)
dur=16/float(dur)
print("dur - ",dur)
cmd='ffmpeg -i input.avi -vf fps={} out%d.jpg'.format(str(dur))
sp.call(cmd,shell=True)
