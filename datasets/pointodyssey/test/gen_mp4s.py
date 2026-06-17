import os
ld = os.listdir('.')
for li in ld:
    if os.path.isdir(li):
        print('working on', li)
        os.system('/usr/bin/ffmpeg -y -hide_banner -loglevel error -f image2 -framerate 24 -pattern_type glob -i "./%s/rgbs/*.jpg" -c:v libx264 -crf 20 -pix_fmt yuv420p %s.mp4' % (li, li))

