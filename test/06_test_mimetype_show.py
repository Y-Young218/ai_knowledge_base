from mimetypes import guess_type

image_name = "xx.png"

video_name = "xx.mp4"

print(guess_type(image_name)[0])
print(guess_type(video_name)[0])