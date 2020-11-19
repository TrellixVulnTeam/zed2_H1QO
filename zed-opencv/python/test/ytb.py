from pytube import YouTube

source = YouTube('https://www.youtube.com/watch?v=LVzw0hfH6Ao')


en_caption = source.captions.get_by_language_code('jp')

en_caption_convert_to_srt =(en_caption.generate_srt_captions())

print(en_caption_convert_to_srt)
#save the caption to a file named Output.txt
text_file = open("Output.txt", "w")
text_file.write(en_caption_convert_to_srt)
text_file.close()