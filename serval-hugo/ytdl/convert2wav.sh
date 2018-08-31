#!/bin/bash
# A simple script to convert youtube downloads to 16kHz signed 16bit wav files

# count files
#ls -1 *.wav | wc -l

# segment  to process
#segment="balanced_train" 
segment="unbalanced_train" 
#segment="eval" 

# direcories
source_dir="../../../data/audio/serval-data/raw/youtube-downloads/$segment/"
target_dir="../../../data/audio/serval-data/wav-16bit-16khz/ytdl/$segment/"

echo "Start conversion from " $source_dir " to " $target_dir

## ffmpeg op de commandline
# option -n only copies if target does not exist
#for vid in $source_dir*.m4a; do ffmpeg -i "$vid" -vn -acodec pcm_s16le -ar 16000 "$target_dir${vid%.m4a}.wav"; done

for vid in $source_dir*.m4a; 
do 
fid=$(basename "$vid" '.m4a')
#echo $fid;
#echo "$target_dir$fid.wav"; 
ffmpeg -loglevel panic -i "$vid" -vn -n -hide_banner -acodec pcm_s16le -ar 16000 "$target_dir$fid.wav"
done

for vid in $source_dir*.opus; 
do 
fid=$(basename "$vid" '.opus')
#echo $fid;
#echo "$target_dir$fid.wav"; 
ffmpeg -loglevel panic -i "$vid" -vn -n -hide_banner -acodec pcm_s16le -ar 16000 "$target_dir$fid.wav"
done

for vid in $source_dir*.ogg; 
do 
fid=$(basename "$vid" '.ogg')
#echo $fid;
#echo "$target_dir$fid.wav"; 
ffmpeg -loglevel panic -i "$vid" -vn -n -hide_banner -acodec pcm_s16le -ar 16000 "$target_dir$fid.wav"
done


#for vid in *.opus; do ffmpeg -i "$vid" -vn -acodec pcm_s16le -ar 16000 "${vid%.opus}.wav"; done
#for vid in *.ogg; do ffmpeg -i "$vid" -vn -acodec pcm_s16le -ar 16000 "wav/${vid%.ogg}.wav"; done

echo "Finished conversion succesfully!"
