for i in $(seq -f "%03g" 2 22); do
    ffmpeg -i "data/audio_${i}.m4a" "audio_${i}.wav"
done
