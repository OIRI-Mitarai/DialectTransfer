#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
export TF_CPP_MIN_LOG_LEVEL=3

wav="test.wav"
echo 似顔絵描いちゃるけん、ちょいけえ
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_001.wav"
echo 今日はしんけん寒いなあ
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_002.wav"
echo 今日は寒いけん、外に出られんわ
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_003.wav"
echo 明日は雨が降るっち思うわ
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_004.wav"
echo もうご飯食べよんの？
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_005.wav"
echo どこ行きよんの？
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_006.wav"
echo なんしちょん？
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_007.wav"
echo こんちはー。元気しちょった？
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_008.wav"
echo ほら行くで！
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_009.wav"
echo あんたも来よな
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


wav="./data/audio_010.wav"
echo じゃあまたなあ
python 21_test_sample_finetune.py $wav
python 22_test_sample_origin.py $wav
echo


# wav="./data/audio_00.wav"
# echo
# python 21_test_sample_finetune.py $wav
# python 22_test_sample_origin.py $wav
#
#
# wav="./data/audio_00.wav"
# echo
# python 21_test_sample_finetune.py $wav
# python 22_test_sample_origin.py $wav
#
#
# wav="./data/audio_00.wav"
# echo
# python 21_test_sample_finetune.py $wav
# python 22_test_sample_origin.py $wav
#
#
# wav="./data/audio_00.wav"
# echo
# python 21_test_sample_finetune.py $wav
# python 22_test_sample_origin.py $wav
#
#
# wav="./data/audio_00.wav"
# echo
# python 21_test_sample_finetune.py $wav
# python 22_test_sample_origin.py $wav
