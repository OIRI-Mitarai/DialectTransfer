#!/bin/sh

# csv→jsonl
# python 01_csv2jsonl.py



# echo -----------------------------------
# echo START TESTING rinna-gpt2-medium
# echo -----------------------------------
# python 02_rinna-gpt2-medium_10.py 2>&1 |tee "logs/02_rinna-gpt2-medium_10_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 03_rinna-gpt2-medium_25.py 2>&1 |tee "logs/03_rinna-gpt2-medium_25_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 04_rinna-gpt2-medium_50.py 2>&1 |tee "logs/04_rinna-gpt2-medium_50_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 05_rinna-gpt2-medium_75.py 2>&1 |tee "logs/05_rinna-gpt2-medium_75_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 06_rinna-gpt2-medium_100.py 2>&1 |tee "logs/06_rinna-gpt2-medium_100_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python




# echo ------------------------------------
# echo START TESTING open-calm-medium
# echo ------------------------------------
# python 25_open-calm-medium_10.py 2>&1 |tee "logs/25_open-calm-medium_10_$(date +%Y%m%d_%H%M%S).log"
# python 26_open-calm-medium_25.py 2>&1 |tee "logs/26_open-calm-medium_25_$(date +%Y%m%d_%H%M%S).log"
# python 27_open-calm-medium_50.py 2>&1 |tee "logs/27_open-calm-medium_50_$(date +%Y%m%d_%H%M%S).log"
# python 28_open-calm-medium_75.py 2>&1 |tee "logs/28_open-calm-medium_75_$(date +%Y%m%d_%H%M%S).log"
# python 29_open-calm-medium_100.py 2>&1 |tee "logs/29_open-calm-medium_100_$(date +%Y%m%d_%H%M%S).log"



# echo ------------------------------------
# echo START TESTING open-calm-large
# echo ------------------------------------
# python 30_open-calm-large_10.py 2>&1 |tee "logs/30_open-calm-large_10_$(date +%Y%m%d_%H%M%S).log"
# python 31_open-calm-large_25.py 2>&1 |tee "logs/31_open-calm-large_25$(date +%Y%m%d_%H%M%S).log"
# python 32_open-calm-large_50.py 2>&1 |tee "logs/32_open-calm-large_50_$(date +%Y%m%d_%H%M%S).log"
# python 33_open-calm-large_75.py 2>&1 |tee "logs/33_open-calm-large_75_$(date +%Y%m%d_%H%M%S).log"
# python 34_open-calm-large_100.py 2>&1 |tee "logs/34_open-calm-large_100_$(date +%Y%m%d_%H%M%S).log"



# echo ------------------------------------
# echo START TESTING llm-jp-3-440m
# echo ------------------------------------
# python 35_llm-jp-3-440m_10.py 2>&1 |tee "logs/35_llm-jp-3-440m_10_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 36_llm-jp-3-440m_25.py 2>&1 |tee "logs/36_llm-jp-3-440m_25_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 37_llm-jp-3-440m_50.py 2>&1 |tee "logs/37_llm-jp-3-440m_50_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 38_llm-jp-3-440m_75.py 2>&1 |tee "logs/38_llm-jp-3-440m_75_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python
# python 39_llm-jp-3-440m_100.py 2>&1 |tee "logs/39_llm-jp-3-440m_100_$(date +%Y%m%d_%H%M%S).log"
# sleep 2
# pkill -u $USER -f python



echo ------------------------------------
echo START TESTING 8x152M
echo ------------------------------------
python 40_DU-0.5-8x152M_10.py 2>&1 |tee "logs/40_DU-0.5-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 41_DU-0.5-8x152M_25.py 2>&1 |tee "logs/41_DU-0.5-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 42_DU-0.5-8x152M_50.py 2>&1 |tee "logs/42_DU-0.5-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 43_DU-0.5-8x152M_75.py 2>&1 |tee "logs/43_DU-0.5-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 44_DU-0.5-8x152M_100.py 2>&1 |tee "logs/44_DU-0.5-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 45_DU-1-8x152M_10.py 2>&1 |tee "logs/45_DU-1-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 46_DU-1-8x152M_25.py 2>&1 |tee "logs/46_DU-1-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 47_DU-1-8x152M_50.py 2>&1 |tee "logs/47_DU-1-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 48_DU-1-8x152M_75.py 2>&1 |tee "logs/48_DU-1-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 49_DU-1-8x152M_100.py 2>&1 |tee "logs/49_DU-1-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 50_NU-8x152M_10.py 2>&1 |tee "logs/50_NU-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 51_NU-8x152M_25.py 2>&1 |tee "logs/51_NU-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 52_NU-8x152M_50.py 2>&1 |tee "logs/52_NU-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 53_NU-8x152M_75.py 2>&1 |tee "logs/53_NU-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 54_NU-8x152M_100.py 2>&1 |tee "logs/54_NU-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 55_RNU-0.5-8x152M_10.py 2>&1 |tee "logs/55_RNU-0.5-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 56_RNU-0.5-8x152M_25.py 2>&1 |tee "logs/56_RNU-0.5-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 57_RNU-0.5-8x152M_50.py 2>&1 |tee "logs/57_RNU-0.5-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 58_RNU-0.5-8x152M_75.py 2>&1 |tee "logs/58_RNU-0.5-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
python 59_RNU-0.5-8x152M_100.py 2>&1 |tee "logs/59_RNU-0.5-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
sleep 2
pkill -u $USER -f python
