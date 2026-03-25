#!/bin/sh

# csv→jsonl
python 01_csv2jsonl.py



# echo -----------------------------------
# echo START FINE-TUNING rinna-gpt2-medium
# echo -----------------------------------
# python 02_rinna-gpt2-medium_10.py 2>&1 |tee "logs/02_rinna-gpt2-medium_10_$(date +%Y%m%d_%H%M%S).log"
# python 03_rinna-gpt2-medium_25.py 2>&1 |tee "logs/03_rinna-gpt2-medium_25_$(date +%Y%m%d_%H%M%S).log"
# python 04_rinna-gpt2-medium_50.py 2>&1 |tee "logs/04_rinna-gpt2-medium_50_$(date +%Y%m%d_%H%M%S).log"
# python 05_rinna-gpt2-medium_75.py 2>&1 |tee "logs/05_rinna-gpt2-medium_75_$(date +%Y%m%d_%H%M%S).log"
# python 06_rinna-gpt2-medium_100.py 2>&1 |tee "logs/06_rinna-gpt2-medium_100_$(date +%Y%m%d_%H%M%S).log"



# echo -------------------------------
# echo START FINE-TUNING rinna-gpt2-1b
# echo -------------------------------
# python 08_rinna-gpt2-1b_10.py 2>&1 |tee "logs/08_rinna-gpt2-1b_10_$(date +%Y%m%d_%H%M%S).log"
# python 09_rinna-gpt2-1b_25.py 2>&1 |tee "logs/09_rinna-gpt2-1b_25_$(date +%Y%m%d_%H%M%S).log"
# python 10_rinna-gpt2-1b_50.py 2>&1 |tee "logs/10_rinna-gpt2-1b_50_$(date +%Y%m%d_%H%M%S).log"
# python 11_rinna-gpt2-1b_75.py 2>&1 |tee "logs/11_rinna-gpt2-1b_75_$(date +%Y%m%d_%H%M%S).log"
# python 12_rinna-gpt2-1b_100.py 2>&1 |tee "logs/12_rinna-gpt2-1b_100_$(date +%Y%m%d_%H%M%S).log"
# python 13_rinna-gpt2-1b_270.py 2>&1 |tee "logs/13_rinna-gpt2-1b_270_$(date +%Y%m%d_%H%M%S).log"



# echo ----------------------------------
# echo START FINE-TUNING rinna-neox-small
# echo ----------------------------------
# python 14_rinna-gpt-neox-small_10.py 2>&1 |tee "logs/14_rinna-gpt-neox-small_10_$(date +%Y%m%d_%H%M%S).log"
# python 15_rinna-gpt-neox-small_25.py 2>&1 |tee "logs/15_rinna-gpt-neox-small_25_$(date +%Y%m%d_%H%M%S).log"
# python 16_rinna-gpt-neox-small_50.py 2>&1 |tee "logs/16_rinna-gpt-neox-small_50_$(date +%Y%m%d_%H%M%S).log"
# python 17_rinna-gpt-neox-small_75.py 2>&1 |tee "logs/17_rinna-gpt-neox-small_75_$(date +%Y%m%d_%H%M%S).log"
# python 18_rinna-gpt-neox-small_100.py 2>&1 |tee "logs/18_rinna-gpt-neox-small_100_$(date +%Y%m%d_%H%M%S).log"




# echo ------------------------------------
# echo START FINE-TUNING rinna-neox-3.6b
# echo ------------------------------------
# python 20_rinna-gpt-neox_10.py 2>&1 |tee "logs/20_rinna-gpt-neox_10_$(date +%Y%m%d_%H%M%S).log"
# python 21_rinna-gpt-neox_25.py 2>&1 |tee "logs/21_rinna-gpt-neox_25_$(date +%Y%m%d_%H%M%S).log"
# python 22_rinna-gpt-neox_50.py 2>&1 |tee "logs/22_rinna-gpt-neox_50_$(date +%Y%m%d_%H%M%S).log"
# python 23_rinna-gpt-neox_75.py 2>&1 |tee "logs/23_rinna-gpt-neox_75_$(date +%Y%m%d_%H%M%S).log"
# python 24_rinna-gpt-neox_100.py 2>&1 |tee "logs/24_rinna-gpt-neox_100_$(date +%Y%m%d_%H%M%S).log"



echo ------------------------------------
echo START FINE-TUNING open-calm-medium
echo ------------------------------------
python 25_open-calm-medium_10.py 2>&1 |tee "logs/25_open-calm-medium_10_$(date +%Y%m%d_%H%M%S).log"
python 26_open-calm-medium_25.py 2>&1 |tee "logs/26_open-calm-medium_25_$(date +%Y%m%d_%H%M%S).log"
python 27_open-calm-medium_50.py 2>&1 |tee "logs/27_open-calm-medium_50_$(date +%Y%m%d_%H%M%S).log"
python 28_open-calm-medium_75.py 2>&1 |tee "logs/28_open-calm-medium_75_$(date +%Y%m%d_%H%M%S).log"
python 29_open-calm-medium_100.py 2>&1 |tee "logs/29_open-calm-medium_100_$(date +%Y%m%d_%H%M%S).log"



echo ------------------------------------
echo START FINE-TUNING open-calm-large
echo ------------------------------------
python 30_open-calm-large_10.py 2>&1 |tee "logs/30_open-calm-large_10_$(date +%Y%m%d_%H%M%S).log"
python 31_open-calm-large_25.py 2>&1 |tee "logs/31_open-calm-large_125$(date +%Y%m%d_%H%M%S).log"
python 32_open-calm-large_50.py 2>&1 |tee "logs/32_open-calm-large_50_$(date +%Y%m%d_%H%M%S).log"
python 33_open-calm-large_75.py 2>&1 |tee "logs/33_open-calm-large_75_$(date +%Y%m%d_%H%M%S).log"
python 34_open-calm-large_100.py 2>&1 |tee "logs/34_open-calm-large_100_$(date +%Y%m%d_%H%M%S).log"



# echo ------------------------------------
# echo START FINE-TUNING llm-jp-3-440m
# echo ------------------------------------
# python 35_llm-jp-3-440m_10.py 2>&1 |tee "logs/35_llm-jp-3-440m_10_$(date +%Y%m%d_%H%M%S).log"
# python 36_llm-jp-3-440m_25.py 2>&1 |tee "logs/36_llm-jp-3-440m_25_$(date +%Y%m%d_%H%M%S).log"
# python 37_llm-jp-3-440m_50.py 2>&1 |tee "logs/37_llm-jp-3-440m_50_$(date +%Y%m%d_%H%M%S).log"
# python 38_llm-jp-3-440m_75.py 2>&1 |tee "logs/38_llm-jp-3-440m_75_$(date +%Y%m%d_%H%M%S).log"
# python 39_llm-jp-3-440m_100.py 2>&1 |tee "logs/39_llm-jp-3-440m_100_$(date +%Y%m%d_%H%M%S).log"



# echo ------------------------------------
# echo START FINE-TUNING 8x152M
# echo ------------------------------------
# python 40_NU-8x152M_10.py 2>&1 |tee "logs/40_NU-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
# python 41_NU-8x152M_25.py 2>&1 |tee "logs/41_NU-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
# python 42_NU-8x152M_50.py 2>&1 |tee "logs/42_NU-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
# python 43_NU-8x152M_75.py 2>&1 |tee "logs/43_NU-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
# python 44_NU-8x152M_100.py 2>&1 |tee "logs/44_NU-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
# python 40_DU-0.5-8x152M_10.py 2>&1 |tee "logs/40_DU-0.5-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
# python 41_DU-0.5-8x152M_25.py 2>&1 |tee "logs/41_DU-0.5-8x152M_25_$(date +%Y%m%d_%H%M%S).log"
# python 42_DU-0.5-8x152M_50.py 2>&1 |tee "logs/42_DU-0.5-8x152M_50_$(date +%Y%m%d_%H%M%S).log"
# python 43_DU-0.5-8x152M_75.py 2>&1 |tee "logs/43_DU-0.5-8x152M_75_$(date +%Y%m%d_%H%M%S).log"
# python 44_DU-0.5-8x152M_100.py 2>&1 |tee "logs/44_DU-0.5-8x152M_100_$(date +%Y%m%d_%H%M%S).log"
# python 41_NU-8x152M_10.py 2>&1 |tee "logs/41_NU-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
# python 42_NU-8x152M_10.py 2>&1 |tee "logs/42_NU-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
# python 43_DU-0.5-8x152M_10.py 2>&1 |tee "logs/42_DU-0.5-8x152M_10_$(date +%Y%m%d_%H%M%S).log"
# python 44_BTX-8x152M_10.py 2>&1 |tee "logs/45_VTX-8x152M_10_$(date +%Y%m%d_%H%M%S).log"



# echo ------------------------------------
# echo START FINE-TUNING Llama
# echo ------------------------------------
# python 45_llama-3.2-1b_10.py 2>&1 |tee "logs/45_Llama-3.2-1b_10_$(date +%Y%m%d_%H%M%S).log"
# python 46_llama-3.2-1b_25.py 2>&1 |tee "logs/46_Llama-3.2-1b_25_$(date +%Y%m%d_%H%M%S).log"
