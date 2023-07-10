# 1. Create environment using environment.yaml
  conda env create -f environment.yaml
# 2. Resample your audio .wav files to 16KHz by using resample.py
# 3. Give appropriate path of your data and your pretrained model path and where you want to store finetuned model as finetuned path
# 4. Run main.py
Most important
1.be aware of CUDA Error which can resolved by batch size and no gpu should be run at that time
2. Hugging Face data size create No space on the disk if then I have already wrriten that in code
# wav2vec2-emotion-11
# wav2vec2-LID
