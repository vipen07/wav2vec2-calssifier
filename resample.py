import os
import torchaudio
import torchaudio.transforms as transforms
import librosa, soundfile

directory_path = '/data/ganji_sreeram/Interns/Vipendra:Emotion_Recognition/Wav2vec2_Emotion/Resampled-12-Language-Data'
print(directory_path)
# Iterate over all folders in the directory
for folder_name in os.listdir(directory_path):
    folder_path = os.path.join(directory_path, folder_name)
   
    if os.path.isdir(folder_path):
        # Do something with the folder_path
        input_folder = folder_path
        category = input_folder.split('/')[-1]
        print(category)
        output_folder = f"/data/ganji_sreeram/Interns/Vipendra:Emotion_Recognition/Wav2vec2_Emotion/Language-Data/{category}"
        # target_sampling_rate = 16000  # Desired sampling rate

        # Create the output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Get the list of audio files in the input folder
        audio_files = os.listdir(input_folder)
        audios_files=[]
        count = 0
        for f in audio_files:
            if (f.endswith(".wav") and count<=5000):
                count+=1
                audios_files.append(f)
            else:
                break
                 # Filter only WAV files if needed
        print(count)
        # Resample and save each audio file
        for audio_file in audios_files:
            # Load the audio file
           
            file_path = os.path.join(input_folder, audio_file)
            try:
                s, fs =librosa.load(file_path,sr=16000, mono=True)
                # Generate the output file path
                output_file_path = os.path.join(output_folder, audio_file)
                # Save the resampled audio to the output file path
                soundfile.write(output_file_path, s, 16000)
                
            except:
                None

    
