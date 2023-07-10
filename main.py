from wav2vec2_classifier import Wav2vec2Classifier
from statistics import mean
import os
import shutil
#Use any free gpu if cuda :1 is available then do this
gpu_indices = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpu_indices)
# Instantiate the Wav2vec2Classifier class
wav2vec2classifier = Wav2vec2Classifier()



# Set the path to the data directory
path_to_data = "/data/ganji_sreeram/Interns/Vipendra:Emotion_Recognition/Wav2vec2_Emotion/Resampled-11-Emotion-Data"
# Set the path to the CSV files
path_to_exp_files = "Exps"
# Set the path to the pretrained model
pretrained_path = "/data/ganji_sreeram/Interns/Vipendra:Emotion_Recognition/Wav2vec2_Emotion/pretrained-model/wav2vec2-large-xlsr-53-greek"


#train
test_accuracy=0
for i in range(1,4):
    #path to each exp files in which you want to store each exps result
    path_to_each_exp = path_to_exp_files+f"/Exp-{i}"
    #making directory dor each exp
    os.makedirs(path_to_each_exp, exist_ok=True)
    #checkpoints directory is where we are storing checkpoints inside each exp file
    checkpoints_dir = path_to_each_exp+"/Checkpoints"
    #storing each exp csv files inside each exp file
    path_to_exp_csv_files = path_to_each_exp+f"/CSV-{i}"
    #making directory for CSV files inside each exp
    os.makedirs(path_to_exp_csv_files, exist_ok=True)
    #returning path to train and test csv file as a list=[train,test] 
    path_to_csv = wav2vec2classifier.data_preparation(path_to_data,path_to_exp_csv_files,i,16000)
    #test csv files are stored inside each 
    path_to_test_csv = f"{path_to_exp_csv_files}/test.csv"
    path_to_train_csv = f"{path_to_exp_csv_files}/train.csv"
    path_to_store_finetuned = path_to_exp_files+f"/Exp-{i}/finetuned"
    os.makedirs(path_to_store_finetuned, exist_ok=True)
    try :
        finetuned_path = wav2vec2classifier.train_function(path_to_exp_csv_files,pretrained_path,path_to_store_finetuned,checkpoints_dir)
    except:
        shutil.rmtree("/data/opt/dotfiles/.cache/huggingface", ignore_errors=False)
        finetuned_path = wav2vec2classifier.train_function(path_to_exp_csv_files,pretrained_path,path_to_store_finetuned,checkpoints_dir)
    new_accuracy = wav2vec2classifier.test_function(path_to_test_csv,finetuned_path)
    test_accuracy+=new_accuracy
    file1 = open("result.txt","a")
    file1.write(f"{new_accuracy*100}+%")
#print the test accuracy for 3 validation
print((test_accuracy)*100)
#To predict accuracy of a particular file or to get its class below line is used
# emotion.predict_function(path_to_particular_file,path_to_particular_finetuned_path)

