#importing required libraries
from signal import signal, SIGPIPE, SIG_DFL  
signal(SIGPIPE,SIG_DFL) 
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pathlib import Path
from tqdm import tqdm
import torchaudio
from sklearn.model_selection import train_test_split, StratifiedKFold
import transformers
from transformers import AutoConfig, Wav2Vec2Processor, EvalPrediction, Wav2Vec2FeatureExtractor, AutoTokenizer, Trainer, TrainingArguments, is_apex_available
import os
from sklearn.metrics import accuracy_score
import sys
import librosa
import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from datasets import load_dataset, load_metric
from dataclasses import dataclass
from typing import Optional, Tuple, Union, Dict, Any, List
from transformers.file_utils import ModelOutput
import torch.nn.functional as F
from packaging import version
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)
from accelerate import Accelerator

if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) # Creates a linear layer with input size and output size both equal to config.hidden_size.
        self.dropout = nn.Dropout(config.final_dropout) # Creates a dropout layer with dropout probability specified by config.final_dropout.This usaully to prevent netwok from overfitting
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels) # Creates a linear layer with input size equal to config.hidden_size and output size equal to config.num_labels.


    def forward(self, features, **kwargs):
        x = features  # Assign the input features to variable x.
        x = self.dropout(x)  # Apply dropout to the input features.
        x = self.dense(x)  # Pass the features through the linear layer.
        x = torch.tanh(x)  # Apply the hyperbolic tangent activation function.
        x = self.dropout(x)  # Apply dropout to the output of the activation function.
        x = self.out_proj(x)  # Pass the features through the final linear layer.
        return x  # Return the output.

class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)  # Call the constructor of the parent class
        self.num_labels = config.num_labels  # Number of output labels
        self.pooling_mode = config.pooling_mode  # Pooling mode for feature aggregation
        self.config = config  # Store the configuration object
        self.wav2vec2 = Wav2Vec2Model(config)  # Wav2Vec2 model for feature extraction
        self.classifier = Wav2Vec2ClassificationHead(config)  # Classification head for final predictions
        self.init_weights()  # Initialize the model weights


    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()
         # Freeze the parameters of the feature extractor


    def merged_strategy(self, hidden_states, mode="mean"):
        if mode == "mean":
            # Compute the mean along the time dimension
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            # Compute the sum along the time dimension
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            # Compute the maximum along the time dimension
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            # Invalid pooling mode
            raise Exception("The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")
        return outputs

    def forward(
        self,
        input_values,  # Input values or sequences to be processed by the model.
        attention_mask=None,  # Optional attention mask for masking certain elements of the input.
        output_attentions=None,  # Flag indicating whether to output attention weights.
        output_hidden_states=None,  # Flag indicating whether to output hidden states.
        return_dict=None,  # Flag indicating whether to return the output as a dictionary.
        labels=None,  # Optional labels for the input sequences.
    ):
        # Set the value of return_dict to the provided value if not None, otherwise use the value from the model's configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass the input values and other optional arguments to the wav2vec2 model
        # Store the outputs (hidden states, attentions, etc.) in the 'outputs' variable
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # Extract the hidden states from the outputs
        hidden_states = outputs[0]

        # Apply the pooling strategy to the hidden states
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)

        # Pass the pooled hidden states to the classifier
        logits = self.classifier(hidden_states) 
        #logits represents the unnormalized predictions produced by the classifier based on the hidden states of the input


        # Initialize the loss variable
        loss = None

        # Calculate the loss if labels are provided
        if labels is not None:
            # Determine the problem type based on the number of labels and label data type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # Compute the loss based on the problem type
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # Prepare the output based on the return_dict flag
        if not return_dict:
            # Create the output tuple without loss
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output
        else:
            # Create a SpeechClassifierOutput object with the relevant outputs and loss
            return SpeechClassifierOutput(
                loss=loss,
                logits=logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None


    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]
        d_type = torch.long if isinstance(label_features[0], int) else torch.float
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        batch["labels"] = torch.tensor(label_features, dtype=d_type)
        return batch
    

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """
        model.cuda().train()
        inputs = self._prepare_inputs(inputs)
        if self.use_cuda_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        if self.use_cuda_amp:
            self.scaler.scale(loss.mean()).backward()
        else:
            self.accelerator.backward(loss.mean())
        return loss.mean().detach()

'''
    Here we have defined 5 function that are used in our main class of Labels 
    which are as follows:
'''

#compute metrices

def compute_metrics(p: EvalPrediction, is_regression=False):
    # Extract the predictions from the evaluation prediction object
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    # Squeeze the predictions if it is a regression problem, else find the indices of the highest probabilities
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
    if is_regression:
        # Compute Mean Squared Error (MSE) for regression problems
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        # Compute accuracy for classification problems
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# Convert a label to its corresponding id/index in the label list
def label_to_id(label, label_list):
    #print(len(label_list))
     # If the label list is not empty
    if len(label_list) > 0:
       # Return the index of the label in the label list if found, else return -1
        return label_list.index(label) if label in label_list else -1
    # Return the label itself if the label list is empty
    return label

    
# Load a speech file and convert it to a NumPy array
def speech_file_to_array_fn(path, sampling_rate):
    # Load the speech file using librosa and get the original sampling rate
    speech, fs = librosa.load(path, sr=sampling_rate, mono=True)
    # Return the speech array
    return speech


# Load a speech file and convert it to an array
def speech_file_to_array(path):
    # Load the speech file using librosa and set the target sampling rate and mono option
    speech, fs = librosa.load(path, sr=16000, mono=True)
    # Return the speech array
    return speech
    
# Preprocess the examples for training
def preprocess_function(examples, target_sampling_rate=16000, input_column="path", output_column="labels_classes"):
    # Convert the speech file paths to arrays using the defined function
    speech_list = [speech_file_to_array(path) for path in examples[input_column]]
    # Convert the labels to their corresponding ids using the defined function
    target_list = [label_to_id(label, label_list) for label in examples[output_column]]
    try:
        # Process the speech arrays using the defined processor for training data
        result = processor_train(speech_list, sampling_rate=target_sampling_rate)
        # Add the target labels to the result dictionary
        result["labels"] = list(target_list)
        # Return the preprocessed data
        return result
    except Exception as e:
        a = [[]]
        for i in range(0, 11):
            a.append([0.000000000, 0])
            # Create a dummy output in case of an exception
        return a
        # Return the dummy output


def predict_file(path, sampling_rate, device, config, model, feature_extractor):
    # Predict the label from a single speech file
    speech = speech_file_to_array_fn(path, sampling_rate)
    # Convert the speech file to an array using the defined function
    inputs = feature_extractor(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    # Extract the features from the speech array using the feature extractor
    inputs = {key: inputs[key].to(device) for key in inputs}
    # Move the inputs to the specified device
    with torch.no_grad():
        logits = model(**inputs).logits
        # Pass the inputs through the model to obtain the logits
        scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
        # Compute the softmax probabilities from the logits and convert them to a NumPy array
        outputs = [
            {"Labels_classes": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)
        ]
        # Create a list of dictionaries containing the predicted label and corresponding score
    return outputs
    # Return the predicted outputs
def predict_test(path, sampling_rate, device, config, model):
    # Predict the label from a single speech file (testing mode)
    speech = speech_file_to_array_fn(path, sampling_rate)
    # Convert the speech file to an array using the defined function
    features = processor_test(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    # Process the speech array using the defined processor for testing data
    input_values = features.input_values.to(device)
    attention_mask = features.attention_mask.to(device)
    # Move the input values and attention mask to the specified device
    inputs = feature_extractor_test(speech, sampling_rate=sampling_rate, return_tensors="pt", padding=True)
    # Extract the features from the speech array using the feature extractor
    inputs = {key: inputs[key].to(device) for key in inputs}
    # Move the inputs to the specified device
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        logits = model(**inputs).logits
        # Pass the inputs through the model to obtain the logits
    scores = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    # Compute the softmax probabilities from the logits and convert them to a NumPy array
    outputs = [
        {"Labels_classes": config.id2label[i], "Score": f"{round(score * 100, 3):.1f}%"} for i, score in enumerate(scores)
    ]
    # Create a list of dictionaries containing the predicted labels and corresponding score
    return outputs
    # Return the predicted outputs


def speech_file_to_fn(batch):
    
    speech_array,sampling_rate = librosa.load(batch["path"],sr = processor_predict.feature_extractor.sampling_rate)
    batch["speech"] = speech_array
    return batch


def predict(batch):
    features = processor_predict(batch["speech"], sampling_rate=processor_predict.feature_extractor.sampling_rate, return_tensors="pt", padding=True)

    input_values = features.input_values.to(predict_device)
    attention_mask = features.attention_mask.to(predict_device)

    with torch.no_grad():
        logits = model_predict(input_values, attention_mask=attention_mask).logits 

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


''' This class is our main class which we will use and import it in our main.py file
        This class contains majorly three Functions :
            1. data_preparation
            2. train_function
            3. test_function: 
            4. prediction_function:
            
'''
class Wav2vec2Classifier(DataCollatorCTCWithPadding,SpeechClassifierOutput,Wav2Vec2ClassificationHead,Wav2Vec2ForSpeechClassification,CTCTrainer):
    def __init__(self):
        pass
    ''' data_preparation funtion:
        In which we are giving audio data in wav files and then we 
        process these files convert them into target sampling rate 
        and convert them into train.csv and test.csv'''
    def data_preparation(self,path_to_data,path_to_csv_files,x,fs=16000):
        data = []
        for path in tqdm(Path(path_to_data).glob("**/*.wav")):
            name = str(path).split('/')[-1].split('.')[0]
            label = str(path).split('.')[-2].split('/')[-2]
            try:
                # There are some broken files
                s = torchaudio.load(path)
                data.append({
                    "name": name,
                    "path": path,
                    "labels_classes": label
                })
            except Exception as e:

                print(str(path), e)
                pass

            # break
        df = pd.DataFrame(data)
        # Add a new column "status" to the DataFrame indicating if the path exists or not
        df["status"] = df["path"].apply(lambda path: True if os.path.exists(path) else None)
        # Drop rows where the "path" column is missing or invalid
        df = df.dropna(subset=["path"])
        # Drop the "status" column from the DataFrame
        df = df.drop(labels="status", axis=1)
        # Print the length of the DataFrame after the initial filtering step
        print(f"Step 1: {len(df)}")
        # Shuffle the DataFrame randomly
        df = df.sample(frac=1)
        # Reset the index of the DataFrame after shuffling
        df = df.reset_index(drop=True)
        df.head()
        # Print the unique labels in the "labels_classes" column of the DataFrame
        print("Labels: ", df["labels_classes"].unique())
        # Count the number of paths for each labels category
        df.groupby("labels_classes").count()[["path"]]
        save_path = path_to_csv_files
        # Split the DataFrame into train and test sets
        train_df, test_df = train_test_split(df, test_size=0.17, random_state=(int)(101/x), stratify=df["labels_classes"])
        # Reset the index of the train and test DataFrames
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        # Save the train DataFrame to a CSV file
        train_df.to_csv(f"{save_path}/train.csv", sep="\t", encoding="utf-8", index=False)
        # Save the test DataFrame to a CSV file
        test_df.to_csv(f"{save_path}/test.csv", sep="\t", encoding="utf-8", index=False)
        # Print the shape of the train and test DataFrames
        print(f"train.csv contains {train_df.shape}")
        print(f"test.csv contains {test_df.shape}")
        # Return the paths of the train.csv and test.csv files
        return [f"{save_path}/train.csv", f"{save_path}/test.csv"]


    '''train_function:
        In this we are taking these as train dataset and test dataset
        mapping them to there corresponding speech array 
        and then using CTCTrainer we are traing the model with given training arguments
    '''


    def train_function(self,path_to_csv_files,pretrained_path,path_to_store_finetuned,output_dir):
        # Define the data file paths
        
        data_files = {
            "train": f"{path_to_csv_files}/train.csv",
            "validation": f"{path_to_csv_files}/test.csv",
        }
        # Load dataset using the dataset library
        dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
        # Separate train and validation datasets
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"]
        # Specify input and output columns
        input_column = "path"
        output_column = "labels_classes"
        # Get unique labels from the train_dataset and set them as the output column
        global label_list
        label_list = train_dataset.unique(output_column)
        label_list.sort()  # Sort for determinism
        num_labels = len(label_list)
        print(f"A classification problem with {num_labels} classes: {label_list}")
        # Specify the model name or path and pooling mode
        model_name_or_path = pretrained_path
        pooling_mode = "mean"
        # Configure model and set pooling mode
        config = AutoConfig.from_pretrained(
            model_name_or_path,
            num_labels=num_labels,
            label2id={label: i for i, label in enumerate(label_list)},
            id2label={i: label for i, label in enumerate(label_list)},
            finetuning_task="wav2vec2_clf",
        )
        setattr(config, 'pooling_mode', pooling_mode)
        # Initialize the Wav2Vec2 processor for training
        global processor_train
        processor_train = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        target_sampling_rate = processor_train.feature_extractor.sampling_rate
        print(f"The target sampling rate: {target_sampling_rate}")
        # Preprocess train and validation datasets
        train_dataset = train_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4,   

        )
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batch_size=100,
            batched=True,
            num_proc=4,
           
        )
        # Data collator for CTC training with padding
        data_collator = DataCollatorCTCWithPadding(processor=processor_train, padding=True)
        is_regression = False
        # Initialize the model for speech classification
        model = Wav2Vec2ForSpeechClassification.from_pretrained(
            model_name_or_path,
            config=config,
        )
        model = model.cuda()
        model.freeze_feature_extractor()
        # Specify the training arguments
        training_args = TrainingArguments(
            auto_find_batch_size=True,
            output_dir=output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            evaluation_strategy="steps",
            max_steps=1000,
            fp16=True,
            save_steps=60,
            eval_steps=60,
            logging_steps=60,
            learning_rate=0.0001,
            save_total_limit=2,
        )
        # Initialize the CTCTrainer for training
        trainer = CTCTrainer(
            model=model.cuda(),
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=processor_train.feature_extractor,
        )
        # Train the model
        trainer.train()
        # Save the fine-tuned model and tokenizer
        from transformers import AutoTokenizer
        trainer.save_model(path_to_store_finetuned)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        tokenizer.save_pretrained(path_to_store_finetuned)
        # Return the path of the fine-tuned model
        return path_to_store_finetuned




    # Define the test function
    def test_function(self, path_to_test, finetuned_path):
        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Specify the model name or path
        model_name_or_path = finetuned_path
        # Load the configuration
        config = AutoConfig.from_pretrained(model_name_or_path)
        # Initialize the feature extractor for testing
        global feature_extractor_test
        feature_extractor_test = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        sampling_rate = feature_extractor_test.sampling_rate
        # Load the model for speech classification
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
        # Initialize the processor for testing
        global processor_test
        processor_test = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        # Read the test data from the file
        test = pd.read_csv(path_to_test, sep="\t")
        test.head()
        # Get the path and label from the first row of the test data
        df_row = test.iloc[2]
        print(df_row)
        path, labels_classes = df_row["path"], df_row["labels_classes"]
        # Create a DataFrame for the test output
        df = pd.DataFrame([{"Labels_classes": labels_classes, "Sentence": "    "}])
        # Load the speech data and preprocess it
        speech, sr = torchaudio.load(path)
        speech = speech[0].numpy().squeeze()
        speech = librosa.resample(np.asarray(speech), orig_sr=sr, target_sr=sampling_rate)
        # Make predictions on the test data
        outputs = predict_test(path, sampling_rate, device, config,model)
        # Create a DataFrame for the predictions
        r = pd.DataFrame(outputs)
        # Print the percentage of corresponding label, the outputs, the label and the model
        print("Percentage of corresponding label_class:", outputs)
        print("Labels_classes :", labels_classes)
        test_dataset = load_dataset("csv", data_files={"test": path_to_test}, delimiter="\t")["test"]
        test_dataset
        global predict_device
        predict_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
       
        config = AutoConfig.from_pretrained(model_name_or_path)
        global processor_predict
        processor_predict = Wav2Vec2Processor.from_pretrained(model_name_or_path)
        global model_predict
        model_predict = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
        test_dataset = test_dataset.map(speech_file_to_fn)
        result = test_dataset.map(predict, batched=True, batch_size=11)
        label_names = [config.id2label[i] for i in range(config.num_labels)]
        label_names
        print(label_names)
        y_true = [config.label2id[name] for name in result["labels_classes"]]
        y_pred = result["predicted"]
        print(classification_report(y_true, y_pred, target_names=label_names))
        a = list(range(0,len(label_names)))
        print(confusion_matrix(y_true,y_pred,labels=a))
        print(accuracy_score(y_true,y_pred))
        return (accuracy_score(y_true,y_pred))

    #prediction on single audio
    def predict_function(self, path_to_audio, finetuned_path):
        # Specify the model name or path
        model_name_or_path = finetuned_path
        # Set the device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Load the configuration
        config = AutoConfig.from_pretrained(model_name_or_path)
        # Initialize the feature extractor
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name_or_path)
        # Set the sampling rate
        sampling_rate = feature_extractor.sampling_rate
        # Load the model for speech classification
        model = Wav2Vec2ForSpeechClassification.from_pretrained(model_name_or_path).to(device)
        # Specify the path to the audio file
        path = path_to_audio
        # Set the sampling rate for prediction
        sampling_rate = 16000
        # Make predictions on the audio file
        outputs = predict_file(path, sampling_rate, device, config, model, feature_extractor)
        # Print the outputs
        print(outputs)

    def extract_embedding(self, model_path, wav_path):
        input_audio, sample_rate = librosa.load(wav_path,  sr=self.sample_frequency)
        model = Wav2Vec2Model.from_pretrained(model_path)
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        i = feature_extractor(input_audio, return_tensors="pt", sampling_rate=sample_rate)
        with torch.no_grad():
            o = model(i.input_values, output_hidden_states=True)
 
        # print(o.keys())
        # print(o.last_hidden_state.shape)
        # print(o.extract_features.shape)

        embedding = np.mean(o.last_hidden_state.numpy()[0,:,:], axis=0)

        return embedding














        
