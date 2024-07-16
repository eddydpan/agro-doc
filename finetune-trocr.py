import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator
from datasets import load_metric

class IAMDataset(Dataset):
    def __init__(self, root_dir, df, processor, max_target_length=128):
        self.root_dir = root_dir
        self.df = df
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # get file name + text 
        file_name = self.df['file_name'][idx]
        text = self.df['text'][idx]
        file_ids = file_name.split("-")
        file_location = f"{file_ids[0]}/{file_ids[0]}-{file_ids[1]}/{file_name}"
        # prepare image (i.e. resize + normalize)
        image = Image.open(self.root_dir + file_location).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        # add labels (input_ids) by encoding the text
        labels = self.processor.tokenizer(text, 
                                          padding="max_length", 
                                          max_length=self.max_target_length).input_ids
        # important: make sure that PAD tokens are ignored by the loss function
        labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]

        encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        return encoding

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": cer}    
# For clean_sentences.txt:
# df = pd.read_csv('data/IAM_ascii/clean_sentences.txt', sep=" ", header=None, names=["file_name", "sentence_num", "word_segmentation", "graylevel_binarization", "num_components", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "text"])


# Read clean_lines.txt into a pandas dataframe
df = pd.read_csv('data/IAM_ascii/clean_lines.txt', sep=" ", header=None, names=["file_name", "word_segmentation", "graylevel_binarization", "num_components", "bbox_x", "bbox_y", "bbox_w", "bbox_h", "text"])

labels_df = df[["file_name", "text"]]
labels_df['file_name'] = labels_df['file_name'].apply(lambda x: x + '.png')
labels_df['text'] = labels_df['text'].apply(lambda x: x.replace("|", " "))


train_df, test_df = train_test_split(labels_df, test_size=0.2)
# we reset the indices to start from zero
train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
train_dataset = IAMDataset(root_dir='data/IAM_lines/',
                           df=train_df,
                           processor=processor)
eval_dataset = IAMDataset(root_dir='data/IAM_lines/',
                           df=test_df,
                           processor=processor)

# print("Number of training examples:", len(train_dataset))
# print("Number of validation examples:", len(eval_dataset))

# encoding = train_dataset[0]
# for k,v in encoding.items():
#     print(k, v.shape)

######### ex. case
# file_name = train_df['file_name'][0]
# file_ids = file_name.split("-")
# file_location = f"{file_ids[0]}/{file_ids[0]}-{file_ids[1]}/{file_name}"
# print(f"File Location: {file_location}")

# image = Image.open(train_dataset.root_dir + file_location).convert("RGB")
# image.format= "PNG"
# image.show()

# labels = encoding['labels']
# labels[labels == -100] = processor.tokenizer.pad_token_id
# label_str = processor.decode(labels, skip_special_tokens=True)
# print(label_str)
######


model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-stage1")
# set special tokens used for creating the decoder_input_ids from the labels
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
# make sure vocab size is set correctly
model.config.vocab_size = model.config.decoder.vocab_size

# set beam search parameters
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    fp16=True, 
    output_dir="./",
    logging_steps=2,
    save_steps=1000,
    eval_steps=200,
)
cer_metric = load_metric("cer")

trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=processor.feature_extractor,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=default_data_collator,
)
torch.backends.cudnn.enabled = False

trainer.train()