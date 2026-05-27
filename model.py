import os
import gc
import numpy as np
import pandas as pd
import torch
import evaluate

from datasets import (load_dataset,load_from_disk,Image as HFImage)
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel, Seq2SeqTrainer, Seq2SeqTrainingArguments, default_data_collator)



# GPU OPTIMIZATION
torch.backends.cudnn.benchmark = True


# SETTINGS
IAM_PARQUET_PATH = "./combined_parquet"
MODEL_NAME = "microsoft/trocr-small-handwritten"
OUTPUT_DIR = "./etrocr_model"
PROCESSED_DATASET_PATH = "./processed_dataset"
BATCH_SIZE = 1
NUM_EPOCHS = 2
MAX_TARGET_LENGTH = 128
LEARNING_RATE = 5e-5
IMAGE_SIZE = (256, 96)
TEST_SUBSET_SIZE = 100



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using Device:", device)

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA Version:", torch.version.cuda)


    # LOAD PROCESSOR
    processor = TrOCRProcessor.from_pretrained(MODEL_NAME)

    # LOAD DATASET
    if os.path.exists(PROCESSED_DATASET_PATH):
        print("Loading cached processed dataset...")
        dataset = load_from_disk(PROCESSED_DATASET_PATH)

    else:
        print("Loading parquet dataset...")
        dataset = load_dataset("parquet",
            data_files={
                "train": os.path.join(IAM_PARQUET_PATH, "train.parquet"),
                "validation": os.path.join(IAM_PARQUET_PATH, "validation.parquet"),
                "test": os.path.join(IAM_PARQUET_PATH, "test.parquet"),
            }
        )

        dataset = dataset.cast_column("image", HFImage())
        print(dataset)


        # PREPROCESSING
        def preprocess(example):
            image = example["image"]

            # Convert to RGB if not already
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize image
            image = image.resize(IMAGE_SIZE)

            # Convert image to tensor
            pixel_values = processor(image, return_tensors="pt").pixel_values

            # tokenize text
            labels = processor.tokenizer(example["text"], padding="max_length", truncation=True, max_length=MAX_TARGET_LENGTH).input_ids

            # Replace padding token ids of the labels by -100 so it's ignored by the loss
            labels = [
                label
                if label != (processor.tokenizer.pad_token_id)
                else -100
                for label in labels
            ]

            return {
                "pixel_values": pixel_values.squeeze(),
                "labels": labels
            }


        # Preprocess dataset and save it
        print("Preprocessing dataset...")

        dataset = dataset.map(preprocess, remove_columns=dataset["train"].column_names, num_proc=1, load_from_cache_file=False)
        dataset.save_to_disk(PROCESSED_DATASET_PATH)

        print("Processed dataset cached!")


    # TRAINING LOOP
    lr = LEARNING_RATE

    # clear memory before loading model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # load model
    model = (VisionEncoderDecoderModel.from_pretrained(MODEL_NAME))
    model.to(device)

    # MODEL CONFIGURATION
    model.config.decoder_start_token_id = (processor.tokenizer.cls_token_id) # start token
    model.config.pad_token_id = (processor.tokenizer.pad_token_id) # pad token
    model.config.eos_token_id = (processor.tokenizer.sep_token_id) # end token
    model.generation_config.max_length = 64 # max output length
    model.generation_config.num_beams = 1 # no beam search for faster training
    model.gradient_checkpointing_enable() # memory optimization


    # TRAINING ARGS
    training_args = (
        Seq2SeqTrainingArguments(
            output_dir= f"{OUTPUT_DIR}_{lr}",
            learning_rate= lr,
            num_train_epochs= NUM_EPOCHS,
            per_device_train_batch_size= BATCH_SIZE,
            per_device_eval_batch_size= BATCH_SIZE,
            gradient_accumulation_steps=1,
            dataloader_num_workers=0,
            fp16=torch.cuda.is_available(),
            logging_steps=1000,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=5000,
            save_total_limit=1,
            predict_with_generate=False,
            report_to="none"
        )
    )

    # INITIALIZE TRAINER
    trainer = Seq2SeqTrainer(
        model=model,
        processing_class=processor,
        args=training_args,
        train_dataset=dataset["train"],
        data_collator=default_data_collator
    )

    # TRAIN
    trainer.train()

    print("\nTraining finished!")

    best_model = model

    # CLEAR MEMORY
    del trainer

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


    # SAVE MODEL
    best_model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("Model saved!")

    
    # FINAL EVALUATION
    print("\nRunning final evaluation...")

    # LOAD METRICS
    cer_metric = evaluate.load("cer")
    wer_metric = evaluate.load("wer")

    test_subset = (dataset["test"].select(range(min(TEST_SUBSET_SIZE, len(dataset["test"])))))

    # MANUAL PREDICTION LOOP
    pred_texts = []
    true_texts = []

    best_model.eval()

    for sample in test_subset:
        # convert pixel values to tensor and move to device
        pixel_values = torch.tensor(sample["pixel_values"], dtype=torch.float32).unsqueeze(0).to(device)

        # generate prediction
        with torch.no_grad():
            generated_ids = best_model.generate(pixel_values)
        prediction = processor.batch_decode(generated_ids,skip_special_tokens=True)[0]

        # decode labels- filter out -100 and decode to text
        labels = sample["labels"]
        labels = [
            label
            for label in labels
            if label != -100
        ]

        # decode labels to text
        ground_truth = processor.tokenizer.decode(labels,skip_special_tokens=True)

        pred_texts.append(prediction)
        true_texts.append(ground_truth)


    # FINAL METRICS
    final_cer = cer_metric.compute(predictions=pred_texts,references=true_texts)
    final_wer = wer_metric.compute(predictions=pred_texts,references=true_texts)
    final_accuracy = np.mean([
        pred.strip() ==
        label.strip()
        for pred, label in zip(pred_texts,true_texts)
    ])

    print("\n===== FINAL RESULTS =====")
    print("CER:", final_cer)
    print("WER:", final_wer)
    print("Accuracy:", final_accuracy)


    # SAVE RESULTS
    results_df = pd.DataFrame({
        "Ground Truth": true_texts,
        "Prediction": pred_texts
    })
    results_df.to_csv("prediction_results.csv",index=False)

    metrics_df = pd.DataFrame({
        "CER": [final_cer],
        "WER": [final_wer],
        "Accuracy": [final_accuracy],
        "Best Learning Rate": [LEARNING_RATE]
    })
    metrics_df.to_csv("evaluation_metrics.csv",index=False)

    print("Evaluation results saved!")


    # MEMORY CLEANUP

    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\nTraining Complete!")


if __name__ == "__main__":
    main()