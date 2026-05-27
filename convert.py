import os

from datasets import (Image as HFImage, load_dataset, Dataset, concatenate_datasets)

# SETTINGS
IAM_PARQUET_PATH = "./parquet"
CUSTOM_IMAGE_FOLDER = "./custom_images"
OUTPUT_DIR = "./etrocr_model"


# LOAD IAM-LINE PARQUET DATASET
dataset = load_dataset(
    "parquet",
    data_files={
        "train": os.path.join(IAM_PARQUET_PATH, "train*.parquet"),
        "validation": os.path.join(IAM_PARQUET_PATH, "validation*.parquet"),
        "test": os.path.join(IAM_PARQUET_PATH, "test*.parquet"),
    }
)

print(dataset)


# CREATE CUSTOM DATASET
custom_data = []

for filename in os.listdir(CUSTOM_IMAGE_FOLDER):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(CUSTOM_IMAGE_FOLDER, filename)

        # filename becomes label
        label = os.path.splitext(filename)[0]

        custom_data.append({
            "image": image_path,
            "text": label
        })


custom_dataset = Dataset.from_list(custom_data)
print("Custom Dataset Size:", len(custom_dataset))

# Cast IAM dataset image column
dataset = dataset.cast_column("image", HFImage())

# Cast custom dataset image column
custom_dataset = custom_dataset.cast_column("image", HFImage())

# Split custom dataset into 70:15:15
split_dataset = custom_dataset.train_test_split(test_size=0.30, seed=42)

custom_train = split_dataset["train"]
temp_dataset = split_dataset["test"]

temp_split = temp_dataset.train_test_split(test_size=0.50,seed=42)

custom_validation = temp_split["train"]
custom_test = temp_split["test"]


# COMBINE DATASETS
train_dataset = concatenate_datasets([
    dataset["train"],
    custom_train
])

validation_dataset = concatenate_datasets([
    dataset["validation"],
    custom_validation
])

test_dataset = concatenate_datasets([
    dataset["test"],
    custom_test
])

print("Combined Train Size:", len(train_dataset))
print("Combined Validation Size:", len(validation_dataset))
print("Combined Test Size:", len(test_dataset))


# Create output folder
os.makedirs("./combined_parquet", exist_ok=True)

# Save datasets
train_dataset.to_parquet("./combined_parquet/train.parquet")
validation_dataset.to_parquet("./combined_parquet/validation.parquet")
test_dataset.to_parquet("./combined_parquet/test.parquet")

print("Parquet files saved successfully!")