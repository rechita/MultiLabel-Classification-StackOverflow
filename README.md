# MultiLabel-Classification-StackOverflow

## Overview

I've created this GitHub repository for a fascinating project on multi-label classification worth 10 points. In this project, I will guide you through the process of identifying tags for Stack Exchange Questions using a machine learning model. I've prepared a dataset for this purpose, and I'll provide details on the dataset, model architecture, and necessary steps to successfully complete this project.

## Task

In this homework assignment, I'll be working on multilabel classification. Our objective is to assign relevant tags to questions from the Stack Exchange website, focusing on ten specific technology domains. Each question can have multiple tags, making this a multi-label classification problem. The ten categories for tags in the dataset are as follows:

1. c#
2. java
3. php
4. javascript
5. android
6. jquery
7. c++
8. python
9. iphone
10. asp.net

## Data

The dataset required for this project is available from the "0_Data" folder in the file named "df_multilabel_hw_cleaned.joblib". The dataset is in the form of a Pandas DataFrame, and you can load it using the `joblib.load(file)` method. Here are some important points regarding the dataset:

- No preprocessing is required since you will be provided with a cleaned dataset.
- The "Tag_Number" column is used as your labels, but the data type of list elements is 'object'. To convert it to integers, you can use the `ast.literal_eval()` function from the 'ast' library. You'll need to import this library first.
- Since this is a multilabel dataset, you'll need to perform one-hot encoding of your dependent variable. You can achieve this using the MultiLabelBinarizer from the 'sklearn' library.
- You should create train/valid/test splits from the dataset. Use 60% for training, 20% for testing, and the remaining 20% for the validation dataset.

## Model

My neural network should have the following layers:

1. EmbeddingBag
2. Hidden Layer 1
3. ReLU Activation Function
4. Dropout Layer 1
5. Batch Normalization Layer 1
6. Hidden Layer 2
7. ReLU Activation Function
8. Dropout Layer 2
9. Batch Normalization Layer 2
10. Output Layer

For training the model, I'll use the following hyperparameters:

- `EMBED_DIM` = 300
- `VOCAB_SIZE` = len(your_vocab)
- `OUTPUT_DIM` = 10
- `HIDDEN_DIM1` = 200
- `HIDDEN_DIM2` = 100
- `EPOCHS` = 5
- `BATCH_SIZE` = 128
- `LEARNING_RATE` = 0.001
- `WEIGHT_DECAY` = 0.000
- `CLIP_TYPE` = 'value'
- `CLIP_VALUE` = 10
- `PATIENCE` = 5

To incorporate gradient clipping, I'll add the following lines within the step function discussed in the class:

```python
loss.backward()

# Clip gradients before the optimizer step
clip_grad_value_(model.parameters(), clip_value=10.0)

# Update parameters
optimizer.step()
