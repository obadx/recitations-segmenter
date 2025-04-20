from pathlib import Path
import json
import os

from dotenv import load_dotenv
import wandb
from transformers import (
    Wav2Vec2BertForAudioFrameClassification,
    TrainingArguments,
    Trainer
)
from huggingface_hub import login as hf_login
from datasets import load_dataset, DatasetDict, Dataset
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import default_collate
import torch
from torch.nn import CrossEntropyLoss


# TODO:
# * Hyberparamets
# * RUN full training on part of the real dataset [DONE]
# * disable gradient checkpinting (NO) ????
# * evaluate per half epoch ????
# * add train logs [Ignored] slowing training code
# * add wandb [DONE]
# * save test logs [DONE]
# * push to hup [DONE]
# * start/resume [DONE]
# * add seed [DONE]
# * understand padding (padd with 0s but since we use log mel filter bank it becomes 1) [DONE]
# * use evalute library [Not Needed]
# * see logging (tensor board) [DONE]

def load_secrets():
    # Load environment variables from .env
    load_dotenv()

    # Retrieve tokens
    wandb_token = os.getenv("WANDB_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_TOKEN")

    # Initialize WandB (automatic if env var is set)
    if wandb_token:
        wandb.login(key=wandb_token)
    else:
        print("WandB token not found!")

    # Log into HuggingFace Hub
    if hf_token:
        hf_login(token=hf_token)
    else:
        print("HuggingFace token not found!")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Flatten and filter out ignored indices (-100)
    predictions_flat = predictions.flatten()
    labels_flat = labels.flatten()
    mask = labels_flat != -100
    preds = predictions_flat[mask]
    lbs = labels_flat[mask]

    return {
        'accuracy': accuracy_score(lbs, preds),
        'precision': precision_score(lbs, preds, average='binary', zero_division=0),
        'recall': recall_score(lbs, preds, average='binary', zero_division=0),
        'f1': f1_score(lbs, preds, average='binary', zero_division=0),
    }


class CustomTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        outputs = model(
            input_features=inputs['input_features'],
            attention_mask=inputs['attention_mask'],
        )
        logits = outputs[0]
        batch_size, seq_len, num_labels = logits.shape
        loss_fact = CrossEntropyLoss()
        loss = loss_fact(
            logits.view(-1, num_labels),
            inputs['labels'].view(-1))

        return (loss, outputs) if return_outputs else loss


class LabelProcessor:
    def __call__(self, features):
        # Process labels
        new_features = []
        for feature in features:
            new_feature = {}

            new_feature['labels'] = torch.tensor(
                feature['labels'], dtype=torch.long)
            new_feature['attention_mask'] = torch.tensor(
                feature['attention_mask'], dtype=torch.long)
            new_feature['input_features'] = torch.tensor(
                feature['input_features'], dtype=torch.float32)

            # Replace -100 with 0
            labels = new_feature['labels']
            labels[labels == -100] = 0
            new_feature["labels"] = labels
            new_features.append(new_feature)

        # Use default collator for batching (no padding changes)
        return default_collate(new_features)


if __name__ == '__main__':
    every = 0.25
    # loading wandb tokens ans HF login
    load_secrets()

    # Initializaze wanddb
    # set the wandb project where this run will be logged
    os.environ["WANDB_PROJECT"] = "recitation-segmenter-v2"

    # save your trained model checkpoint to wandb
    os.environ["WANDB_LOG_MODEL"] = "false"

    # turn off watch to log faster
    os.environ["WANDB_WATCH"] = "false"

    # Load dataset
    # Update with your dataset path
    dataset = load_dataset(
        'obadx/recitation-segmentation-augmented', num_proc=16)

    # For testing only
    # dataset['train'] = dataset['train'].take(400)
    # dataset['validation'] = dataset['validation'].take(100)
    # dataset['test'] = dataset['test'].take(100)
    #
    # # TODO: for testing only
    # new_ds = {'train': [], 'validation': [], 'test': []}
    # for split in dataset:
    #     for item in dataset[split]:
    #         new_ds[split].append(item)
    #     new_ds[split] = Dataset.from_list(new_ds[split])
    # dataset = DatasetDict(new_ds)

    # Load pre-trained model
    model = Wav2Vec2BertForAudioFrameClassification.from_pretrained(
        'facebook/w2v-bert-2.0',
        num_labels=2,  # Binary classification (0s and 1s)
        problem_type="single_label_classification"
    )

    # Configure training arguments
    training_args = TrainingArguments(
        seed=42,
        output_dir='./results',
        eval_strategy='steps',
        eval_steps=every,
        save_strategy='steps',
        save_steps=every,
        logging_strategy='steps',
        logging_steps=every,
        learning_rate=5e-5,
        per_device_train_batch_size=50,
        per_device_eval_batch_size=64,
        num_train_epochs=1,
        dataloader_num_workers=16,
        weight_decay=0.01,
        logging_dir='./logs',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        # push_to_hub=True,  # this pushed every checkpoint to the hup we want to push the best model only
        hub_model_id='obadx/recitation-segmenter-v2',  # Update with your model name
        bf16=True,
        warmup_ratio=0.2,
        resume_from_checkpoint='./results/checkpoint-1097',
        optim='adamw_torch',
        lr_scheduler_type='constant',
        report_to=["tensorboard", "wandb"],
        gradient_checkpointing=True,  # Optional for memory savings # TODO :set it to False
        save_total_limit=3,
    )

    # Initialize label processor
    label_processor = LabelProcessor()

    # Initialize Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        compute_metrics=compute_metrics,
        data_collator=label_processor,
    )

    # # Start training
    # if list(Path('./results').glob('checkpoint-*')):
    #     print('Resuming !')
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()

    # Final evaluation on test set
    print('Start Testing')
    test_results = trainer.evaluate(
        dataset['validation'], metric_key_prefix='eval_')
    with open('./results/last_eval_results.json', 'w') as f:
        json.dump(test_results, f, indent=4)
    print("Eval Results:", test_results)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # # Push model and tokenizer to Hub
    # trainer.push_to_hub()
