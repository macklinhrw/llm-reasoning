#!/bin/bash

wandb login $WANDB_TOKEN
huggingface-cli login --token $HUGGING_FACE_TOKEN

aws configure set aws_acces_key_id $AWS_ACCESS_KEY_ID
aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
aws configure set region us-west-1