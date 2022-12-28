#!/bin/bash

# download and unzip the dataset to desired location
data_dir="$HOME/Projects/data"
wget -P $data_dir http://cs231n.stanford.edu/tiny-imagenet-200.zip
unzip tiny-imagenet-200.zip