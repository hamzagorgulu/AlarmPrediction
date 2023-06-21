There are 3 scripts in this folder. 

1 - preprocessing.py
-  This script reads alarm data from a CSV file and performs some data preprocessing. It selects specific columns from the dataframe, converts the 'StartTime' and 'EndTime' columns to Pandas datetime format, and creates a list of dictionaries representing the alarm data. It then segments the alarm sequences based on a time delta of 30 minutes and creates a list of alarm sequences. Finally, it saves the alarm sequences to a text file.

2 - helpers.py
- The helpers.py script contains functions to preprocess data for further use. One of the functions is remove_chattering_alarms, which removes chattering alarms that occur more than a specified number of times within a 1 minute interval. This function returns the filtered list of alarms.
Another function in the script is sequence_segmentation, which segments an alarm sequence based on a specified time delta. It divides the alarms into separate lists, each representing a segment with alarms occurring within the given time delta.
Additionally, the sequence_lst function generates a list of sequences based on the segmented alarm data. It joins the alarm definitions within each segment and appends the sequence to a list if it meets a minimum sequence length requirement.

3 - run.py:
- This script implements a decoder-only Transformer model for alarm prediction. It reads preprocessed alarm text data, and splits it into train, validation, and test sets. The model consists of several components such as Multi-Head Attention, FeedForward, and Block modules. It trains the model using the train set, evaluates the performance on the validation set, saves the training records, and finally tests the model on the test set. It also includes functions to calculate and plot the training and validation losses.

To run this project, use the following commands after putting your alarm dataset:
python preprocess.py
python run.py