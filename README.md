# SERVAL

Sound Event Recognition-based Vigilance for Alerting and Localization

Rework of the original setup in 2020, old code is archived in the folder 20200209-old-code.

## Installation

See the notebooks for more details.

We use [this](https://github.com/igor-panteleev/youtube-8m#running-on-your-own-machine) now

Old original [EARS project](https://github.com/karoldvl/EARS/tree/master/ears) for setup details. 

## Use docker with tensorflow
We use docker images on a GPU enabled host machine.

    docker run --gpus all -it -p 8888:8888 -v /home/hugo/data/git/serval:/tf/serval tensorflow/tensorflow:latest-gpu-py3-jupyter

    docker run --gpus all -it -p 8888:8888 -v /home/hugo/data/git/serval:/tf/serval tensorflow/tensorflow:1.15.2-gpu-py3-jupyter

We are using the latest tensorflow version 1.15.2 , so we do NOT use tensorflow 2.x yet.

## Description of application

please vistit our website [sensingclues/serval](https://sensingclues.com/serval/) for a description of the real-life application.

## Useful other info

To train classification model next resources have been used:

    [Google AudioSet](https://research.google.com/audioset/)
    [YouTube-8M model](https://github.com/google/youtube-8m)
    [Tensorflow vggish model](https://github.com/tensorflow/models/tree/master/research/audioset)
    [Original Device hive blog](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/)
    [Igor's Youtube8M startercode](https://www.iotforall.com/tensorflow-sound-classification-machine-learning-applications/)

You can try to train model with more steps/samples to get more accuracy.

### LICENCE
Copyright for portions of project SERVAL are held by 2017 Karol J. Piczak as part of project [EARS](https://github.com/karoldvl/EARS). All other copyright for project SERVAL are held by Sensing Clues 2017.

GPL-3.0 © Sensing Clues 2017 

## Notebooks
### 00-01-00-Enumerate-all-custom-wav-samples
This notebook enumerates all custom (non-youtube) wav samples from the "wav_samples_custom" folder.
The output is a list of all the custom wav samples in the "wav_samples_custom" folder and their class / label information as provided in the "wav_samples_custom_labels.csv" file

#### Input
input_wav_samples_folder      = serval_data_folder + "/wav_samples_custom"
wav_samples_custom_labels_csv = input_wav_samples_folder + "/wav_samples_custom_labels.csv"

#### Output
target_wav_samples_custom_all_enumerated_csv = serval_data_folder + "/wav_samples_custom/wav_samples_custom_all_enumerated_and_labeled.csv"

### 00-01-00-Enumerate-all-youtube-wav-samples
This notebook enumerates all youtube provided wav samples from the "wav_samples_youtube" folder.
The output is a list of all the youtube wav samples in the "wav_samples_youtube" folder and their class / label information as provided in the "balanced_train_segments.csv", "eval_segments.csv" and "unbalanced_train_segments.csv" files

#### Input
youtube_wav_samples_directory = serval_data_folder + "/wav_samples_youtube"

youtube_class_labels_filepath = serval_data_folder + "/csv_files/class_labels_indices.csv"

youtube_wav_balanced_train_class_labels_filepath   = youtube_wav_samples_directory + "/balanced_train_segments.csv"
youtube_wav_balanced_eval_class_labels_filepath    = youtube_wav_samples_directory + "/eval_segments.csv"
youtube_wav_unbalanced_train_class_labels_filepath = youtube_wav_samples_directory + "/unbalanced_train_segments.csv"

youtube_wav_balanced_train_sample_directory   = youtube_wav_samples_directory + "/bal"
youtube_wav_balanced_eval_sample_directory    = youtube_wav_samples_directory + "/eval"
youtube_wav_unbalanced_train_sample_directory = youtube_wav_samples_directory + "/unbal"

#### Output
target_balanced_wav_samples_enumerated_filepath   = serval_data_folder + "/csv_files/wav_samples_youtube_balanced_all_enumerated_and_labeled.csv"
target_unbalanced_wav_samples_enumerated_filepath = serval_data_folder + "/csv_files/wav_samples_youtube_unbalanced_all_enumerated_and_labeled.csv"
target_eval_wav_samples_enumerated_filepath       = serval_data_folder + "/csv_files/wav_samples_youtube_eval_all_enumerated_and_labeled.csv"

### 00-01-00-Select-wav-samples
This notebook enumerates selects the wav samples from the complete pool of wav samples. The samples you select here will be available for your project. Later on you can select the exact labels / classes that you would like to use to build your model with. From now on all projects files will be stored in the project directory. Selected classes (labels) are stored in the "input_selected_wav_samples.csv" file in your project directory.

#### Input
Project directory:
project_name = 'Amsterdam_custom_samples'

Set your serval data folder (should be correctly set already):
serval_data_folder = "../data"
project_data_folder = serval_data_folder + '/' + project_name

input_balanced_wav_samples_enumerated_filepath   = serval_data_folder + "/csv_files/wav_samples_youtube_balanced_all_enumerated_and_labeled.csv"
input_unbalanced_wav_samples_enumerated_filepath = serval_data_folder + "/csv_files/wav_samples_youtube_unbalanced_all_enumerated_and_labeled.csv"
input_eval_wav_samples_enumerated_filepath       = serval_data_folder + "/csv_files/wav_samples_youtube_eval_all_enumerated_and_labeled.csv"
input_custom_wav_samples_enumerated_filepath     = serval_data_folder + "/wav_samples_custom/wav_samples_custom_all_enumerated_and_labeled.csv"

#### Output
input_selected_classes_filepath = project_data_folder + '/csv_files/input_selected_wav_samples.csv'
target_selected_classes_filepath = project_data_folder + '/csv_files/output_selected_wav_samples.csv'

### 00-01-02-Audio-Resample-Wavfiles
This notebook re-samples the selected wav samples to have a distance of 6 db, 12 db and 18 db distance. This is done by reducing the volume. Note: In this step we also discard samples that are deemed to be to short (less than 9.5 seconds in length). The resampled wav files are stored in the "wav_samples" directory inside your project directory.

During the resampeling process we also create a train and test (eval) set.

#### Input
input_selected_wav_samples_filepath       = project_data_folder + "/csv_files/output_selected_wav_samples.csv"
target_resampled_wav_samples              = project_data_folder + "/csv_files/output_resampled_wav_samples.csv"
target_intermediate_resampled_wav_samples = project_data_folder + "/csv_files/intermediate_output_resampled_wav_samples.csv"

#### Output
target_resampled_wav_folder               = project_data_folder + "/wav_samples"

### 00-01-03-Audio-Combine-Wavfiles
This notebook combines two wav files as defined by the "input_wav_samples_to_combine.csv" file. The output of this notebook is written back to the "output_resampled_wav_samples.csv". Note: Running this notebook multiple times will enlarge the "output_resampled_wav_samples.csv" file that is used for the next step. To be sure you can always take a step back we have added a backup routine that will store the original "output_resampled_wav_samples.csv" with a timestamp.

#### Input
input_wav_samples_to_combine_filepath     = project_data_folder + "/csv_files/input_wav_samples_to_combine.csv"

#### Output
target_resampled_wav_samples              = project_data_folder + "/csv_files/output_resampled_wav_samples.csv"
target_resampled_wav_samples_backup       = project_data_folder + "/csv_files/output_resampled_wav_samples_backup_" + datetime.today().strftime('%Y%m%d_%H%M%S') + '.csv'
target_resampled_wav_folder               = project_data_folder + "/wav_samples"

### 00-02-01-Convert-wav-2-tfrecord
This notebook converts all available wav files to tfrecords. The input is the "output_resampled_wav_samples.csv" file which contains all available wav files. The output is stored in the tfrecords directory.

#### Input
input_wav_sample_filepath = project_data_folder + '/csv_files/output_resampled_wav_samples.csv'

#### Output
target_tfrecord_folder    = project_data_folder + '/tfrecords_all'

### 00-01-03-Audio-Combine-Wavfiles
This notebook combines two wav files as defined by the "input_wav_samples_to_combine.csv" file. The output of this notebook is written back to the "output_resampled_wav_samples.csv". Note: Running this notebook multiple times will enlarge the "output_resampled_wav_samples.csv" file that is used for the next step. To be sure you can always take a step back we have added a backup routine that will store the original "output_resampled_wav_samples.csv" with a timestamp.

#### Input
input_wav_samples_to_combine_filepath     = project_data_folder + "/csv_files/input_wav_samples_to_combine.csv"

#### Output
target_resampled_wav_samples              = project_data_folder + "/csv_files/output_resampled_wav_samples.csv"
target_resampled_wav_samples_backup       = project_data_folder + "/csv_files/output_resampled_wav_samples_backup_" + datetime.today().strftime('%Y%m%d_%H%M%S') + '.csv'
target_resampled_wav_folder               = project_data_folder + "/wav_samples"

### 00-03-01-TFrecord-re-labeling-and-batches
This notebook uses the "input_selected_classes.csv" file to select and re-label the chosen classes / labels. This csv file defines exactly which classes to use, what proportion to keep, what to oversample for both train and eval sets. Once this step is completely you will have multiple tfrecords that are randomly permutated. The output with exact number of selected samples will be stored in the "output_class_mapping.csv" file.

#### Input
input_selected_classes_filepath = serval_data_folder + '/' + custom_data_folder + '/csv_files/input_selected_classes.csv'

tfrecords_train_search_string   = serval_data_folder + '/' + custom_data_folder + '/tfrecords_all/train_*.tfrecord'
tfrecords_eval_search_string    = serval_data_folder + '/' + custom_data_folder + '/tfrecords_all/eval_*.tfrecord'

#### Output
output_class_mapping_filepath   = serval_data_folder + '/' + custom_data_folder + '/csv_files/output_class_mapping.csv' 
output_tfrecords_train_path     = serval_data_folder + '/' + custom_data_folder + '/tfrecords_model_input/train'
output_tfrecords_eval_path      = serval_data_folder + '/' + custom_data_folder + '/tfrecords_model_input/eval'

### 01-00-Train-Serval
This notebook trains the neural network based on the available training samples in the '/tfrecords_model_input/train' directory. The output is the actual model.

#### Input
tdp = serval_data_folder + '/' + custom_data_folder + '/tfrecords_model_input/train'

#### Output
tmd = serval_data_folder + '/' + custom_data_folder + '/model_output'

### 02-00-Evaluate-Serval-model
This notebook evaluates the neural network based on the available evaluation samples in the '/tfrecords_model_input/eval' directory. The output is the resulting classification per evaluation sample.

#### Input
edp = serval_data_folder + '/' + custom_data_folder + '/tfrecords_model_input/eval'
class_map = serval_data_folder + '/' + custom_data_folder + "/csv_files/output_class_mapping.csv"

#### Output
tmd = serval_data_folder + '/' + custom_data_folder + '/model_output'
tmdl = tmd + "/eval_log"
evaluation_results_csv = serval_data_folder + '/' + custom_data_folder + "/csv_files/output_serval_evaluation.csv"

### 02-00-Evaluate-Serval-model
This notebook interprets the evaluation of the neural network from the previous step.

#### Input
class_map = serval_data_folder + '/' + custom_data_folder + "/csv_files/output_class_mapping.csv"

#### Output
er_csv = serval_data_folder + '/' + custom_data_folder + "/csv_files/output_serval_evaluation.csv"