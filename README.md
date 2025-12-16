# PhysioNet-Challenge-CIC_UPIITA_TEAM

This project implements a classifier for the annual George B. Moody PhysioNet Challenges, which aims to develop an automatic process for the identification of Chagas cases using electrocardiogram (ECG) signal data.

In this first installment, a proposal is presented that includes the preprocessing of ECG signals. Filters are applied to correct the out-of-phase signal and highlight important characteristics of the signals. Subsequently, a wavelet transform is used to obtain the approximation and details of the signals. These components are concatenated and used as input for a one-dimensional convolutional neural network (1D-CNN), designed to distinguish between people with and without Chagas.

## Project structure: 
- CNN1D.py: Contains the implementation of the one-dimensional convolutional neural network.
- data_loader.py: Loads and processes databases using biological signal processing techniques.
- run_code.py: Compile all code and calculate model performance metrics.
- requirements.txt: Contains all the necessary dependencies to run the project.

## How do I run these scripts in Docker?

Docker and similar platforms allow you to containerize and package your code with specific dependencies so that your code can be reliably run in other computational environments.

To increase the likelihood that we can run your code, please [install](https://docs.docker.com/get-docker/) Docker, build a Docker image from your code, and run it on the training data. To quickly check your code for bugs, you may want to run it on a small subset of the training data, such as 1000 records.

If you have trouble running your code, then please try the follow steps to run the example code.

This code uses a GPU to train the architecture. It's recommended that you have CUDA and CUDNN installed for everything to work properly.

1. Create a folder `example` in your home directory with several subfolders.

        user@computer:~$ cd ~/
        user@computer:~$ mkdir example
        user@computer:~$ cd example
        user@computer:~/example$ mkdir training_data holdout_data model holdout_outputs

2. Download the training data from the [Challenge website](https://physionetchallenges.org/2025/#data). Put some of the training data in `training_data` and `holdout_data`. You can use some of the training data to check your code (and you should perform cross-validation on the training data to evaluate your algorithm).

3. Download or clone this repository in your terminal.

        user@computer:~/example$ git clone https://github.com/physionetchallenges/python-example-2025.git
   

5. Build a Docker image and run the example code in your terminal.

        user@computer:~/example$ ls
        holdout_data  holdout_outputs  model  python-example-2025  training_data

        user@computer:~/example$ cd python-example-2025/

        user@computer:~/example/python-example-2025$ docker build -t image .

        Sending build context to Docker daemon  [...]kB
        [...]
        Successfully tagged image:latest

If you want to run it without GPU, use the following line: 

        user@computer:~/example/python-example-2025$ docker run -it -v ~/example/model:/challenge/model -v ~/example/holdout_data:/challenge/holdout_data -v ~/example/holdout_outputs:/challenge/holdout_outputs -v ~/example/training_data:/challenge/training_data image bash

If you want ro run it with GPU, use the following line:

        user@computer:~/example/python-example-2025$ docker run --gpus all -it -v ~/example/model:/challenge/model -v ~/example/holdout_data:/challenge/holdout_data -v ~/example/holdout_outputs:/challenge/holdout_outputs -v ~/example/training_data:/challenge/training_data image bash

Then

        root@[...]:/challenge# ls
            Dockerfile             holdout_outputs        run_model.py
            evaluate_model.py      LICENSE                training_data
            helper_code.py         README.md      
            holdout_data           requirements.txt

        root@[...]:/challenge# python train_model.py -d training_data -m model -v

        root@[...]:/challenge# python run_model.py -d holdout_data -m model -o holdout_outputs -v

        root@[...]:/challenge# python evaluate_model.py -d holdout_data -o holdout_outputs
        [...]

        root@[...]:/challenge# exit
        Exit

## Load Data 
This project exclusively uses the SAMI-TROP database. In order to run the program correctly, this database needs to be loaded. Here's an example of how data is loaded into your code:
```
if os.path.isdir("Samitrop"):
    df_samitrop = pd.read_csv('Samitrop/exams.csv')
    samitrop = h5py.File('Samitrop/exams.hdf5','r')
```
Make sure you have the Samitrop folder in the working directory, which contains the exams.csv and exams.hdf5 files. The exams.csv file should include the tags needed for model training and evaluation, while exams.hdf5 should contain the electrocardiogram signals.

## Note
This is a first approximation in which we focus on the processing of the signals before applying the classification algorithm. The main objective of this proposal is to highlight the distinctive characteristics of each signal and provide the algorithm with a clean and optimized signal, thus facilitating its classification.

## Authors
1. Team Member 1:
   - Name: José-Adrián Chávez-Olvera
   - Email: jchavezo1603@alumno.ipn.mx
2. Team Member 2:
   - Name: Arián Villalba-Tapia
   - Email: avillalbat1600@alumno.ipn.mx
3. Team Member 3:
   - Name: Blanca Tovar-Corona
   - Email: bltovar@ipn.mx
4. Team Member 4:
   - Name: René Luna-García
   - Email: rlunag@ipn.mx
