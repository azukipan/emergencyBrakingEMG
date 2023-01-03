"""
This code produces an EMG dataset according to the methodology described by Haufe et al. in "EEG potentials predict upcoming emergency brakings during simulated driving." dx.doi.org/10.1088/1741-2560/8/5/056001. Instead of the original modeling approach, we train and test other approaches for all test subjects from the Haufe et al. study. 

The original EMG data are part of a dataset from Haufe et al. called "Emergency braking during simulated driving." The dataset is available at http://bnci-horizon-2020.eu/database/data-sets
"""
import numpy as np
import h5py
import subprocess
import os
from glob import glob
from tqdm import tqdm
from statistics import mean
from pathlib import Path
    
from datasets import createDatasetFromEMGEvents, createDatasetFromEMGWithoutEvents, createDatasets
from modelDev import trainModel, evaluateModel
from models import getModel

def runcmd(cmd, verbose = False, *args, **kwargs):
    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

def main():
    #Find all paths for the test subject data. Download data files if the paths are not found.
    testSubjectDataFilePaths =  glob('../EMG_Dataset_Haufe/*.mat')
    if len(testSubjectDataFilePaths) < 18:
        os.chdir(r"../EMG_Dataset_Haufe/")
        runcmd("wget -i links.txt --no-check-certificate", verbose = True)
        os.chdir(r"../code/")
        testSubjectDataFilePaths =  glob('../EMG_Dataset_Haufe/*.mat')
    PSDComponentsUpperLimit = 129 #129 = Maximum number of power spectral density (PSD) components for samplingRate = 200 hz.
    #Setup progress bar
    pbar = tqdm(range(1, PSDComponentsUpperLimit+1)) #Number of PSD components from 1 to maximum in increments of 20.
    #pbar = tqdm(range(129, 130)) 
    
    modelOptions = ["Perceptron",
                    "LDA", 
                    "SVM",
                    "Logistic Regression"
                   ]
    
    #Create model development results folder.
    p = Path("../results")
    p.mkdir(exist_ok=True)
    
    for selectedModel in modelOptions:
        #Create log file for AUCs and number of PSD components.
        AUCLog = open("../results/{}_AUC.csv".format(selectedModel), "a+")
        AUCLog.write("Number of Components,Area Under Curve\n") #Headers
        AUCLog.close()

        pbar.set_description("Model:{}\nCalculating AUC for each number of PSD components.".format(selectedModel))
        for numberOfPSDComponents in pbar: #Iterate through all user-specified time segments to obtain AUCs.
            allTestSubjectAUCs = []
            for path in testSubjectDataFilePaths:
                f = h5py.File(path,'r')
                '''
                Read, sort and assign experimental to variables for: 
                signal channel names: cnt.clab
                sampling frequency: cnt.fs
                time-series data: cnt.x
                '''
                cnt = f.get('cnt')
                cnt.clab = np.array(cnt['clab'])
                cnt.fs = np.array(cnt['fs'])
                cnt.x = np.array(cnt['x']) 

                samplingRate = cnt.fs[0][0] #Down-/upsample rate for all data = 200Hz.

                #Read data for events corresponding to experimental data.
                mrk = f.get('mrk')
                mrk.classNames = mrk['className']
                mrk.time = mrk['time']
                mrk.y = mrk['y']
                mrk.events = mrk['event']

                #Find all car braking events (brake lights of lead vehicle turn on) and store corresponding timestamps.
                carBrakeTime = []
                for i in range(0, len(mrk.y)):
                    if mrk.y[i][1] == 1: #Check if car is braking, i.e. y[i] = 1
                        carBrakeTime.append(mrk.time[i][0]) #Store timestamp 

                #Create train and validation datasets
                brakingEvent_emg_PSD_train = []
                brakingEvent_emg_PSD_val = []

                noEvent_emg_PSD_train = []
                noEvent_emg_PSD_val = []

                data = cnt.x[61] #Channel 61 for EMG of tibialis anterior
                event_emg_PSD_train, event_emg_PSD_val = createDatasetFromEMGEvents(carBrakeTime, 
                                                                                    data, 
                                                                                    samplingRate, 
                                                                                    numberOfPSDComponents)
                _noEvent_emg_PSD_train, _noEvent_emg_PSD_val = createDatasetFromEMGWithoutEvents(mrk.time, 
                                                                                                 data, 
                                                                                                 samplingRate, numberOfPSDComponents)
                for array in event_emg_PSD_train: brakingEvent_emg_PSD_train.append(array)
                for array in event_emg_PSD_val: brakingEvent_emg_PSD_val.append(array)
                for array in _noEvent_emg_PSD_train: noEvent_emg_PSD_train.append(array)
                for array in _noEvent_emg_PSD_val: noEvent_emg_PSD_val.append(array)
                trainData, trainLabels, valData, valLabels = createDatasets(brakingEvent_emg_PSD_train,
                                                                            noEvent_emg_PSD_train,
                                                                            brakingEvent_emg_PSD_val,
                                                                            noEvent_emg_PSD_val)
                model = getModel(selectedModel)
                
                trainedModel = trainModel(model, trainData, trainLabels)
                AUCaccuracy = evaluateModel(trainedModel, selectedModel, valData, valLabels)
                allTestSubjectAUCs.append(AUCaccuracy)
            grandAverageAUC = round(mean(allTestSubjectAUCs), 3)
            print("Number of PSD components = ", numberOfPSDComponents, " | AUC accuracy = ", grandAverageAUC)
            
            #Save AUC and number of PSD components.
            AUCLog = open("../results/{}_AUC.csv".format(selectedModel), "a+")
            AUCLog.write("{},{}\n".format(numberOfPSDComponents, grandAverageAUC))
            AUCLog.close()
        
if __name__ == "__main__":
    main()