# -*- coding: utf-8 -*-
# MIT License
# 
# Copyright(c) 2018 Aalborg University
# Joakim Bruslund Haurum, March 2018
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files(the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions :
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
import os 

def plot_distribution_params(x_axis, gaussParams, kalmanGaussParams, threshold = 0.35, outputPath = ""):
    """
    Plots the Mean, Standard Deviation and Mixture Proportion obtained from the EM algorithm and Kalman filtering, against the frame count.
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        gaussParams : Panda DataFrame containing the EM estiamted Gaussian parameters of the rain streaks, with dimensions [n_frames, 3]
        kalmanGaussParams : Panda DataFrame containing the Kalman filtered Gaussian parameters of the rain streaks, with dimensions [n_frames, 3]
        threshold : The threshold value of the Gaussian Mixture Proportion used to determine whether to update Kalman filter or not
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf with 3 plots, where the EM estiamted and Kalman filtered parameters are compared against each other
    """
    
    gaussValues = gaussParams.values
    kalmanValues = kalmanGaussParams.values
    x_axis_min = np.min(x_axis)
    x_axis_max = np.max(x_axis)
    
    titles = ["Mean", "Standard Deviation", "Mixture Proportion"]
    lgnd = ["EM", "Kalman"]
    
    
    plt.figure(1,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    plt.suptitle("Distribution parameters")
    
    plt.subplot(1,3,1)
    plt.title(titles[0])
    plt.plot(x_axis,gaussValues[:,0],'b')
    plt.plot(x_axis,kalmanValues[:,0],'orange')
    plt.legend(lgnd, loc='lower right')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    
    plt.subplot(1,3,2)
    plt.title(titles[1])
    plt.plot(x_axis,gaussValues[:,1],'b')
    plt.plot(x_axis,kalmanValues[:,1],'orange')
    plt.legend(lgnd, loc='lower right')
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    
    
    plt.subplot(1,3,3)
    plt.title(titles[2])
    plt.plot(x_axis,gaussValues[:,2],'b')
    plt.plot(x_axis,kalmanValues[:,2],'orange')
    
    horiz_line_data = np.array([threshold for i in range(len(x_axis))])
    plt.plot(x_axis, horiz_line_data, 'g--') 
    plt.xlabel("Frame")
    plt.grid(True)
    plt.xlim(x_axis_min, x_axis_max)
    
    lgnd.append("Threshold: {:.2f}".format(threshold))    
    plt.legend(lgnd, loc='lower right')
    
    
    #plt.show()
    plt.savefig(outputPath + "Distribution_Parameters.pdf", bbox_inches="tight")


def plot_rain_intensity(x_axis, rainIntensities, outputPath = ""):
    """
    Plots the calculated rain intensity value against the number of frames. Both the rain intensity based on the EM estimated and Kalman filtered Mixture Proportion are plotted
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        rainIntensities : Panda DataFrame containing the rain intensities, with dimensions [n_frames, 2]
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf where the EM estiamted and Kalman filtered rain intensity are compared against each other
    """
    
    rainValues = rainIntensities.values
    rainNames = list(rainIntensities)
    
    plt.figure(1,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    for i in range(rainValues.shape[1]):
        plt.plot(x_axis, rainValues[:,i], label = rainNames[i])
    plt.grid(True)
    plt.xlabel("Frame")
    plt.xlim(np.min(x_axis), np.max(x_axis))
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(outputPath + "Rain_Intensity.pdf", bbox_inches="tight")


def plot_GOF_certainty(x_axis, GOF, threshold=0.06, outputPath = ""):
    """
    Plots the calculated Goodness-Of-Fit value against the number of frames.
    
    Input:
        x_axis : Panda Series containing the frame numbers, with dimensions [n_frames,]
        GOF : Panda Series containing the GOF values, with dimensions [n_frames,]
        threshold : The threshold value of the GOF values used to determine whether the estimated Gaussian distribution fits the observed histogram
        outputPath : File path for where the output pdf should be saved
        
    Output:
        Outputs a single pdf where the GOF value is plotted agains the frame number, with a superimposed line at the threshold value
    """
    
    gofValues = GOF.values
    
    plt.figure(1,(15,5)) #second argument is size of figure in integers
    plt.clf()
    
    plt.plot(x_axis, gofValues, label = GOF.name)
    
    horiz_line_data = np.array([threshold for i in range(len(x_axis))])
    plt.plot(x_axis, horiz_line_data, 'r--', label = "Threshold: {:.2f}".format(threshold)) 
    plt.grid(True)
    plt.xlabel("Frame")
    plt.xlim(np.min(x_axis), np.max(x_axis))
    
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig(outputPath +  "Goodness_Of_Fit.pdf", bbox_inches="tight")


def read_yaml(file):
    """
    Takes a YAML formatted file as input, and returns it as dictionary
    
    Input:
        file : File path to the input file (Assumed to have '%YAML:1.0' as its first line)
        
    Output:
        Returns a python dictionary containing the elements in the YAML input file
    """
    with open(file, 'r') as fi:
        fi.readline()   #Skips the %YAML:1.0 on the first line
        return yaml.load(fi)
    


def analyseBossuCSVData(args):
    """
    Analyses the output of the BossuRainGauge C++ project
    """
    #Load the supplied csv file
    rain_dataframe = pd.read_csv(args["dataFilePath"], sep=";")
    
    # Load the differnet parts of the csv file
    SettingsFile = rain_dataframe[rain_dataframe.columns[0]].values[0]
    InputVideo = rain_dataframe[rain_dataframe.columns[1]].values[0]
    Frames = rain_dataframe[rain_dataframe.columns[2]]
    gaussParams = rain_dataframe[rain_dataframe.columns[3:6]]
    GOF = rain_dataframe[rain_dataframe.columns[6]]
    kalmanGaussParams = rain_dataframe[rain_dataframe.columns[7:10]]
    rainIntensities = rain_dataframe[rain_dataframe.columns[10:12]]
    
    
    #If a settingsFile is found then set the threshold values accordingly
    gofThresh = 0.06
    kalmanGaussSurfaceThresh = 0.35
    if type(SettingsFile) == str and SettingsFile != "":
        settings = read_yaml(SettingsFile)
        kalmanGaussSurfaceThresh = settings["minimumGaussianSurface"]
        gofThresh = settings["maxGoFDifference"]
    
    
    # Create the output directory
    OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
    
    if args["outputPath"] != "":
        OUTPUT_DIR = args["outputPath"]
        
        
    OUTPUT_DIR = os.path.join(OUTPUT_DIR, InputVideo.split(".")[0] + "/")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    
    #Plot the different values
    plot_distribution_params(Frames, gaussParams, kalmanGaussParams, kalmanGaussSurfaceThresh, OUTPUT_DIR)
    plot_rain_intensity(Frames, rainIntensities, OUTPUT_DIR)
    plot_GOF_certainty(Frames, GOF, gofThresh, OUTPUT_DIR)

    


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes the .csv output file from the 'BossuRainGauge' C++ code and plots the different distribution parameters, threholded values and rain intensity agains the frame count")
    ap.add_argument("-dataFilePath", "--dataFilePath", type=str, default = "D://VAP_RainRemoval/BossuRainGauge/Output/testVid2.avi_Results.csv",
                    help="Path to the data file")
    ap.add_argument("-outputPath", "--outputPath", type=str, default = "",
                    help="Path to main output folder. If provided a folder will be made containing the output plots. Else it will be saved in a folder in where the script is placed")
    args = vars(ap.parse_args())
    
    analyseBossuCSVData(args)

