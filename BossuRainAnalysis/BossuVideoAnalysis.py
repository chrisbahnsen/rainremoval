# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:34:20 2018

@author: Joakim
"""
import os
import subprocess
import argparse
from BossuCSVAnalysis import analyseBossuCSVData

def analyseVideos(args):
    """
    Takes a txt file as input where the path to differetn videos which has to be analyzed with the BossuRainGauge C++ project is defined.
    """    
	
    filePathNames = []
    videos = []
    with  open(args["videoPathFile"], "r") as f:
        videos = f.readlines()
        for i in range(len(videos)):
                videos[i] = videos[i].strip("\n")
                fileName = videos[i].split("/")[-1]
                filePath = videos[i][:-len(fileName)]
                filePathNames.append((filePath, fileName))
                
    
    exe_args_base = []
    
    exe_args_base.append(args["exeFilePath"])
    exe_args_base.append("--d="+str(args["debugFlag"]))
    exe_args_base.append("--v="+str(args["verboseFlag"]))
    exe_args_base.append("--i="+str(args["saveImageFlag"]))
    exe_args_base.append("--s="+str(args["saveSettingsFlag"]))
    
    
    # Setup output directory
    OUTPUT_DIR_BASE = os.path.abspath(os.path.dirname(__file__))
    if args["outputPath"] != "":
        OUTPUT_DIR_BASE = args["outputPath"]
        
    
    
    for file in filePathNames:
    
        OUTPUT_DIR = os.path.join(OUTPUT_DIR_BASE, file[1])
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
            
        exe_args = exe_args_base.copy()
        exe_args.append("--of="+OUTPUT_DIR)
        exe_args.append("--fileName=" + file[1])
        exe_args.append("--filePath=" + file[0])
        subprocess.call(exe_args,  creationflags=subprocess.CREATE_NEW_CONSOLE)
        
        CSV_analysis = {}
        CSV_analysis["dataFilePath"] = OUTPUT_DIR+"/"+file[1]+"_Results.csv"
        CSV_analysis["outputPath"] =  OUTPUT_DIR
        analyseBossuCSVData(CSV_analysis)
        
        




if __name__ == "__main__":
    ap = argparse.ArgumentParser(
            description = "Takes a .txt file where each line is the filepath to a video, and then runs the BossuRainGauge C++ project on it")
    ap.add_argument("-videoPathFile", "--videoPathFile", type=str, required = True,
                    help="Path to the txt data file, which contains all filepaths to the videos. The filepaths should be the complete path to the file, including the filename itself")
    ap.add_argument("-exeFilePath", "--exeFilePath", type=str, required = True,
                    help="Path to the exe file of the BossuRainGauge C++ project. Provide the entire filepath including the filename")
    ap.add_argument("-outputPath", "--outputPath", type=str, required = True,
                    help="Path to main output folder. If provided a folder will be made containing the output .csv and settings file. Else it will be saved in a folder in where the script is placed. The path given should be the entire filepath")
    ap.add_argument("-debugFlag", "--debugFlag", type=int, default = 0,
                    help="States whether the debug flag should be set or not. Not set if equal to 0")
    ap.add_argument("-verboseFlag", "--verboseFlag", type=int, default = 0,
                    help="States whether the verbose flag should be set or not. Not set if equal to 0")
    ap.add_argument("-saveImageFlag", "--saveImageFlag", type=int, default = 0,
                    help="States whether the saveImage flag should be set or not. Not set if equal to 0")
    ap.add_argument("-saveSettingsFlag", "--saveSettingsFlag", type=int, default = 1,
                    help="States whether the saveSettings flag should be set or not. Not set if equal to 0")
    
    args = vars(ap.parse_args())
    
    analyseVideos(args)


