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
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Construct the argument parser and parse the arguments

ap = argparse.ArgumentParser()
ap.add_argument("-sc", "--sc", type=int, default=500,
	help="Total count of rain strokes")
ap.add_argument("-gaussMean", "--gaussMean", type=float, default=65.0,
	help="Mean of the Gaussian distribution")
ap.add_argument("-gaussStdDev", "--gaussStdDev", type=float, default=10.0,
	help="Standard Deviation of the Gaussian distribution")
ap.add_argument("-gaussMix", "--gaussMix", type=float, default=0.75,
	help="The ratio of the Gaussian distribution. The ratio of the unifrom distribution is 1 minus this value.")
ap.add_argument("-lengthMean", "--lengthMean", type=float, default=10.0,
	help="Mean length of the rain strokes")
ap.add_argument("-lengthStdDev", "--lengthStdDev", type=float, default=0.2,
	help="Standard Deviation of the rain stroke length")
ap.add_argument("-widthMean", "--widthMean", type=float, default=2.0,
	help="Mean width of the rain strokes")
ap.add_argument("-widthStdDev", "--widthStdDev", type=float, default=0.1,
	help="Standard Deviation of the rain stroke width")
ap.add_argument("-imgHeight", "--imgHeight", type=int, default=640,
	help="Height of the output image")
ap.add_argument("-imgWidth", "--imgWidth", type=int, default=480,
	help="Width of the output image")
ap.add_argument("-outputPath", "--outputPath", type=str, default="",
	help="Path for the output images")
ap.add_argument("-seed", "--seed", type=int, default=None,
	help="Seed for the numpy random number generator")
args = vars(ap.parse_args())



# Assert that the provided arguments are valid

assert args["gaussMean"] >= 0.0 and args["gaussMean"] <= 180.0, "The argument 'gaussMean' should be between 0.0 and 180.0. You supplied {}".format(args["gaussMean"])

assert args["gaussStdDev"] > 0.0, "The argument 'gaussStdDev' should be positive. You supplied {}".format(args["gaussStdDev"])

assert args["gaussMix"] >= 0.0 and args["gaussMix"] <= 1.0, "The argument 'gaussMix' should be a value between 0.0 and 1.0. You supplied {}".format(args["gaussMix"])

assert args["lengthMean"] > 0.0, "The argument 'lengthStdDev' should be positive. You supplied {}".format(args["lengthMean"])

assert args["lengthStdDev"] > 0.0, "The argument 'lengthStdDev' should be positive. You supplied {}".format(args["lengthStdDev"])

assert args["widthMean"] > 0.0, "The argument 'widthMean' should be positive. You supplied {}".format(args["widthMean"])

assert args["widthStdDev"] > 0.0, "The argument 'widthStdDev' should be positive. You supplied {}".format(args["widthStdDev"])

assert args["imgWidth"] > 0 and args["imgHeight"] > 0, "The output image should have a positive image size. You supplied {} by {} pixels".format(args["imgHeight"], args["imgWidth"])

if args["seed"] != None:
    assert args["seed"] >= 0, "The argument 'seed' should be non-negative. You supplied {}".format(args["seed"])
    np.random.seed(args["seed"])



# Setup output directory

OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))

if args["outputPath"] != "":
    OUTPUT_DIR = args["outputPath"]
    
OUTPUT_DIR = os.path.join(OUTPUT_DIR, "Synthetic_Rain_Data")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

    
    
# alcualte the ratio of samples per distribution
sampleCount = args["sc"]
gaussSampleCount = np.ceil(sampleCount*args["gaussMix"]).astype(np.int32)
uniformSampleCount = (sampleCount - gaussSampleCount).astype(np.int32)
    

# Sample the positon, width and length of the rain strokes, as well as their orientations according to the input orientations.
xy_start = np.random.rand(sampleCount, 2)

xy_start[:,0] *= args["imgHeight"]
xy_start[:,1] *= args["imgWidth"]

lengths = np.random.normal(args["lengthMean"], args["lengthStdDev"], size=[sampleCount, ])
widths = np.round(np.random.normal(args["widthMean"], args["widthStdDev"], size=[sampleCount, ])).astype(np.int32)

orientation_gauss = np.random.normal(args["gaussMean"], args["gaussStdDev"], size = [gaussSampleCount, ])
orientation_uniform = np.random.uniform(0, 180, size=[uniformSampleCount,])
orientations = np.concatenate((orientation_gauss, orientation_uniform))
orientations_rad = np.deg2rad(orientations)


# Calcualte endpoint position of the end point of the line. Assumes the line's initital orientation is 0 i.e. y is 0 and the end point is intially at x+length
# Note the inverted orientation is used as to have the strokes oriented counter clockwise
y_init = np.zeros(shape=(sampleCount,))

x_end = xy_start[:,0] - np.multiply(lengths, np.cos(orientations_rad))# - np.multiply(y_init, np.sin(orientations_rad))
y_end = xy_start[:,1] + np.multiply(lengths, np.sin(orientations_rad))# + np.multiply(y_init, np.cos(orientations_rad))

xy_end = np.column_stack((x_end, y_end))


# Inititalize a blank image and draw the lines onto them.
im = Image.new('L', (args["imgHeight"], args["imgWidth"]), 0) 
draw = ImageDraw.Draw(im) 
for i in range(sampleCount):
    draw.line((tuple(xy_start[i].astype(np.int32)), tuple(xy_end[i].astype(np.int32))), fill=255, width = widths[i])
    

# Write the image and histogram to the output directory
filename = "GMix_{}-Mean_{}-Sigma_{}-LengthM_{}-LenghtSD_{}-WidhtM_{}-widthSD_{}-SC_{}".format(args["gaussMix"],args["gaussMean"],args["gaussStdDev"],args["lengthMean"],args["lengthStdDev"],args["widthMean"],args["widthStdDev"],args["sc"])    
filepath = os.path.join(OUTPUT_DIR,filename)

im.save(fp = filepath+".png")

hist = plt.hist(orientations, bins=36, range = (0, 180))  # arguments are passed to np.histogram
plt.xticks(range(0,181,20))
plt.axis([0, 180, 0, max(hist[0])+5])
plt.savefig(filepath+"_hist.pdf", bbox_inches="tight")