# MIT License
# 
# Copyright(c) 2017 Aalborg University
# Chris H. Bahnsen, June 2017
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

import datetime
import math
import os


def toRadians(number):
    return number * math.pi / 180

class Location(object):
    """Represents a physical location, given by the latitude
       and longitude, WGS84
    """

    def __init__(self, lat, long, name, id):
        self.lat = lat
        self.long = long
        self.name = name
        self.id = id

    def measureDistance(self, location):
        # Code converted from JavaScript, original source
        # http://www.movable-type.co.uk/scripts/latlong.html

        R = 6371e3; # metres
        φ1 = toRadians(self.lat);
        φ2 = toRadians(location.lat);
        Δφ = toRadians(location.lat-self.lat);
        Δλ = toRadians(location.long-self.long);

        a = (math.sin(Δφ/2) * math.sin(Δφ/2) +
            math.cos(φ1) * math.cos(φ2) *
            math.sin(Δλ/2) * math.sin(Δλ/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

        d = R * c;

class Recording(object):
    """Represents a rain gauge recording with a
       physical location and start and end time
    """

    def __init__(self, location, startTime, endTime):
        self.location = location
        self.startTime = startTime
        self.endTime = endTime


class RainGauges(object):
    """Manages a list of data from real rain gauges 
       and provides an interface for retrieving the rain 
       data from a particular location and time span
        
    """

    def __init__(self, rainGaugeFolder):
        
        self.rainGaugeFolder = rainGaugeFolder;
        self.rainGauges = {}

        self.nearestRainGauge = None;

        self.rainGaugeRecordings = []
        self.rainGaugeLocations = {}



        __inspectMeasurementFiles()

    def __inspectMeasurementFiles(self):
        """ Inspect the rain measurements in the folder provided in the 
            initialisation and create an object that provides easy lookup
            of the location and time span of the measurements"""

        files = [file for file in os.listdir(self.rainGaugeFolder)
                 if os.path.isfile(os.path.join(self.rainGaugeFolder, file))]


        # Get information of the location of the rain gauges
        for file in files:
            if 'GaugeInfo' in file:
                with open(file) as f:
                    lines = f.readlines()

                    for idx in range(1, len(lines)):
                        entries = lines[idx].split(',')

                        if len(entries) < 8:
                            continue

                        id = int(entries[0])
                        name = entries[1]
                        lat = entries[6]
                        long = entries[7]

                        gauge = Location(lat, long, name, id)
                        self.gaugeLocations[id] = gauge;
        
        
        # Get the information of the duration of the rain gauge recordings 
        # and couple the location of the gauge with the gauge id
        for file in files:
            if '.txt' in file and 'GaugeInfo' not in file:
                fileInfoParts = file.split('-');

                if len(fileInfoParts) >= 7:
                    startTime = datetime.datetime(fileInfoParts[3], fileInfoParts[2], fileInfoParts[1]);
                    endTime = datetime.datetime(fileInfoParts[6], fileInfoParts[5], fileInfoParts[4]);

                    with open(file) as f:
                        # Just read the first line - it contains the information
                        # we need for now
                        entries = f.readline().split(',')

                        if len(entries) > 1:
                            for idx in range(1, len(entries)):
                                id = entries[idx]
                                location = self.gaugeLocations[id]

                                recording = Recording(location, startTime, endTime)  
                                self.rainGaugeRecordings.append(recording)



    def __getGaugeInfo(file):
        with open(file) as f:
            for line in f:
                columns = line.split(',')


    
    def getNearestRainData(self, location, startTime, endTime):

        # Find the closest rain gauge to the location listed
        # In order to quality, the rain gauge must have a recording
        # within the specified start and end time

        shortestDistance = 10000; # If we can't find a rain gauge within 10 km, we have failed
        bestLocation = None

        for recording in self.rainGaugeRecordings:


            distance = recording.location.measureDistance(location)
            


        return


     