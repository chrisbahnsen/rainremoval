import csv
import datetime
import RainGauges as RG
import os
gauges = RG.RainGauges('P:\Private\Traffic safety\Programs\PixelRainGauge\Viborg_Aalborg_Data')

egensevej = RG.Location(57.021092, 10.011279, 'Egensevej-Klarupvej', 1)
recordingInfoFile = 'P:\Private\Traffic safety\Programs\PixelRainGauge\PixelRainGauge\Context and video.csv'

with open(recordingInfoFile) as csvfile:
    reader = csv.reader(csvfile, delimiter=';')

    writefile = os.path.splitext(recordingInfoFile)[0] + '-withPrecipitation.csv'

    with open(writefile, 'wb') as csvWriteFile:
        writer = csv.writer(csvWriteFile)

        firstRow = True
    
        for row in reader:
            if firstRow:
                firstRow = False
                writer.writerow(row)
                continue

            if row[0] is '':
                continue

            lat = float(row[7])
            long = float(row[8])

            rawStartDate = row[9]
            startDate = rawStartDate.split(' ')[0].split('-')
            rawStartTime = row[14]
            startTime = rawStartTime.split(':')
            rawEndTime = row[15]
            endTime = rawEndTime.split(':')
        
            startDateTime = datetime.datetime(int(startDate[2]), int(startDate[1]), int(startDate[0]), int(startTime[0]), int(startTime[1]))
            location = RG.Location(lat, long, row[1], 0)

            measurement = gauges.getNearestRainData(location, startDateTime, endDateTime)
            row[5] = measurement.totalInMm

            writer.writerow(row)
        
