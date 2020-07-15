import csv
import cv2

lines = []
with open ('./Data/Round1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append (line)
images = []
measurements = []

for line in lines:
	source_path = line[0]
	tokens = source_path.split('\\')
	filename = tokens[-1]
	local_path = "./Data/" + filename
	image = cv2.imread(local_path)
	images.append(image)
	measurement = line[3]
	measurements.append(measurement)
	
print (len(images))
print (len(measurements))


