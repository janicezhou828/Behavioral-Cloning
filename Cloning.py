import csv

lines = []
with open ('./Data/Round1/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append (line)

for line in lines:
	source_path = line[0]
	tokens = source_path.split('/')
	print(tokens)
	exit()