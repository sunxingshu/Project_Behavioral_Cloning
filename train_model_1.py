import csv

lines = []

with open('./Sample_Data/driving_log.csv') as csvfile:
    temp = csv.reader(csvfile)
    for line in temp:
        print(line)
        lines.append(line)