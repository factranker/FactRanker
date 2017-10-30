from sklearn import metrics

threshold = 0.0
while threshold < 1.0:
	threshold += 0.05
	with open('/home/bt1/13CS10037/btp_final_from_server/codes/liar_dataset/buster_all.tsv') as buster_file:
		predicted = []
		actual = []
		buster_file.readline()
		for line in buster_file:
			line = line.strip().split('\t')
			score = float(line[1])
			klass = int(line[2])
			if score>=threshold:
				predicted.append(1)
			else:
				predicted.append(0)
			actual.append(klass)
		report = metrics.classification_report(actual, predicted)
		print("Threshold:", threshold)
		print(report)
		print()