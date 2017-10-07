import argparse

parser = argparse.ArgumentParser()
parser.add_argument('exp_num')
args = parser.parse_args()

min_loss_file = ""
min_loss = 1000

for i in range(0,int(args.exp_num)+1):
	with open('parameters/trained_model' + str(i) + '_parameters.tsv', 'r') as file:
		line_num = 0
		for line in file:
			if line_num == 7:
				splited_line = line.strip('\n').split('\t')
				if float(splited_line[1]) < min_loss:
					min_loss = float(splited_line[1])
					min_loss_file = 'trained_model' + str(i) + '_parameters.tsv'
			line_num += 1

print('min_loss:',min_loss)
print('min_loss_file:',min_loss_file)