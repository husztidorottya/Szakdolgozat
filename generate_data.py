import argparse

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

# ebbe kerulnek bele a generalt harmasok
data = []

match_statistic = {}

# generalt adathalmaz beolvasasa
with open(args.filename,'r') as file:
	for line in file:
		# csak akkor vizsgalja, ha nem ures sor vagy elvalaszto sor vagy nem szam
		if line != '\n' and line.find('OTHER') == -1 and line.find('Num|Digit') == -1:	
			# feldarabolja tabok menten a peldat
			splited_line = line.strip('\n').split('\t')
			# eltarolja egyezes tipusat es noveli eggyel az egyezes szamat
			if splited_line[5] not in match_statistic.keys():
				match_statistic[splited_line[5]] = 1
			else:
				match_statistic[splited_line[5]] += 1
			# azokat a sorokat rakja csak bele, mely szoalakvaltozassal jar, azaz toldalekolas tortenik
			if splited_line[6].find(' ') != -1:
				# pelda harmasok osszeallitasa
				sample = []
<<<<<<< HEAD
				# azert kell, hogy ne legyen forras/cel nagybetus mig cel/forras kisbetus
				if (splited_line[1][0].isupper()==True and splited_line[0][0].isupper()==False):
					splited_line[1] = splited_line[1].lower()
				if (splited_line[1][0].isupper()==False and splited_line[0][0].isupper()==True):
					splited_line[0] = splited_line[0].lower()

				sample.append(splited_line[1])
				# tagek vesszovel elvalasztott formajat eloallitja
				tags = splited_line[7].replace(' ',',')
				sample.append(tags)
=======
				sample.append(splited_line[1])
				# tagek vesszovel elvalasztott formajat eloallitja
				#tags = splited_line[7].replace(' ',',').replace('.',',').replace('[','').replace(']','').replace('/','')
				sample.append(splited_line[7])
>>>>>>> d53f7aafc98b96fbcad64dd7e9231ac6aa857694
				sample.append(splited_line[0])
				data.append(sample)


# statisztika keszites illeszkedes eloszlasarol
with open('adathalmazStatisztika.tsv','w') as statfile:
	statfile.write('osszes elem:\t{}\n'.format(len(data)))
	for k,v in match_statistic.items():
		statfile.write('{}:\t{}\n'.format(k,v))

#for i in range(10):
#	print(data[i])

# append new samples to generated sample' file
with open('generated_samples.tsv','a') as outputfile:
	for sample in data:
		for i in range(0,len(sample)):
			if i == 2:
				outputfile.write('{}\n'.format(sample[i]))
			else:
				outputfile.write('{}\t'.format(sample[i]))



