
lemmas = {}
targets = {}
samples_num = 0

with open('generated_samples.tsv','r') as file:
	for line in file:
		samples_num += 1

		samples = line.split('\t')
		if samples[0] not in lemmas.keys():
			lemmas[samples[0]] = 1
		else:
			lemmas[samples[0]] += 1

		if samples[2] not in targets.keys():
			targets[samples[2]] = 1
		else:
			targets[samples[2]] += 1


unique_lemmas = 0
unique_targets = 0

for k,v in lemmas.items():
	if v == 1:
		unique_lemmas += 1

for k,v in targets.items():
	if v == 1:
		unique_targets += 1

# hiszen lemma egyedi
print('teljesen egyedi példa:',unique_lemmas)
# egyedi célszóalak, de lemma ismétlődhet
print('azonos lemma, de egyedi célszóalak:',unique_targets)
print('ismétlődő példák:',samples_num-unique_lemmas-unique_targets)
print('összes példa:',samples_num)