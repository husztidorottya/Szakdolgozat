import tensorflow as tf 
import numpy as np
import helpers
import operator
import random
import argparse

# create (source morphological tags + target morphological tags + source/target word) sequence
def create_sequence(data_line_, parameters):
    sequence = []
    
    if len(data_line_) == 2:
        # source and target morphological tags are appended only to the input
        for i in data_line_[1]:
            sequence.append(i)
        
    # append beginning of the input
    sequence.append(parameters.BOS)

    for i in data_line_[0]:
        sequence.append(i)

    sequence.append(parameters.EOS)
             
    return sequence


# encoding input data
def encoding(data, alphabet_and_morph_tags):
	coded_word = []
	for character in data:
		index = alphabet_and_morph_tags.setdefault(character, len(alphabet_and_morph_tags) + 3)
		coded_word.append(index)
        
	return coded_word

# converts vector of numbers back to characters
def convert_back_tostring(parameters, data, alphabet_and_morph_tags):
	word = ''
	for element in data:
		for char in element:
			if char != parameters.EOS and char != parameters.BOS and char != parameters.PAD:
				# https://stackoverflow.com/questions/23295315/get-key-by-value-dict-python
				word = word + (list(alphabet_and_morph_tags.keys())[list(alphabet_and_morph_tags.values()).index(char)])

	return word


# class stores model parameters
class Parameters:
   def __init__(self, BOS, EOS, PAD, character_changing_num, batches_in_epoch, input_embedding_size, neuron_num, epoch):
      self.BOS = BOS
      self.EOS = EOS
      self.PAD = PAD
      self.character_changing_num = character_changing_num
      self.batches_in_epoch = batches_in_epoch
      self.input_embedding_size = input_embedding_size
      self.neuron_num = neuron_num
      self.epoch = epoch


def main():
	# GLOBAL CONTANTS
	parameters = Parameters(2, 1, 0, 10, 100, 300, 100, 100)

	alphabet_and_morph_tags = dict()

	# read vocab
	with open('alphabet_and_morph_tags.tsv','r') as inputfile:
		for line in inputfile:
			line_data = line.strip('\n').split('\t')
			alphabet_and_morph_tags[line_data[0]] = int(line_data[1])

	samplenum = 0
	sampleright = 0
	accuracy = 0

	# ---------------
	# read from command line pl. "dobókocka N;IN+ABL;PL"
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument('reinflection')
	args = parser.parse_args()
	split_input = args.reinflection.split(' ')
	word = split_input[0]
	tags = split_input[1].split(';')
	#word = 'őrbódé'
	#tags = ['N', 'IN+ABL', 'PL']
	target = 'őrbódékból'
	#word = 'dobókocka'
	#tags = ['N', 'DAT', 'PL']
	#target = 'dobókockáknak'
	'''

	with open('task1_test.tsv','r') as input_file:
		source_data = []
		target_data = []
        # read it line-by-line
		for line in input_file:
			data_line_ = line.strip('\n').split('\t')
			source_data.append([data_line_[0],data_line_[2]])
			target_data.append(data_line_[1])
 
	for wordnum in range(len(source_data)):
 		word = source_data[wordnum][0]
 		tags = source_data[wordnum][1].split(';')
 		data = []
 		data.append(encoding(word, alphabet_and_morph_tags))
 		data.append(encoding(tags, alphabet_and_morph_tags))

 		sdata = []
 		sdata.append(create_sequence(data, parameters))

 		target = target_data[wordnum]
 		data = []
 		data = (encoding(target, alphabet_and_morph_tags))
 		tdata = []
 		tdata.append(create_sequence([data], parameters))

 		tf.reset_default_graph() 

 		with tf.Session() as sess:
 			sess.run(tf.global_variables_initializer())
 			saver = tf.train.import_meta_graph('trained_model_100_adam_identity.meta')
 			saver.restore(sess, tf.train.latest_checkpoint('./'))
 			max_alphabet_and_morph_tags = alphabet_and_morph_tags[max(alphabet_and_morph_tags.items(), key=operator.itemgetter(1))[0]]

 			vocab_size = max_alphabet_and_morph_tags + 1
 			encoder_hidden_units = parameters.neuron_num 
 			decoder_hidden_units = encoder_hidden_units * 2 
 			encoder_inputs = sess.graph.get_tensor_by_name("encoder_inputs:0")
 			encoder_inputs_length = sess.graph.get_tensor_by_name("encoder_inputs_length:0")
 			embeddings = sess.graph.get_tensor_by_name('embeddings:0')
 			encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
 			encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)
 			no_suffix = []
 			for v in tf.trainable_variables():
 				no_suffix.append(v)

 			no_suffix_rnn = []
 			for v in no_suffix:
 				if v.name.find('rnn/gru_cell/') != -1:
 					no_suffix_rnn.append(v)

 			((encoder_fw_outputs,
 				encoder_bw_outputs),
 			(encoder_fw_final_state,
 				encoder_bw_final_state)) = (
 			tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
 				cell_bw=encoder_cell,
 				inputs=encoder_inputs_embedded,
 				sequence_length=encoder_inputs_length,
 				dtype=tf.float32, time_major=True))

 			suffix = []
 			for v in tf.trainable_variables():
 				if v.name.find('_1') != -1:
 					suffix.append(v)
 			for i in range(len(suffix)):
 				assign_op = suffix[i].assign(no_suffix[i+1])
 				sess.run(assign_op)
 			j = 0
 			for i in range(len(tf.trainable_variables())):
 				if tf.trainable_variables()[i].name.find('_1') != -1:
 					sess.run(tf.trainable_variables()[i].assign(suffix[j]))	
 					j += 1
 			encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
 			encoder_final_state = tf.concat(
 				(encoder_fw_final_state, encoder_bw_final_state), 1)
 			decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

 			encoder_max_time, parameters.batch_size = tf.unstack(tf.shape(encoder_inputs))
 			decoder_lengths = encoder_inputs_length + parameters.character_changing_num
 			W = sess.graph.get_tensor_by_name("W:0")
 			b = sess.graph.get_tensor_by_name("b:0")
 			assert parameters.EOS == 1 and parameters.PAD == 0 and parameters.BOS == 2
 			bos_time_slice = tf.fill([parameters.batch_size], 2, name='BOS')
 			eos_time_slice = tf.ones([parameters.batch_size], dtype=tf.int32, name='EOS')
 			pad_time_slice = tf.zeros([parameters.batch_size], dtype=tf.int32, name='PAD')

 			parameters.batch_size = 20

 			bos_step_embedded = tf.nn.embedding_lookup(embeddings, bos_time_slice)
 			eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
 			pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

 			def loop_fn_initial():
 				initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
 				initial_input = bos_step_embedded
 				initial_cell_state = encoder_final_state
 				initial_cell_output = None
 				initial_loop_state = None  # we don't need to pass any additional information
 				return (initial_elements_finished,
 					initial_input,
 					initial_cell_state,
 					initial_cell_output,
 					initial_loop_state)

 			def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
 				def get_next_input():
 					output_logits = tf.add(tf.matmul(previous_output, W), b)
 					prediction = tf.argmax(output_logits, axis=1)
 					next_input = tf.nn.embedding_lookup(embeddings, prediction)

 					return next_input

 				elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
 				finished = tf.reduce_all(elements_finished) # -> boolean scalar
 				input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
 				state = previous_state
 				output = previous_output
 				loop_state = None

 				return (elements_finished, 
 					input,
 					state,
 					output,
 					loop_state)

 			def loop_fn(time, previous_output, previous_state, previous_loop_state):
 				if previous_state is None:    # time == 0
 					assert previous_output is None and previous_state is None
 					return loop_fn_initial()
 				else:
 					return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

 			# EZ A SZAR MEGINT INITIALIZÁLATLAN SUFFIXES VERZIÓT HOZ LÉTRE (MEGINT FELÜL KELL CSAPNI
 			decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
 			# overwrite rnn trainable_variables with suffix with the value of non-suffix ones
 			j = 0
 			for i in range(len(tf.trainable_variables())):
 				if tf.trainable_variables()[i].name.find('rnn/gru_cell/') != -1:
 					if tf.trainable_variables()[i].name.find('_1') != -1:
 						sess.run(tf.trainable_variables()[i].assign(no_suffix_rnn[j]))
 						j +=1

 			decoder_outputs = decoder_outputs_ta.stack()

 			decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
 			decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
 			decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
 			decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
 			decoder_prediction = tf.argmax(decoder_logits, 2)
 			encoder_inputs_, encoder_input_lengths_ = helpers.batch(sdata)
 			predict_ = sess.run(decoder_prediction, feed_dict={
 				encoder_inputs: encoder_inputs_,
 				encoder_inputs_length: encoder_input_lengths_
 			})

 			samplenum += 1
 			if convert_back_tostring(parameters, predict_.T, alphabet_and_morph_tags) == target:
 				sampleright += 1

 			print(samplenum)

	print('accuracy:',(sampleright/samplenum))

if __name__ == '__main__':
    main()


