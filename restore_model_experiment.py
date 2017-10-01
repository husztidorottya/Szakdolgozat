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


	# ---------------
	# read from command line pl. "dobókocka N;IN+ABL;PL"
	parser = argparse.ArgumentParser()
	parser.add_argument('reinflection')
	parser.add_argument('trained_model')
	args = parser.parse_args()
	split_input = args.reinflection.split(' ')
	word = split_input[0]
	tags = split_input[1].split(';')
	
	data = []
	data.append(encoding(word, alphabet_and_morph_tags))
	data.append(encoding(tags, alphabet_and_morph_tags))

	sdata = []
	sdata.append(create_sequence(data, parameters))
	# -----------------

	tf.reset_default_graph() 

	# we need to load the trained model parameters
	with open(args.trained_model[:args.trained_model.find('.')] + '_parameters.tsv', 'r') as input_parameters:
		line_num = 0
		for line in input_parameters:
			param_line = line.strip('\n').split('\t')
			if line_num == 0:
				parameters.input_embedding_size = int(param_line[1])
			if line_num == 1:
				parameters.neuron_num = int(param_line[1])
			if line_num == 2:
				parameters.epoch = int(param_line[1])
			if line_num == 3:
				parameters.early_stopping_delta = float(param_line[1])
			if line_num == 4:
				parameters.early_stopping_patience = int(param_line[1])
			if line_num == 5:
				parameters.batch_size = param_line[1]
				print(param_line[1])
			if line_num == 6:
				parameters.learning_rate = float(param_line[1])
			line_num += 1

  
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		#First let's load meta graph and restore 
		saver = tf.train.import_meta_graph(args.trained_model)
		saver.restore(sess, './' + args.trained_model[:args.trained_model.find('.')])

		# get max value of encoded forms
		max_alphabet_and_morph_tags = alphabet_and_morph_tags[max(alphabet_and_morph_tags.items(), key=operator.itemgetter(1))[0]]

		# calculate vocab_size
		vocab_size = max_alphabet_and_morph_tags + 1

		# num neurons
		encoder_hidden_units = parameters.neuron_num 
		# in original paper, they used same number of neurons for both encoder
		# and decoder, but we use twice as many so decoded output is different, the target value is the original input 
		#in this example
		decoder_hidden_units = encoder_hidden_units * 2 

		encoder_inputs = sess.graph.get_tensor_by_name("encoder_inputs:0")
		# contains the lengths for each of the sequence in the batch, we will pad so all the same
		# if you don't want to pad, check out dynamic memory networks to input variable length sequences
		encoder_inputs_length = sess.graph.get_tensor_by_name("encoder_inputs_length:0")

		# randomly initialized embedding matrrix that can fit input sequence
		# used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
		# reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
		embeddings = sess.graph.get_tensor_by_name('embeddings:0')
		#embeddings = tf.Variable(tf.eye(vocab_size, input_embedding_size), dtype='float32')


		# this thing could get huge in a real world application
		encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

		# define encoder
		encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

		# save trainable_variables without suffix
		no_suffix = []
		for v in tf.trainable_variables():
			no_suffix.append(v)

		# save trainable_variables for rnn without suffix
		no_suffix_rnn = []
		for v in no_suffix:
			if v.name.find('rnn/gru_cell/') != -1:
				no_suffix_rnn.append(v)

	
		# EZ A SZAR LÉTREHOZ EGY SUFFIXES INITIALITÁLATLAN VÁLTOZATOT, FELÜL KELL CSAPNI
		# define bidirectionel function of encoder (backpropagation)
		((encoder_fw_outputs,
		encoder_bw_outputs),
		(encoder_fw_final_state,
		encoder_bw_final_state)) = (
        	tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)
		)

		# ITT JÖN AZ ELSŐ HIBA, HOGY INITIALIZÁLATLAN weights_1, biases_1
		#print(sess.run(tf.report_uninitialized_variables()))
	
		# save trainable_variables with suffix (created by bidirectional_dynamic_rnn function)
		suffix = []
		for v in tf.trainable_variables():
			if v.name.find('_1') != -1:
				suffix.append(v)
		# overwrite suffix variables with value of non-suffix ones
		for i in range(len(suffix)):
			assign_op = suffix[i].assign(no_suffix[i+1])
			sess.run(assign_op)
		# overwrite suffix variables with value of non-suffix ones at tf.trainable_variables
		j = 0
		for i in range(len(tf.trainable_variables())):
			if tf.trainable_variables()[i].name.find('_1') != -1:
				sess.run(tf.trainable_variables()[i].assign(suffix[j]))	
				j += 1


		#Concatenates tensors along one dimension.
		encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

		# because by GRUCells the state is a Tensor, not a Tuple like by LSTMCells
		encoder_final_state = tf.concat(
        	(encoder_fw_final_state, encoder_bw_final_state), 1)


		decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

		#we could print this, won't need
		encoder_max_time, parameters.batch_size = tf.unstack(tf.shape(encoder_inputs))
		decoder_lengths = encoder_inputs_length + parameters.character_changing_num
		# +(character_changing_num-1) additional steps, +1 leading <EOS> token for decoder inputs

		#manually specifying since we are going to implement attention details for the decoder in a sec
		#weights
		W = sess.graph.get_tensor_by_name("W:0")
		b = sess.graph.get_tensor_by_name("b:0")

		#create padded inputs for the decoder from the word embeddings
		#were telling the program to test a condition, and trigger an error if the condition is false.
		assert parameters.EOS == 1 and parameters.PAD == 0 and parameters.BOS == 2

		bos_time_slice = tf.fill([parameters.batch_size], 2, name='BOS')
		eos_time_slice = tf.ones([parameters.batch_size], dtype=tf.int32, name='EOS')
		pad_time_slice = tf.zeros([parameters.batch_size], dtype=tf.int32, name='PAD')

		# send 20 sequences into encoder at one time
		parameters.batch_size = 20

		#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
		bos_step_embedded = tf.nn.embedding_lookup(embeddings, bos_time_slice)
		eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
		pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
    
		#manually specifying loop function through time - to get initial cell state and input to RNN
		#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

		#we define and return these values, no operations occur here
		def loop_fn_initial():
			initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    		#end of sentence
    		# -------------
    		#initial_input = eos_step_embedded
			initial_input = bos_step_embedded
    		# -------------
    		#last time steps cell state
			initial_cell_state = encoder_final_state
    		#none
			initial_cell_output = None
    		#none
			initial_loop_state = None  # we don't need to pass any additional information
			return (initial_elements_finished,
            	initial_input,
            	initial_cell_state,
            	initial_cell_output,
            	initial_loop_state)


		#attention mechanism --choose which previously generated token to pass as input in the next timestep
		def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):

			def get_next_input():
        		#dot product between previous ouput and weights, then + biases
				output_logits = tf.add(tf.matmul(previous_output, W), b)
        		#Logits simply means that the function operates on the unscaled output of 
        		#earlier layers and that the relative scale to understand the units is linear. 
        		#It means, in particular, the sum of the inputs may not equal 1, that the values are not probabilities 
        		#(you might have an input of 5).
        		#prediction value at current time step
        
        		#Returns the index with the largest value across axes of a tensor.
				prediction = tf.argmax(output_logits, axis=1)
        		#embed prediction for the next input
				next_input = tf.nn.embedding_lookup(embeddings, prediction)
            
				return next_input
    
    
			elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended
    
    		#Computes the "logical and" of elements across dimensions of a tensor.
			finished = tf.reduce_all(elements_finished) # -> boolean scalar
    		#Return either fn1() or fn2() based on the boolean predicate pred.
			input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    
    		#set previous to current
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

		#Unpacks the given dimension of a rank-R tensor into rank-(R-1) tensors.
		#reduces dimensionality
		decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
		#flettened output tensor
		decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))
		#pass flattened tensor through decoder
		decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
		#prediction vals
		decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))

		#final prediction
		decoder_prediction = tf.argmax(decoder_logits, 2)

		#print(sess.run(tf.report_uninitialized_variables()))
		#sess.run(tf.initialize_variables([v for v in tf.all_variables() if v.name.startswith("local")]))

		encoder_inputs_, encoder_input_lengths_ = helpers.batch(sdata)
    
		predict_ = sess.run(decoder_prediction, feed_dict={
    		encoder_inputs: encoder_inputs_,
    		encoder_inputs_length: encoder_input_lengths_
		})

		#print('expected:',target)
		print('predicted:',convert_back_tostring(parameters, predict_.T, alphabet_and_morph_tags))

		#print('elvart:',tdata)
		#print('predict:',predict_.T)

if __name__ == '__main__':
    main()


