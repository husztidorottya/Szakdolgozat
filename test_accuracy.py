import tensorflow as tf 
import numpy as np
import helpers
import operator
import argparse

# handle sigm2016 tasks as input
def sigm_task_2016(data_line_, item, alphabet_and_morph_tags):
    coded_word = []
    # task1
    if len(data_line_) == 3:
        if item == 1:
            # split morphological tags
            tags = data_line_[item].split(',')
                
            coded_word = encoding(tags, coded_word, alphabet_and_morph_tags)
        else:
            # encode source and target word
            coded_word = encoding(data_line_[item], coded_word, alphabet_and_morph_tags)
    # task2
    else:
        if item == 0 or item == 2:
            # split morphological tags
            tags = data_line_[item].split(',')
                
            coded_word = encoding(tags, coded_word, alphabet_and_morph_tags)
        else:
            # encode source and target word
            coded_word = encoding(data_line_[item], coded_word, alphabet_and_morph_tags)

    return coded_word


# handle sigm2017 tasks as input
def sigm_task_2017(data_line_, item, alphabet_and_morph_tags):
    coded_word = []

    # task1
    if item == 2:
        # split morphological tags
        tags = data_line_[item].split(';')
                
        coded_word = encoding(tags, coded_word, alphabet_and_morph_tags)
    else:
            # encode source and target word
            coded_word = encoding(data_line_[item], coded_word, alphabet_and_morph_tags)

    return coded_word


# create sequence for input or output
def create_sequence(data_line_, issource, parameters, is2016):
    sequence = []

    if is2016:
        if issource:
            # task2
            if len(data_line_) == 4:
                # source morph tags
                for i in data_line_[0]:
                    sequence.append(i)
                # target morph tags
                for i in data_line_[2]:
                    sequence.append(i)
            # task1
            else:
                for i in data_line_[1]:
                    sequence.append(i)
            
            sequence.append(parameters.SOS)

            # append source word
            for i in data_line_[len(data_line_)-3]:
                sequence.append(i)

            sequence.append(parameters.EOS)
        else:
            sequence.append(parameters.SOS)

            # append target word
            for i in data_line_[len(data_line_)-1]:
                sequence.append(i)
    else:
        if issource:
            # morphological tags are appended only to the input
            for i in data_line_[2]:
                sequence.append(i)
        
            # append beginning of the input
            sequence.append(parameters.SOS)

            for i in data_line_[0]:
                sequence.append(i)

            sequence.append(parameters.EOS)
        else:
            sequence.append(parameters.SOS)
            # append target word
            for i in data_line_[1]:
                sequence.append(i)
             
    return sequence


# encoding input data
def encoding(data, coded_word, alphabet_and_morph_tags):
    for character in data:
        index = alphabet_and_morph_tags.setdefault(character, len(alphabet_and_morph_tags) + 3)
        coded_word.append(index)
        
    return coded_word


# read, split and encode input data
def read_split_encode_data(filename, alphabet_and_morph_tags, parameters):
    with open(filename,'r') as input_file:
        source_data = []
        target_data = []
        idx = 0
        # read it line-by-line
        for line in input_file:
            data_line_ = line.strip('\n').split('\t')

            # generated data's format is similar to sigm2016 task1
            is2016 = (line.find('=') != -1) or (line.find('[') != -1) 
            # encode words into vector of ints 
            for item in range(0,len(data_line_)):         
                # contains encoded form of word
                coded_word = []
            
                # check if input is sigm2016 or sigm2017
                if is2016:
                    # sigm2016
                    coded_word = sigm_task_2016(data_line_, item, alphabet_and_morph_tags)
                # sigm2017 input
                else:
                    coded_word = sigm_task_2017(data_line_, item, alphabet_and_morph_tags)

                # store encoded form
                data_line_[item] = coded_word

            # store encoder input - morph tags + source word
            source_data.append(create_sequence(data_line_, True, parameters, is2016))
            
            # store decoder expected outputs - target word
            target_data.append(create_sequence(data_line_, False, parameters, is2016))
            
            # stores line number (needed for shuffle) - reference for the target_data
            idx += 1

    return source_data, target_data


# feeds the encoder with the next batch size sequences
def next_feed(source_batches, target_batches, encoder_inputs, encoder_inputs_length, decoder_targets, parameters):
        # get transpose of source_batches[batch_num]
	encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_batches)
    
        # get max input sequence length
	max_input_length = max(encoder_input_lengths_)
    
        # target word is max character_changing_num character longer than source word 
        # get transpose of target_batches[i] and put an EOF and PAD at the end
	decoder_targets_, _ = helpers.batch(
            [(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in target_batches]
        )
   
	return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }


# class stores model's parameters
class Parameters:
   def __init__(self, SOS, EOS, PAD, character_changing_num, input_embedding_size, neuron_num, epoch, delta, patience, batch_size, learning_rate):
      self.SOS = SOS
      self.EOS = EOS
      self.PAD = PAD
      self.character_changing_num = character_changing_num
      self.input_embedding_size = input_embedding_size
      self.neuron_num = neuron_num
      self.epoch = epoch
      self.early_stopping_delta = delta
      self.early_stopping_patience = patience
      self.batch_size = batch_size
      self.learning_rate = learning_rate


def main():
	parameters = Parameters(2, 1, 0, 10, 300, 100, 100, 0.001, 5, 20, 0.001)

	# read test filename and trained model's name from command line
	parser = argparse.ArgumentParser()
	parser.add_argument('filename')
	parser.add_argument('trained_model')
	args = parser.parse_args()

	alphabet_and_morph_tags = dict()

	# read vocab
	with open('alphabet_and_morph_tags.tsv','r') as inputfile:
		for line in inputfile:
			line_data = line.strip('\n').split('\t')
			alphabet_and_morph_tags[line_data[0]] = int(line_data[1])

	# we need to load the trained model's parameters
	with open('parameters/' + args.trained_model + '_parameters.tsv', 'r') as input_parameters:
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
			if line_num == 6:
				parameters.learning_rate = float(param_line[1])
			line_num += 1
	
	# read and encode test file data
	source_data, target_data = read_split_encode_data(args.filename, alphabet_and_morph_tags, parameters)

	tf.reset_default_graph() 

	# define the session 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		# first load meta graph and restore 
		saver = tf.train.import_meta_graph('trained_models/' + args.trained_model + '/' + args.trained_model + '.meta')
		saver.restore(sess, tf.train.latest_checkpoint('trained_models/' + args.trained_model + '/'))

		# get max value of encoded forms
		max_alphabet_and_morph_tags = alphabet_and_morph_tags[max(alphabet_and_morph_tags.items(), key=operator.itemgetter(1))[0]]

		# calculate vocab_size
		vocab_size = max_alphabet_and_morph_tags + 1

		# num neurons
		encoder_hidden_units = parameters.neuron_num 
		# in original paper, they used same number of neurons for both encoder
		# and decoder, but we use twice as many so decoded output is different
		decoder_hidden_units = encoder_hidden_units * 2 

		encoder_inputs = sess.graph.get_tensor_by_name("encoder_inputs:0")
		# contains the lengths for each of the sequence in the batch, we will pad so all the same
		encoder_inputs_length = sess.graph.get_tensor_by_name("encoder_inputs_length:0")
		decoder_targets = sess.graph.get_tensor_by_name("decoder_targets:0")

		# used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
		# reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
		embeddings = sess.graph.get_tensor_by_name('embeddings:0')

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
		
		# define bidirectionel function of encoder (backpropagation)
		# creates uninitialized suffix variables, we have to overwrite them
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

		# define decoder
		decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

		#we could print this, won't need
		encoder_max_time, parameters.batch_size = tf.unstack(tf.shape(encoder_inputs))
		# (character_changing_num-1) additional steps, +1 leading <EOS> token for decoder inputs
		decoder_lengths = encoder_inputs_length + parameters.character_changing_num

		#manually specifying since we are going to implement attention details for the decoder in a sec
		# weights and biases
		W = sess.graph.get_tensor_by_name("W:0")
		b = sess.graph.get_tensor_by_name("b:0")

		#create padded inputs for the decoder from the word embeddings
		#were telling the program to test a condition, and trigger an error if the condition is false.
		assert parameters.EOS == 1 and parameters.PAD == 0 and parameters.SOS == 2

		sos_time_slice = tf.fill([parameters.batch_size], 2, name='SOS')
		eos_time_slice = tf.ones([parameters.batch_size], dtype=tf.int32, name='EOS')
		pad_time_slice = tf.zeros([parameters.batch_size], dtype=tf.int32, name='PAD')

		# send 20 sequences into encoder at one time
		parameters.batch_size = len(source_data)

		#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
		sos_step_embedded = tf.nn.embedding_lookup(embeddings, sos_time_slice)
		eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
		pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
    
		#manually specifying loop function through time - to get initial cell state and input to RNN
		
		#we define and return these values, no operations occur here
		def loop_fn_initial():
			initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
			initial_input = sos_step_embedded
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
			# time == 0
			if previous_state is None:
				assert previous_output is None and previous_state is None
				return loop_fn_initial()
			else:
				return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

		# creates uninitialized suffix variables, we have to overwrite them
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

		fd = next_feed(source_data, target_data, encoder_inputs, encoder_inputs_length, decoder_targets, parameters)

		# get decoder predictions
		predict_ = sess.run(decoder_prediction, fd)

		encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_data)
    
        	# get max input sequence length
		max_input_length = max(encoder_input_lengths_)
    
        	# target word is max character_changing_num character longer than source word 
        	# get transpose of target_batches[i] and put an EOF and PAD at the end
		decoder_targets_, _ = helpers.batch(
            		[(sequence) + [parameters.EOS] + [parameters.PAD] * ((max_input_length + parameters.character_changing_num - 1) - len(sequence))  for sequence in target_data]
        	)

        	# calculate accuracy
        	# transpose back the arrays
        	# correct is a list of true and false values depends on equality
		correct = tf.equal(tf.cast(predict_.transpose(), tf.float32), tf.cast(decoder_targets_.transpose(), tf.float32))
		equality = correct.eval(fd)

		samplenum = 0
		sampleright = 0
		
		# analises predicted words for the percentage of full word equality
		for i in equality:
			right = 1
			for j in i:
				if j == False:
					right = 0
					break
			if right == 1:
				sampleright += 1
			samplenum += 1

		print('accuracy:',sampleright/samplenum)


if __name__ == '__main__':
    main()
