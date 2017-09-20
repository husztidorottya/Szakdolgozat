import tensorflow as tf 
import numpy as np
import helpers
import operator
import random


# create (source morphological tags + target morphological tags + source/target word) sequence
def create_sequence(data_line_, word_index, BOS, EOS):
    sequence = []
    
    # task 2
    if len(data_line_) == 4:
        # source and target morphological tags are appended only to the input
        if word_index != 3:
            for i in data_line_[0]:
                sequence.append(i)
            
            for i in data_line_[2]:
                sequence.append(i)
    # task 1,3
    else:
        if word_index != 2:
            # source and target morphological tags are appended only to the input
            for i in data_line_[1]:
                sequence.append(i)
        
    # append beginning of the input
    sequence.append(BOS)

    for i in data_line_[word_index]:
        sequence.append(i)
        
    # append end of the input
    #sequence.append(EOS)
        
    return sequence


# encoding input data
def encoding(data, coded_word, alphabet_and_morph_tags):
    for character in data:
        index = alphabet_and_morph_tags.setdefault(character, len(alphabet_and_morph_tags) + 3)
        coded_word.append(index)
        
    return coded_word


def read_split_encode_data(filename, alphabet_and_morph_tags, BOS, EOS):
    # read, split and encode input data
    with open(filename,'r') as input_file:
        source_data = []
        target_data = []
        idx = 0
        # read it line-by-line
        for line in input_file:
            data_line_ = line.strip('\n').split('\t')
        
            # encode words into vector of ints 
            for item in range(0,len(data_line_)):         
                # contains encoded form of word
                coded_word = []
            
                # task 2
                if len(data_line_) == 4:
                    if item == 1 or item == 3:
                        # encode source and target word
                        coded_word = encoding(data_line_[item], coded_word, alphabet_and_morph_tags)
                    else:
                        # split morphological tags
                        tags = data_line_[item].split(',')
                
                        coded_word = encoding(tags, coded_word, alphabet_and_morph_tags)
                # task 1,3
                else:
                    if item == 1:
                        # split morphological tags
                        tags = data_line_[item].split(',')
                
                        coded_word = encoding(tags, coded_word, alphabet_and_morph_tags)
                    else:
                        # encode source and target word
                        coded_word = encoding(data_line_[item], coded_word, alphabet_and_morph_tags)
                        
                # store encoded form
                data_line_[item] = coded_word
        
            # defines source and target words' index
            source_idx = len(data_line_) - 3
            target_idx = len(data_line_) - 1 
        
            # store encoder input task 2:(source morphological tags + target morphological tags + source word)
            # task 1,3: (source/target morphological tags + source word)
            source_data.append([create_sequence(data_line_, source_idx, BOS, EOS), idx])
        
            # store decoder expected outputs:(target word)
            target_data.append(create_sequence(data_line_, target_idx, BOS, EOS))
        
            # stores line number (needed for shuffle) - reference for the target_data
            idx += 1

    return source_data, target_data


# create batches with size of batch_size
def create_batches(source_data, target_data, batch_size):
    # stores batches
    source_batches = []
    target_batches = []
    # stores last batch ending index
    prev_batch_end = 0
    
    for j in range(0, len(source_data)):
        if j % batch_size == 0 and j != 0:
            # stores a batch
            sbatch = []
            tbatch = []
            for k in range(prev_batch_end+1,j+1):
                # store sequence
                sbatch.append(source_data[k][0])
                # store expected target_data (know from source_data index)
                tbatch.append(target_data[source_data[k][1]])
            # add created batch
            source_batches.append(sbatch)
            target_batches.append(tbatch)
            prev_batch_end = j
            
    # put the rest of it in another batch
    if prev_batch_end != j:
        sbatch = []
        tbatch = []
        for k in range(prev_batch_end+1,j):
            sbatch.append(source_data[k][0])
            tbatch.append(target_data[source_data[k][1]])
        source_batches.append(sbatch)
        target_batches.append(tbatch)
        
    return source_batches, target_batches


def main():

    # GLOBAL CONTANTS
    BOS = 2
    PAD = 0
    EOS = 1
    character_changing_num = 10
    batches_in_epoch = 100
    #character length
    input_embedding_size = 300 
    neuron_num =100
    epoch = 1000
    
    loss_track = []

    # x (store encoder inputs [source morphological tags + target morphological tags + source word])
    source_data = []
    # y (store decoder expected outputs [source morphological tags + target morphological tags + target word])      
    target_data = []

    # stores encoded forms
    alphabet_and_morph_tags = dict()

    source_data, target_data = read_split_encode_data('teszt.tsv', alphabet_and_morph_tags, BOS, EOS)

    # Clears the default graph stack and resets the global default graph.
    tf.reset_default_graph() 
    # initializes a tensorflow session
    sess = tf.InteractiveSession() 

    # get max value of encoded forms
    max_alphabet_and_morph_tags = alphabet_and_morph_tags[max(alphabet_and_morph_tags.items(), key=operator.itemgetter(1))[0]]

    # calculate vocab_size
    vocab_size = max_alphabet_and_morph_tags + 1

    # num neurons
    encoder_hidden_units = neuron_num 
    # in original paper, they used same number of neurons for both encoder
    # and decoder, but we use twice as many so decoded output is different, the target value is the original input 
    #in this example
    decoder_hidden_units = encoder_hidden_units * 2 

    # input placehodlers
    encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
    # contains the lengths for each of the sequence in the batch, we will pad so all the same
    # if you don't want to pad, check out dynamic memory networks to input variable length sequences
    encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
    decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

    # randomly initialized embedding matrrix that can fit input sequence
    # used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
    # reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
    embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)
    #embeddings = tf.Variable(tf.eye(vocab_size, input_embedding_size), dtype='float32')

    # this thing could get huge in a real world application
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

    # define encoder
    encoder_cell = tf.contrib.rnn.GRUCell(encoder_hidden_units)

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

    #Concatenates tensors along one dimension.
    encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)

    # because by GRUCells the state is a Tensor, not a Tuple like by LSTMCells
    encoder_final_state = tf.concat(
        (encoder_fw_final_state, encoder_bw_final_state), 1)


    decoder_cell = tf.contrib.rnn.GRUCell(decoder_hidden_units)

    #we could print this, won't need
    encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

    decoder_lengths = encoder_inputs_length + character_changing_num
    # +(character_changing_num-1) additional steps, +1 leading <EOS> token for decoder inputs

    #manually specifying since we are going to implement attention details for the decoder in a sec
    #weights
    W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
    #W = tf.Variable(tf.eye(decoder_hidden_units, vocab_size), dtype='float32')
    #bias
    b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

    #create padded inputs for the decoder from the word embeddings
    #were telling the program to test a condition, and trigger an error if the condition is false.
    assert EOS == 1 and PAD == 0 and BOS == 2

    bos_time_slice = tf.fill([batch_size], 2, name='BOS')

    eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
    pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

    # send 20 sequences into encoder at one time
    batch_size = 20

    #retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
    eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
    pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)
    bos_step_embedded = tf.nn.embedding_lookup(embeddings, bos_time_slice)

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

    #Creates an RNN specified by RNNCell cell and loop function loop_fn.
    #This function is a more primitive version of dynamic_rnn that provides more direct access to the 
    #inputs each iteration. It also provides more control over when to start and finish reading the sequence, 
    #and what to emit for the output.
    #ta = tensor array
    decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
    decoder_outputs = decoder_outputs_ta.stack()

    #to convert output to human readable prediction
    #we will reshape output tensor

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

    #cross entropy loss
    #one hot encode the target values so we don't rank just differentiate
    stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
        logits=decoder_logits,
    )

    #loss function
    loss = tf.reduce_mean(stepwise_cross_entropy)
    #train it 
    #train_op = tf.train.AdamOptimizer().minimize(loss)
    #train it 
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss) # set learning_rate = 0.001


    sess.run(tf.global_variables_initializer())

    def next_feed(batch_num, source_batches, target_batches):
        # get transpose of source_batches[batch_num]
        encoder_inputs_, encoder_input_lengths_ = helpers.batch(source_batches[batch_num])
    
        # get max input sequence length
        max_input_length = max(encoder_input_lengths_)
    
        # target word is max character_changing_num character longer than source word 
        # get transpose of target_batches[i] and put an EOF and PAD at the end
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] + [PAD] * ((max_input_length + character_changing_num - 1) - len(sequence))  for sequence in target_batches[batch_num]]
        )
   
        return {
            encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_,
        }



    try:
        for epoch_num in range(epoch):
            # get every batches and train the model on it
            print('Epoch:',epoch_num)

            # shuffle it in every epoch for creating random batches
            source_data = random.sample(source_data, len(source_data))
        
            # encoder inputs and decoder outputs devided into batches
            source_batches, target_batches = create_batches(source_data, target_data, batch_size)

            for batch_num in range(0, len(source_batches)):
                fd = next_feed(batch_num, source_batches, target_batches)
   
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)
        
                if batch_num == 0 or batch_num % batches_in_epoch == 0:
                    print('batch {}'.format(batch_num))
                    print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                    predict_ = sess.run(decoder_prediction, fd)
                    for i, (inp, pred) in enumerate(zip(fd[decoder_targets].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('    input     > {}'.format(inp))
                        print('    predicted > {}'.format(pred))
                        if i >= 2:
                            break
                    print()

    except KeyboardInterrupt:
        print('training interrupted')

if __name__ == '__main__':
    main()


