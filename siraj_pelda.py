import tensorflow as tf 
import numpy as np
import helpers
import operator

# x
input_data = []
# y
target_data = []

# betűk kódolása
alphabet = dict()
# forrás és cél morfológiai tagek kódolása
morphological_tags = dict()

# kezdőkarakter
BOS = 0
# zárókarakter
EOS = 1

# beolvassa és feldarabolja input adatot
with open('data2.tsv','r') as input:
    for line in input:
        data_line_ = line.strip('\n').split('\t')
        
        # szavak kódolása számvektorokká    
        for item in range(0,4):          
            coded_word = []
            
            if item == 1 or item == 3:
                # forrás és cél szóalak kódolása
                for character in data_line_[item]:
                    index = alphabet.setdefault(character, len(alphabet) + 2)
                    coded_word.append(index)
            else:
                # morfológiai tag-ek kódolása
                index = morphological_tags.setdefault(data_line_[item], len(morphological_tags) + 2)
                coded_word.append(index)
            
            data_line_[item] = coded_word
        
        source = []
        
        for i in data_line_[0]:
            source.append(i)
            
        for i in data_line_[2]:
            source.append(i)
            
        for i in data_line_[1]:
            source.append(i)
        
        input_data.append(source)
        # cél szóalak
        target = []
        
        for i in data_line_[0]:
            target.append(i)
            
        for i in data_line_[2]:
            target.append(i)
        
        for i in data_line_[3]:
            target.append(i)
           
        target_data.append(target)
        
                
# -----------------------------

tf.reset_default_graph() #Clears the default graph stack and resets the global default graph.
sess = tf.InteractiveSession() #initializes a tensorflow session

PAD = 0
EOS = 1

# calculate vocab_size (max(alphabet,morphological_tags))
max_alphabet = alphabet[max(alphabet.items(), key=operator.itemgetter(1))[0]]
max_morphological_tags = morphological_tags[max(morphological_tags.items(), key=operator.itemgetter(1))[0]]

vocab_size = max([max_alphabet,max_morphological_tags]) + 1 # hány különböző karakterből áll a szó max
input_embedding_size = 30 #character length (hány karakterból áll a szó)

encoder_hidden_units = 20 #num neurons
decoder_hidden_units = encoder_hidden_units * 2 #in original paper, they used same number of neurons for both encoder
#and decoder, but we use twice as many so decoded output is different, the target value is the original input 
#in this example

#input placehodlers
encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
#contains the lengths for each of the sequence in the batch, we will pad so all the same
#if you don't want to pad, check out dynamic memory networks to input variable length sequences
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')


#randomly initialized embedding matrrix that can fit input sequence
#used to convert sequences to vectors (embeddings) for both encoder and decoder of the right size
#reshaping is a thing, in TF you gotta make sure you tensors are the right shape (num dimensions)
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#this thing could get huge in a real world application
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)

# encoder definiálás
encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)


# bidirectional encodre funkció definiálása (backpropagation)
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

#letters h and c are commonly used to denote "output value" and "cell state". 
#http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 
#Those tensors represent combined internal state of the cell, and should be passed together. 

encoder_final_state_c = tf.concat(
    (encoder_fw_final_state.c, encoder_bw_final_state.c), 1)

encoder_final_state_h = tf.concat(
    (encoder_fw_final_state.h, encoder_bw_final_state.h), 1)

#TF Tuple used by LSTM Cells for state_size, zero_state, and output state.
encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
    c=encoder_final_state_c,
    h=encoder_final_state_h
)

decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

#we could print this, won't need
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))

decoder_lengths = encoder_inputs_length + 10
# +2 additional steps, +1 leading <EOS> token for decoder inputs

#manually specifying since we are going to implement attention details for the decoder in a sec
#weights
W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
#bias
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)

#create padded inputs for the decoder from the word embeddings

#were telling the program to test a condition, and trigger an error if the condition is false.
assert EOS == 1 and PAD == 0

eos_time_slice = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice = tf.zeros([batch_size], dtype=tf.int32, name='PAD')

#retrieves rows of the params tensor. The behavior is similar to using indexing with arrays in numpy
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)

#manually specifying loop function through time - to get initial cell state and input to RNN
#normally we'd just use dynamic_rnn, but lets get detailed here with raw_rnn

#we define and return these values, no operations occur here
def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    #end of sentence
    initial_input = eos_step_embedded
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

decoder_outputs


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
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())


batch_size = 10

batches = input_data

# 10-es batchek legyártása forrás adatból
input_batches = []
ten_sequence2 = []
for j in range(0,len(input_data)):
    if j % 10 == 0:
        input_batches.append(ten_sequence2)
        ten_sequence2 = []
        ten_sequence2.append(input_data[j])
    else:
        ten_sequence2.append(input_data[j])

if ten_sequence2 != '': 
    input_batches.append(ten_sequence2)
    
batches = input_batches[1:len(input_batches)] 

######

# 10-es batchek legyártása cél adatból
target_batches = []
ten_sequence = []
for j in range(0,len(target_data)):
    if j % 10 == 0:
        target_batches.append(ten_sequence)
        ten_sequence = []
        ten_sequence.append(target_data[j])
    else:
        ten_sequence.append(target_data[j])

if ten_sequence != '': 
    target_batches.append(ten_sequence)
    
target_batches = target_batches[1:len(target_batches)] 

######

def next_feed(i,batches,target_batches):
    batch = batches[i]
    
    encoder_inputs_, encoder_input_lengths_ = helpers.batch(batch)
    
    
    if len(max(batch,key=len)) < len(max(target_batches[i],key=len)):
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] + [PAD] * (9-(len(max(target_batches[i],key=len))-len(max(batch,key=len)))) for sequence in target_batches[i]]
        )
    else:
        decoder_targets_, _ = helpers.batch(
            [(sequence) + [EOS] + [PAD] * (9+(len(max(batch,key=len))-len(max(target_batches[i],key=len)))) for sequence in target_batches[i]]
        )
    
    
    return {
        encoder_inputs: encoder_inputs_,
        encoder_inputs_length: encoder_input_lengths_,
        decoder_targets: decoder_targets_,
    }


loss_track = []

max_batches = 3001
batches_in_epoch = 10

try:
    #for batch in range(max_batches):
    for batch in range(0,len(batches)):
        i = batch
        
        fd = next_feed(i,batches,target_batches)
   
        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)
        
        '''
        if batch == 0 or batch % batches_in_epoch == 0:
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
        ''' 
        print('batch {}'.format(batch))
        print('  minibatch loss: {}'.format(sess.run(loss, fd)))
        predict_ = sess.run(decoder_prediction, fd)
        for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
            print('  sample {}:'.format(i + 1))
            print('    input     > {}'.format(inp))
            print('    predicted > {}'.format(pred))
            if i >= 2:
                break
        print()
        

except KeyboardInterrupt:
    print('training interrupted')

