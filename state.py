from collections import OrderedDict

def prototype_state():
    state = {} 
     
    # Random seed
    state['seed'] = 1234
    # Logging level
    state['level'] = 'DEBUG'

    # A string representation for the unknown word placeholder for both language
    state['oov'] = '<unk>'
    # These are unknown word placeholders
    state['unk_sym'] = 0
    state['n_samples'] = 40
    
    # These are end-of-sequence marks
    state['start_sym_query'] = '<q>'
    state['end_sym_query'] = '</q>'
    state['end_sym_session'] = '</s>'
    state['n_sym'] = None
 
    # ----- MODEL COMPONENTS -----
    # Low-rank approximation activation function
    state['rank_n_activ'] = 'lambda x: x'
    # Hidden-to-hidden activation function
    state['activ'] = 'lambda x: TT.tanh(x)'

    # ----- SIZES ----
    # Dimensionality of hidden layers
    state['qdim'] = 512
    # Dimensionality of session hidden layer 
    state['sdim'] = 1000
    # Dimensionality of low-rank approximation
    state['rankdim'] = 256
    state['lambda_rank'] = 0.01

    # Threshold to clip the gradient
    state['cutoff'] = 1.
    state['lr'] = 0.0001

    # Early stopping configuration
    state['patience'] = 5
    state['cost_threshold'] = 1.003
    
    # ----- TRAINING METHOD -----
    # Choose optimization algorithm
    state['updater'] = 'adadelta'
    
    # Batch size
    state['bs'] = 128 
     
    # We take this many minibatches, merge them,
    # sort the sentences according to their length and create
    # this many new batches with less padding.
    state['sort_k_batches'] = 20
    
    # Maximum sequence length / trim batches
    state['seqlen'] = 50

    # Should we use a deep output layer
    # and maxout on the outputs?
    state['deep_out'] = True
    state['maxout_out'] = True

    # Maximum number of iterations
    state['max_iters'] = 10
    state['save_dir'] = './'
    
    # ----- TRAINING PROCESS -----
    # Frequency of training error reports (in number of batches)
    state['trainFreq'] = 10
    # Validation frequency
    state['validFreq'] = 5000
    # Number of batches to process
    state['loopIters'] = 3000000
    # Maximum number of minutes to run
    state['timeStop'] = 24*60*31
    # Error level to stop at
    state['minerr'] = -1
    return state

def prototype_rfp_50k():
    state = prototype_state()

    state['train_session'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/rfp.train.clean_50k.train.ses.pkl"
    state['train_rank'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/rfp.train.clean_50k.train.rnk.pkl"
    state['valid_session'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/rfp.train.clean_50k.valid.ses.pkl"
    state['valid_rank'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/rfp.train.clean_50k.valid.rnk.pkl"
    state['dictionary'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/rfp.train.clean_50k.dict.pkl"
     
    state['decoder_bias_all'] = True
    state['reset_recurrence'] = True
    state['prefix'] = "rfp_30min_50k_"
    
    state['lambda_rank'] = 1
    state['deep_out'] = True
    state['maxout_out'] = False 
    state['updater'] = 'adam'

    #
    state['qdim'] = 600
    # Dimensionality of session hidden layer 
    state['sdim'] = 1200 
    # Dimensionality of low-rank approximation
    state['rankdim'] = 300
    # 

    state['bs'] = 80
    state['seqlen'] = 50
    
    state['eos_sym'] = 3
    state['eoq_sym'] = 2
    state['soq_sym'] = 1
    
    state['save_dir'] = '/part/01/Tmp/sordonia/Work/Reformulations/Datasets/Models/'
    state['session_rec_activation'] = "lambda x: T.tanh(x)"
    state['query_rec_activation'] = "lambda x: T.tanh(x)"

    state['session_step_type'] = 'gated'
    state['query_step_type'] = 'gated' 
    return state

def prototype_aol_50k():
    state = prototype_state()

    state['train_session'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/aol.train.clean_50k.train.ses.pkl"
    state['train_rank'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/aol.train.clean_50k.train.rnk.pkl"
    state['valid_session'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/aol.train.clean_50k.valid.ses.pkl"
    state['valid_rank'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/aol.train.clean_50k.valid.rnk.pkl"
    state['dictionary'] = "/part/01/Tmp/sordonia/Work/Reformulations/Datasets/aol.train.clean_50k.dict.pkl"
     
    state['decoder_bias_all'] = True
    state['reset_recurrence'] = True
    state['prefix'] = "aol_30min_50k_"
     
    state['deep_out'] = True
    state['maxout_out'] = False 
    state['updater'] = 'adam'
    state['lambda_rank'] = 0.01

    #
    state['qdim'] = 1000
    # Dimensionality of session hidden layer 
    state['sdim'] = 2000 
    # Dimensionality of low-rank approximation
    state['rankdim'] = 300
    # 

    state['bs'] = 80
    state['seqlen'] = 50
    
    state['eos_sym'] = 3
    state['eoq_sym'] = 2
    state['soq_sym'] = 1
    
    state['save_dir'] = '/part/01/Tmp/sordonia/Work/Reformulations/Datasets/Models/'
    state['session_rec_activation'] = "lambda x: T.tanh(x)"
    state['query_rec_activation'] = "lambda x: T.tanh(x)"

    state['session_step_type'] = 'gated'
    state['query_step_type'] = 'gated' 
    return state
