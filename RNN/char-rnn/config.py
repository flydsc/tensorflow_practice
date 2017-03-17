class para_config():
    batch_size = 32
    n_epoch = 100
    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.9
    grad_clip = 5

    state_size = 100
    num_layers = 3
    seq_length = 20
    log_dir = './logs'
    metadata = 'metadata.tsv'
    gen_num = 500