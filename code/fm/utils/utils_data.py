def batcher(x, y = None, batch_size = -1):
    n_samples = x.shape[0]

    if batch_size == -1:
        batch_size = n_samples
    if batch_size < 1:
       raise ValueError('Parameter batch_size={} is unsupported'.format(batch_size))

    for i in range(0, n_samples, batch_size):
        upper_bound = min(i + batch_size, n_samples)
        ret_x = x[i:upper_bound]
        ret_y = None
        if y is not None:
            ret_y = y[i:i + batch_size]
        yield (ret_x, ret_y)





