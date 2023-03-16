iters_per_epoch = 3517
max_num_epochs = 24
print_freq = 50

scheduler = dict(
    sched='cosine',
    num_steps=iters_per_epoch * max_num_epochs,
    min_lr=1e-6,
    warmup_lr=1e-5,
    warmup_steps=500,
    t_in_epochs=False,
)
