""" Modified from timm."""
""" Scheduler Factory
Hacked together by / Copyright 2021 Ross Wightman
"""
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler
from timm.scheduler.step_lr import StepLRScheduler


def create_scheduler(args, optimizer):
    num_steps = args.num_steps

    cycle_args = dict(
        cycle_mul=getattr(args, 'lr_cycle_mul', 1.),
        cycle_decay=getattr(args, 'lr_cycle_decay', 0.1),
        cycle_limit=getattr(args, 'lr_cycle_limit', 1),
    )

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            t_in_epochs=args.t_in_epochs,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            **cycle_args,
        )
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=args.decay_steps,
            decay_rate=args.decay_rate,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            t_in_epochs=args.t_in_epochs,
        )
    elif args.sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_steps,
            k_decay=getattr(args, 'lr_k_decay', 1.0),
            t_in_epochs=args.t_in_epochs,
            **cycle_args,
        )

    return lr_scheduler
