"""Module for the learning schema configuration."""

from drytorch import LearningSchema
from drytorch.core import protocols as p
from drytorch.lib import gradient_ops, schedulers

from src.config.options import ClipCriterion, GradOp, Schedulers
from src.config.specs import ExperimentConfig, LearningConfig, SchedulerConfig

__all__ = [
    'get_grad_op',
    'get_learning_schema',
    'get_scheduler',
]


def get_scheduler(config: SchedulerConfig, /) -> schedulers.AbstractScheduler:
    """Return the scheduler instance based on config.

    Args:
        config: scheduler configuration parameters.
    """
    if config.function == Schedulers.Constant:
        scheduler: schedulers.AbstractScheduler = schedulers.ConstantScheduler()
    elif config.function == Schedulers.Cosine:
        scheduler = schedulers.CosineScheduler(**config.settings)
    elif config.function == Schedulers.Exponential:
        scheduler = schedulers.ExponentialScheduler(**config.settings)
    else:
        raise ValueError(f'Scheduler {config.function} not supported.')

    if config.restart_interval > 0:
        restart = schedulers.restart(restart_interval=config.restart_interval, restart_fraction=config.restart_fraction)
        scheduler = scheduler.compose(restart)

    if config.warmup_steps > 0:
        warmup = schedulers.warmup(config.warmup_steps)
        scheduler = scheduler.compose(warmup)

    return scheduler


def get_grad_op(config: LearningConfig, /) -> p.GradientOpProtocol:
    """Return the gradient clipping instance based on config.

    Args:
        config: learning configuration parameters.
    """
    if config.grad_op == GradOp.GradParamNormalizer:
        return gradient_ops.GradParamNormalizer()

    if config.grad_op == GradOp.GradZScoreNormalizer:
        return gradient_ops.GradZScoreNormalizer()

    if config.grad_op == GradOp.GradValueClipper:
        return gradient_ops.GradValueClipper()

    if config.grad_op == GradOp.GradNormClipper:
        return gradient_ops.GradNormClipper()

    if config.grad_op == GradOp.HistClipper:
        if config.clip_criterion == ClipCriterion.ZStat:
            return gradient_ops.HistClipper(criterion=gradient_ops.ZStatCriterion())

        if config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.HistClipper(criterion=gradient_ops.EMACriterion())

    if config.grad_op == GradOp.ParamHistClipper:
        if config.clip_criterion == ClipCriterion.ZStat:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.ZStatCriterion())

        if config.clip_criterion == ClipCriterion.EMA:
            return gradient_ops.ParamHistClipper(criterion=gradient_ops.EMACriterion())

    return gradient_ops.NoOp()


def get_learning_schema(exp_cfg: ExperimentConfig, /) -> LearningSchema:
    """Return the configured learning schema.

    Args:
        exp_cfg: subexperiment configuration object.
    """
    config = exp_cfg.train.learn
    return LearningSchema(
        base_lr=config.lr,
        gradient_op=get_grad_op(config),
        optimizer_cls=config.optimizer_cls,
        optimizer_defaults=config.opt_settings,
        scheduler=get_scheduler(config.scheduler),
    )
