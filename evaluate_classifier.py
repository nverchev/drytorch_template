"""Classifier evaluation entry point.

The evaluation partition is the test set only with ``final=true``, the validation set otherwise.

Usage::

    uv run evaluate_classifier.py
    uv run evaluate_classifier.py --config-name signal_classifier
"""

from drytorch import Test

from src.config import ExperimentConfig, Experiment, hydra_main
from src.config.environment import DEBUG_MODE
from src.data.dataset import get_databuilder, get_splits
from src.data.derived_dataset import EncodedDataset
from src.lib import get_encoded_loader, get_loaders
from src.lib.events import ConfusionMatrixEvent
from src.lib.models import get_classifier_ema_model, load_autoencoder_model, load_trained_state
from src.lib.objectives import AggregatedAccuracy
from src.trackers import get_trackers


def evaluate_classification() -> None:
    """Test the trained classifier and publish its confusion matrix.

    With an autoencoding phase, the classifier is tested on the encodings of the trained autoencoder.
    """
    cfg = Experiment.get_config()
    class_cfg = cfg.classification
    model = get_classifier_ema_model()
    load_trained_state(model, epoch=cfg.user.load or -1)
    if cfg.autoencoding is None:
        _, eval_loader = get_loaders(batch_size=class_cfg.train.batch_size_per_device, n_workers=cfg.user.n_workers)
    else:
        _, eval_split = get_splits()
        encoded_eval = EncodedDataset(eval_split, load_autoencoder_model())
        eval_loader = get_encoded_loader(encoded_eval, batch_size=class_cfg.train.batch_size_per_device)

    accuracy = AggregatedAccuracy(get_databuilder().n_classes)
    test = Test(model, loader=eval_loader, metric=accuracy, name='Accuracy')
    test()
    ConfusionMatrixEvent(
        source_name=test.name,
        model_name=model.name,
        epoch=model.epoch,
        matrix=accuracy.confusion_matrix,
    )
    return


@hydra_main
def main(cfg: ExperimentConfig) -> None:
    """Set up the experiment and evaluate the classifier."""
    trackers = get_trackers(cfg)
    exp = Experiment(cfg, name=cfg.name, par_dir=cfg.par_dir, tags=cfg.tags)
    for tracker in trackers:
        exp.trackers.subscribe(tracker)

    with exp.create_run(resume=True, run_id=cfg.user.run_id, record=not DEBUG_MODE):
        evaluate_classification()

    return


if __name__ == '__main__':
    main()
