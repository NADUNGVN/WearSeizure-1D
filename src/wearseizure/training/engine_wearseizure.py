"""WearSeizure-1D uses the exact same per-fold training/threshold/evaluation
procedure as the baselines -- only the model architecture passed into
`run_fold` differs. Re-exported here (rather than duplicated) so callers can
import from the module name matching the model they are training.
"""
from wearseizure.training.engine_baseline import FoldResult, run_fold

__all__ = ["FoldResult", "run_fold"]
