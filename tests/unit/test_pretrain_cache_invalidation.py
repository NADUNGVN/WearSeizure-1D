"""The cohort-init cache must not survive a change of corpus.

`get_or_train_cohort_init` caches one initialisation per held-out subject under
`pretrain/<model>/<window>/seed<N>/`. That key says nothing about WHICH corpus
the initialisation was pre-trained on, so turning lever L5 on -- or narrowing
`data.pretrain_channels` -- would otherwise hand back the initialisation built
from the narrower corpus. The run would complete, the numbers would be
unchanged, and the conclusion would be "L5 does nothing" when L5 never ran.
"""
from __future__ import annotations

import json

import pandas as pd
import torch
from torch import nn

from wearseizure.training.pretrain import get_or_train_cohort_init


class _TinyModel(nn.Module):
    """Smallest thing that maps (B, 1, L) to two logits. Keeps the test about
    caching rather than about training anything meaningful."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x.mean(dim=(1, 2), keepdim=False).unsqueeze(-1))


def _train_once(tmp_path, records, manifest_df, extra=None, force=False):
    return get_or_train_cohort_init(
        records=records,
        manifest_df=manifest_df,
        held_out_subject=min(manifest_df["subject_id"].unique()),
        model_factory=_TinyModel,
        cache_dir=tmp_path / "pretrain",
        window_s=4.0,
        stride_s=1.0,
        seed=0,
        epochs=1,
        lr=1e-3,
        weight_decay=0.0,
        batch_size=64,
        device="cpu",
        early_stopping_patience=1,
        force=force,
        extra_manifest_df=extra,
    )


def test_a_cache_hit_is_reused_when_the_corpus_is_unchanged(tmp_path, synthetic_cohort, caplog):
    manifest_df, records = synthetic_cohort
    first = _train_once(tmp_path, records, manifest_df)
    with caplog.at_level("INFO"):
        second = _train_once(tmp_path, records, manifest_df)
    assert "reusing cached init" in caplog.text
    for key in first:
        assert torch.equal(first[key], second[key])


def test_changing_the_corpus_invalidates_the_cache(tmp_path, synthetic_cohort, caplog):
    manifest_df, records = synthetic_cohort
    _train_once(tmp_path, records, manifest_df)

    # Same cache key (model, window, seed, held-out subject), different corpus:
    # one extra subject's recordings joining the pre-training pool.
    extra_rows = manifest_df[manifest_df["subject_id"] == max(manifest_df["subject_id"])].copy()
    extra_rows["subject_id"] = "synX"
    extra_rows["edf_id"] = ["pre_" + str(i) for i in extra_rows["edf_id"]]
    extra_records = {
        new_id: records[old_id]
        for new_id, old_id in zip(extra_rows["edf_id"], manifest_df.loc[extra_rows.index, "edf_id"])
    }

    with caplog.at_level("WARNING"):
        _train_once(
            tmp_path,
            {**records, **extra_records},
            manifest_df,
            extra=pd.DataFrame(extra_rows),
        )
    assert "built from a different corpus" in caplog.text

    meta = json.loads((tmp_path / "pretrain" / f"{min(manifest_df['subject_id'].unique())}.json")
                      .read_text(encoding="utf-8"))
    assert meta["n_extra_pretrain_rows"] == len(extra_rows)
