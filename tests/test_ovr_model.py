# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

import pytest
import torch

from incremental_feature_cl.models import (
    IdentityBackbone,
    IncrementalFeatureMapModel,
    OneVsRestSkillModel,
    Skill,
    SkillConfig,
)

D = 6


def _data(classes, n=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    xs, ys = [], []
    for c in classes:
        center = torch.zeros(D)
        center[c % D] = 3.0
        xs.append(center + 0.3 * torch.randn(n, D, generator=g))
        ys.append(torch.full((n,), c))
    return torch.cat(xs), torch.cat(ys)


def _model(**kw):
    cfg = SkillConfig(memory_per_class=kw.pop("memory_per_class", 5), epochs=2, **kw)
    return OneVsRestSkillModel(IdentityBackbone(D), config=cfg, seed=1)


def test_skill_creation_and_deterministic_class_mapping():
    m = _model()
    m.update(*_data([16, 17, 18]))
    assert set(m.skills) == {16, 17, 18}
    assert m.class_ids == [16, 17, 18]
    x, _ = _data([16, 17, 18], n=4)
    scores = m.decision_function(x)
    assert scores.shape == (12, 3)
    for j, c in enumerate(m.class_ids):  # column j <-> class_ids[j]
        assert torch.allclose(scores[:, j], m.skills[c].decision_function(m.extract(x)))


def test_no_duplicate_skills_and_identity_is_persistent():
    m = _model()
    m.update(*_data([0, 1]))
    s0, s1 = m.skills[0], m.skills[1]
    m.update(*_data([2]))
    assert m.skills[0] is s0 and m.skills[1] is s1
    m.update(*_data([0, 3]))  # class 0 appears again
    assert m.skills[0] is s0
    assert m.class_ids == [0, 1, 2, 3]
    assert len(m.skills) == 4


def test_class_accumulation():
    m = _model()
    seen = []
    for classes in ([0, 1], [2], [3, 4]):
        m.update(*_data(classes))
        seen.append(set(m.class_ids))
    assert seen == [{0, 1}, {0, 1, 2}, {0, 1, 2, 3, 4}]


def test_old_skills_receive_new_class_as_negative_structurally():
    m = _model()
    m.update(*_data([0, 1]))
    assert m.skills[0].known_negatives == [1]
    assert m.skills[1].known_negatives == [0]
    rec = m.update(*_data([2]))
    assert m.skills[0].known_negatives == [1, 2]
    assert m.skills[1].known_negatives == [0, 2]
    assert m.skills[2].known_negatives == [0, 1]  # new skill sees old classes via replay
    assert rec["new_skills"] == [2] and rec["updated_old_skills"] == [0, 1]
    assert rec["skills"][0]["newly_known_negative_classes"] == [2]
    assert rec["skills"][0]["negatives_used_per_class"].get(2, 0) > 0
    assert 0 not in m.skills[0].negative_memory  # never its own class


def test_replay_memory_has_old_plus_new_negatives_within_budget():
    m = _model(memory_per_class=5)
    m.update(*_data([0, 1], n=12))
    assert len(m.skills[0].negative_memory[1]) == 5  # budget enforced
    m.update(*_data([2], n=12))
    mem = m.skills[0].negative_memory
    assert set(mem) == {1, 2} and len(mem[1]) == 5 and len(mem[2]) == 5
    assert len(m.skills[0].positive_memory) == 5
    rec = m.history[-1]["skills"][0]
    assert rec["replay_positive"] == 5 and rec["replay_negative"] == 5  # old negs replayed


def test_memory_is_shared_by_reference_not_copied():
    m = _model()
    m.update(*_data([0, 1, 2]))
    assert m.skills[0].negative_memory[1] is m.skills[2].negative_memory[1] is m.exemplars[1]


def test_no_replay_budget_still_tracks_known_negatives_but_stores_nothing():
    m = _model(memory_per_class=0)
    m.update(*_data([0, 1]))
    m.update(*_data([2]))
    assert m.skills[0].known_negatives == [1, 2]
    assert all(len(v) == 0 for v in m.skills[0].negative_memory.values())
    assert m.history[-1]["skills"][0]["trained"] is False  # no positives left to train on


def test_zero_memory_new_skill_trains_when_experience_has_negatives():
    m = _model(memory_per_class=0)
    m.update(*_data([0, 1]))

    rec = m.update(*_data([2, 3]))

    assert rec["skills"][2]["trained"] is True
    assert rec["skills"][2]["n_negative"] > 0
    assert rec["skills"][3]["trained"] is True
    assert rec["skills"][3]["n_negative"] > 0


def test_zero_memory_new_skill_cannot_train_without_negative_examples():
    m = _model(memory_per_class=0)
    m.update(*_data([0, 1]))

    rec = m.update(*_data([2]))

    assert rec["skills"][2]["trained"] is False
    assert rec["skills"][2]["reason"] == "no negatives"


def test_requested_vs_realized_ratio_is_recorded():
    m = _model(target_to_negatives=0.5)  # 1:2
    rec = m.update(*_data([0, 1, 2, 3], n=10))
    r = rec["skills"][0]
    assert r["requested_target_to_negatives"] == 0.5
    assert r["n_positive"] == 10 and r["n_negative"] == 20
    assert r["realized_target_to_negatives"] == pytest.approx(0.5)
    assert set(r["negatives_used_per_class"]) == {1, 2, 3}  # every negative class represented


def test_prediction_from_injected_scores():
    m = _model()
    m.update(*_data([0, 1, 2]))
    scores = torch.tensor([[0.2, 0.8, 0.3]])
    assert m.scores_to_predictions(scores).item() == 1
    m2 = _model()
    m2.update(*_data([16, 17, 18]))
    assert m2.scores_to_predictions(torch.tensor([[0.2, 0.8, 0.3]])).item() == 17  # class id


def test_prediction_takes_only_x_and_is_label_free():
    m = _model()
    m.update(*_data([0, 1, 2]))
    x, y = _data([0, 1, 2], n=10, seed=5)
    assert m.predict(x).shape == (30,)
    assert (m.predict(x) == y).float().mean() > 0.8


def test_backbone_is_frozen_and_parameter_accounting():
    m = _model()
    m.update(*_data([0, 1]))
    assert all(not p.requires_grad for p in m.backbone.parameters())
    assert m.parameter_count(True) == 2 * (D + 1)
    m.train()
    assert not m.backbone.training


def test_skill_rejects_own_class_as_negative():
    s = Skill(3, D)
    with pytest.raises(ValueError):
        s.remember(negatives={3: torch.zeros(2, D)})


def test_determinism():
    a, b = _model(), _model()
    for classes in ([0, 1], [2]):
        a.update(*_data(classes))
        b.update(*_data(classes))
    x, _ = _data([0, 1, 2], n=3)
    assert torch.equal(a.decision_function(x), b.decision_function(x))


def test_prediction_mapping_survives_non_sorted_class_arrival():
    m = _model()
    m.update(*_data([17, 18]))
    m.update(*_data([2]))  # a lower class id arrives after the first experience
    assert m.class_ids == [17, 18, 2]

    scores = torch.tensor(
        [
            [0.9, 0.1, 0.2],
            [0.1, 0.8, 0.2],
            [0.1, 0.2, 0.9],
        ]
    )
    assert m.scores_to_predictions(scores).tolist() == [17, 18, 2]

    # class_id_scores accepts input samples, not an already-computed score matrix.
    x, _ = _data([17, 18, 2], n=1, seed=9)
    class_id_scores = m.class_id_scores(x, num_classes=20)
    assert class_id_scores.shape == (3, 20)
    assert class_id_scores.argmax(dim=1).tolist() == m.predict(x).tolist()


def test_skill_uses_persistent_incremental_feature_map():
    skill = Skill(
        target_class=3,
        feature_dim=8,
        config=SkillConfig(
            feature_expansion_dim=4,
            epochs=1,
            batch_size=8,
        ),
        seed=7,
    )

    assert isinstance(skill.model, IncrementalFeatureMapModel)

    x = torch.randn(12, 8)

    before = skill(x).detach().clone()
    original_weight = skill.model.classifier.weights[0].weight
    original_parameter = original_weight.detach().clone()

    skill.model.expand_feature_space(
        new_dim=4,
        seed=11,
    )

    after = skill(x).detach()

    assert after.shape == before.shape
    assert torch.allclose(before, after, atol=1e-6, rtol=1e-5)

    assert skill.model.classifier.weights[0].weight is original_weight
    assert torch.equal(original_weight.detach(), original_parameter)
    assert skill.model.feature_dim == 12
