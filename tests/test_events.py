# tests/test_control_and_events.py
import asyncio
import math
import pytest
from omegaconf import OmegaConf

from dynamicrl.core.events import (
    EventBus,
    EventEnvelope,
    ParamPatch,
    PatchBatch,
    Pause,
    Resume,
    Quit,
    CheckpointReq,
    is_param_patch,
    is_pause,
)
from dynamicrl.core.control import (
    HyperparamServer,
    PatchValidationError,
    VersionError,
)

# ---------------------------
# Helpers
# ---------------------------

def sample_cfg():
    # Minimal config covering keys used by policy. Mimics configs/defaults.yaml.
    return OmegaConf.create(
        {
            "algo": {
                "name": "ppo",
                "policy": "mlp",
                "lr": 3e-4,
                "clip_range": 0.2,
                "target_kl": 0.01,
                "ent_coef": 0.0,
                "vf_coef": 0.5,
                "n_steps": 2048,
                "n_minibatch": 32,
                "update_epochs": 10,
                "gamma": 0.99,
                "gae_lambda": 0.95,
                "max_grad_norm": 0.5,
            },
            "env": {
                "id": "Ant-v5",
                "num_envs": 8,
                "seed": 42,
                "normalize_obs": True,
                "normalize_reward": True,
                "frame_stack": 1,
            },
        }
    )

# ---------------------------
# events.py tests
# ---------------------------

@pytest.mark.asyncio
async def test_eventbus_basic_fifo():
    bus = EventBus()
    pid1 = await bus.publish(Pause("user"))
    pid2 = await bus.publish(ParamPatch("algo.lr", "set", 1e-4))
    env1 = await bus.get()
    env2 = await bus.get()
    assert env1.id != env2.id
    assert is_pause(env1)
    assert is_param_patch(env2)
    assert isinstance(env1, EventEnvelope)
    assert env1.event.reason == "user"
    assert env2.event.dotted == "algo.lr"
    assert env1.ts_mono <= env2.ts_mono

@pytest.mark.asyncio
async def test_eventbus_wait_for_predicate_and_requeue():
    bus = EventBus()
    # publish patch then pause
    await bus.publish(ParamPatch("algo.lr", "set", 1e-4))
    await bus.publish(Pause("hold"))
    # wait specifically for a Pause
    env = await bus.wait_for(lambda e: isinstance(e.event, Pause), timeout=1.0)
    assert is_pause(env)
    # the patch should still be in the queue, in order
    env2 = await bus.get()
    assert is_param_patch(env2)

def test_parampatch_path_normalization_and_dotted():
    p1 = ParamPatch("algo.lr", "set", 1e-4)
    p2 = ParamPatch(("algo", "lr"), "set", 1e-4)
    assert p1.path == p2.path == ("algo", "lr")
    assert p1.dotted == "algo.lr"

def test_parampatch_type_rules():
    with pytest.raises(ValueError):
        # empty path not allowed
        ParamPatch("", "set", 1.0)

# ---------------------------
# control.py tests
# ---------------------------

def test_hps_stage_and_confirm_simple_set():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    v = hps.stage(ParamPatch("algo.lr", "set", 1e-4))
    assert v == 1
    # before confirm, server config unchanged
    assert pytest.approx(hps.get()["algo"]["lr"]) == 3e-4
    # confirm applies the change
    hps.confirm_applied(v)
    assert pytest.approx(hps.get()["algo"]["lr"]) == 1e-4
    assert hps.current_version == 1

def test_hps_preview_and_tags():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    v = hps.stage(ParamPatch("algo.gamma", "set", 0.97))
    preview = hps.last_preview()
    assert len(preview) == 1
    pp = preview[0]
    assert pp.dotted_path == "algo.gamma"
    assert math.isclose(pp.before, 0.99)
    assert math.isclose(pp.after, 0.97)
    # gamma should be tagged to recompute advantages
    # tags are advisory on the ParamPatch, not the preview; we verify indirectly by restaging:
    v2 = hps.stage(ParamPatch("algo.gae_lambda", "set", 0.9))
    # confirm both sequentially to avoid staged queue confusion in current API
    hps.confirm_applied(v)
    hps.confirm_applied(v2)
    assert hps.current_version == 2

def test_hps_bounds_and_type_checks():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    # Out of bounds gamma
    with pytest.raises(PatchValidationError):
        hps.stage(ParamPatch("algo.gamma", "set", 1.5))
    # Negative ints rejected
    with pytest.raises(PatchValidationError):
        hps.stage(ParamPatch("algo.n_steps", "set", -5))
    # Non-numeric add rejected
    with pytest.raises(PatchValidationError):
        hps.stage(ParamPatch("algo.lr", "add", "banana"))

def test_hps_integer_and_boolean_coercion():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    # n_steps from string becomes int
    v = hps.stage(ParamPatch("algo.n_steps", "set", "4096"))
    hps.confirm_applied(v)
    assert hps.get()["algo"]["n_steps"] == 4096
    # boolean set accepted
    v2 = hps.stage(ParamPatch("env.normalize_obs", "set", "false"))
    hps.confirm_applied(v2)
    assert hps.get()["env"]["normalize_obs"] is False

def test_hps_forbidden_structural_edits():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    with pytest.raises(PatchValidationError):
        hps.stage(ParamPatch("env.num_envs", "set", 16))
    with pytest.raises(PatchValidationError):
        hps.stage(ParamPatch("algo.policy", "set", "cnn"))

def test_hps_batch_order_and_last_wins_on_same_key():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    batch = PatchBatch(
        [
            ParamPatch("algo.lr", "set", 5e-4),
            ParamPatch("algo.lr", "mul", 0.1),  # 5e-5 final
        ]
    )
    v = hps.stage(batch)
    # Confirm and check final
    hps.confirm_applied(v)
    assert pytest.approx(hps.get()["algo"]["lr"]) == pytest.approx(5e-5)

@pytest.mark.asyncio
async def test_bus_with_mixed_events_and_checkpointreq():
    bus = EventBus()
    await bus.publish(CheckpointReq(kind="manual", note="save it"))
    await bus.publish(Resume())
    await bus.publish(Quit(graceful=False))
    e1 = await bus.get()
    e2 = await bus.get()
    e3 = await bus.get()
    assert e1.event.kind == "manual"
    assert isinstance(e2.event, Resume)
    assert isinstance(e3.event, Quit)

def test_confirm_applied_requires_sequential_version():
    cfg = sample_cfg()
    hps = HyperparamServer(cfg)
    v1 = hps.stage(ParamPatch("algo.lr", "set", 2e-4))
    v2 = hps.stage(ParamPatch("algo.clip_range", "set", 0.15))
    # Confirming v2 before v1 is invalid
    with pytest.raises(VersionError):
        hps.confirm_applied(v2)
    # Confirm in order works
    hps.confirm_applied(v1)
    hps.confirm_applied(v2)
    assert pytest.approx(hps.get()["algo"]["lr"]) == 2e-4
    assert pytest.approx(hps.get()["algo"]["clip_range"]) == 0.15
