'''
    this module defines the control-plane events(pause, resume, hyperparam adjusts and ...)
    using asynchi0-based bus to deliver the to trainer at safe points
'''
from __future__ import annotations
import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Iterable, Literal, Optional, Sequence, Tuple, Union

'''
    Core types aliases and Literals
'''
# trainer only pause at these "safe-points"
SafePoint = Literal["after_callect", "after_update", "end_of_epoch"]

# Operation semantics for hyperparam patches
PatchOp = Literal["set", "add", "null"]

'''
    Event Dataclasses
'''
@dataclass(frozen=True, slots=True)
class Pause:
    """
        Request a safe pause the trainer now or at the earlies next safe point
    """
    reason: str = "user"
    target: Optional[SafePoint] = None
    
@dataclass(frozen=True, slots=True)
class Resume:
    """
        resume training from last checkpoint created during a pause
    """
    # if checkpoint path provided, trained restore from that path explicitally otherwise from most recent checkpoint
    checkpoint_path: Optional[str] = None
    
@dataclass(frozen=True, slots=True)
class Quit:
    """
        terminating training after finishing current exploration(if false)
        //TODO should think about if the data store after quitting happens where
    """
    graceful: bool = True
    
@dataclass(frozen=True, slots=True)
class CheckpointReq:
    """
        explicitally request a checkpoint
    """
    #optional, trainer may choose a run-managed location instead
    path: optional[str] = None
    #tag for logging type
    kind: Literal["manual", "auto", "pre-paused", "prequit"] = "manual"
    #if true, mark the checkpoint as protected from pruning
    keep: bool = False
    #free form text for human(or maybe llms instead of humans later)
    note: Optional[str] = None
    
@dataclass(frozen=True, slots=True)
class ParamPatch:
    """
        patch a hyperparam at runtime in a validated way
    """
    # hierarchical key as tuple. dottedstring type provided via constructor too(like "algo.lr")
    path:Tuple[str, ...]
    #operation to apply: "set", "add" or "mul"
    op: PatchOp
    #new value
    value: Any
    #aditional notes for logs, optional semantictags and hyperparamserver versioning
    note: Optional[str] = None
    tags: Tuple[str, ...] = field(default_factory=tuple)
    version: Optional[int] = None

    def __init__(
        self,
        path: Union[str, Sequence[str]],
        op: PatchOp,
        value: Any,
        note: Optional[str] = None,
        tags: Optional[Iterable[str]] = None,
        version: Optional[int] = None,
    ) -> None:
        #path to tuple
        if isinstance(path, str):
            norm = tuple(p.strip() for p in path.split(".") if p.strip())
        else: norm = tuple(str(p).strip() for p in path if str(p).strip())
        if not norm:
            raise ValueError("ParamaPatch.path is empty")
        object.__setattr__(self, "path", norm)
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "note", note)
        object.__setattr__(self, "tags", tuple(tags) if tags is not None else tuple())
        object.__setattr__(self, "version", version)
        
    @property
    def dotted(self) -> str:
        #dotted representation
        return ".".join(self.path)
    
@dataclass(frozen=True, slots=True)
class PatchBatch:
    """
        apply multi ParamPatch update automatically at the next safe point
    """
    patches = Tuple[ParamPatch, ...]
    note: Optional[str] = None
    
    def __init__(self, patches: Iterable[ParamPatch], note: Optional[str] = None) -> None:
        _patches = tuple(patches)
        if not _patches:
            raise ValueError("PatBatch must contain at least 1 ParamPatch")
        object.__setattr__(self, "patches", _patches)
        object.__setattr__(self, "note", note)
        
# Union of all events that the trainer know
Event = Union[Pause, Resume, Quit, CheckpointReq, ParamPatch, PatchBatch]

"""
    Event envelope and bus
"""
@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """
        metadata wrapper used on the bus to stamp, trace and order events
    """
    #UUID4 string
    id: str
    #wall-clock seconds since epoch(time.time())
    ts_wall: float
    #monotonic seconds(time.pre_counter()) for ordering across clock jumps
    ts_mono: float
    #logical origin like cli or api or autosaver
    source: str
    event: Event
    
    @staticmethod
    def wrap(event: Event, source: str = "cli") -> "EventEnvelope":
        return EventEnvelope(
            id = str(uuid.uuid4()),
            ts_wall = time.time(),
            ts_mono=time.pref_counter(),
            source = source,
            event = event
        )
        
class EventBus:
    """
        async single consumer event bus
    """
    def __init__(self, *, maxsize: int = 0) -> None:
        self._q: asyncio.Queue[EventEnvelope] = asyncio.Queue(maxsize=maxsize)
        
    async def publish(self, event: Event, *, source: str = "cli") -> str:
        env = EventEnvelope.wrap(event, source)
        await self._q.put(env)
        return env.id

    def publish_nowait(self, event: Event, *, source: str = "cli") -> str:
        env = EventEnvelope.wrap(event, source)
        self._q.put_nowait(env)
        return env.id

    async def get(self) -> EventEnvelope:
        return await self._q.get()

    async def wait_for(
        self,
        predicate: Callable[[EventEnvelope], bool],
        timeout: Optional[float] = None,
    ) -> EventEnvelope:
        deadline = None if timeout is None else (time.monotonic() + timeout)
        stash: list[EventEnvelope] = []
        try:
            while True:
                if deadline is not None and time.monotonic() >= deadline:
                    raise asyncio.TimeoutError
                env = await asyncio.wait_for(self._q.get(), timeout=0.05 if deadline else None)
                if predicate(env):
                    for s in stash:
                        self._q.put_nowait(s)
                    return env
                stash.append(env)
        finally:
            for s in stash:
                self._q.put_nowait(s)

    def empty(self) -> bool:
        return self._q.empty()
    
"""
    Helper predicates
"""
def is_pause(e: EventEnvelope) -> bool:
    return isinstance(e.event, Pause)


def is_resume(e: EventEnvelope) -> bool:
    return isinstance(e.event, Resume)


def is_quit(e: EventEnvelope) -> bool:
    return isinstance(e.event, Quit)


def is_checkpoint(e: EventEnvelope) -> bool:
    return isinstance(e.event, CheckpointReq)


def is_param_patch(e: EventEnvelope) -> bool:
    return isinstance(e.event, ParamPatch)


def is_patch_batch(e: EventEnvelope) -> bool:
    return isinstance(e.event, PatchBatch)
    


__all__ = [
    "SafePoint",
    "PatchOp",
    "Pause",
    "Resume",
    "Quit",
    "CheckpointReq",
    "ParamPatch",
    "PatchBatch",
    "Event",
    "EventEnvelope",
    "EventBus",
    "is_pause",
    "is_resume",
    "is_quit",
    "is_checkpoint",
    "is_param_patch",
    "is_patch_batch",
]
    
    
    