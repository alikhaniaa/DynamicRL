from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional, Sequence, Tuple, Union

"""
    communication structure(control plane) and a common lang for the framework
    implemend asyinc-based event-bus and immutable dataclasses
"""
#type aliases of this module
SafePoint = Literal["end_of_rollout", "between_epochs", "end_of_iteration"]
PatchOp = Literal["set", "add", "mul"]


"""event dataclass definitions"""

#pause the training loop at the --next safe point--
@dataclass(frozen=True, slots=True)
class Pause:
    reason: str = "user_request"
    
#resume the paused training
@dataclass(frozen=True, slots=True)
class Resume:
    pass

#request for shutdown of the training
@dataclass(frozen=True, slots=True)
class Quit:
    pass

#request for immediate checkpoint of the training state
@dataclass(frozen=True, slots=True)
class CheckpointReq:
    #optional tag for the checkpoint
    tag: Optional[str] = None 

#request change to a --single-- hyperparam
@dataclass(frozen=True, slots=True)
class ParamPatch:
    #dot notation path
    path: str
    op: PatchOp
    value: Any
    
#multi batch of hyperparam changes
@dataclass(frozen=True, slots=True)
class PatchBatch:
    patches: Tuple[ParamPatch, ...]
    
"""
    Union type for all possible control plane events
"""
ControlEvent = Union[Pause, Resume, Quit, CheckpointReq, ParamPatch, PatchBatch]

# Event bus imp -- wrapper for events, adding metadata for logging and debug mode
@dataclass(frozen=True, slots=True)
class EventEnvelope:
    event: ControlEvent
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    source_id: str = "default"
    
# Async, Thread-safe message queue for control plane events. central communicate channel between API and CLI and if anything more added to trainer
class EventBus:
    def __init__(self):
        self._queue: asyncio.Queue[EventEnvelope] = asyncio.Queue()
        
    #publish events to the bus
    async def publish(self, event:ControlEvent, source_id: str = "cli"):
        envelope = EventEnvelope(event=event, source_id=source_id)
        await self._queue.put(envelope)
        
    #retrive currently pending events from bus without blocking
    async def get_all_pending(self) -> list[EventEnvelope]:
        events = []
        while not self._queue.empty():
            try:
                events.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return events
    










