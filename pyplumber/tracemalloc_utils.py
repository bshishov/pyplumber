from typing import Optional, NamedTuple, List

import tracemalloc
import uuid
import datetime
import logging


__all__ = ['SnapshotRecord', 'MemoryManager']


logger = logging.getLogger()


def snapshot_traces_size(snapshot: tracemalloc.Snapshot):
    total_size = 0
    for trace in snapshot.traces._traces:
        domain, size, trace_traceback = trace
        total_size += size
    return total_size


class SnapshotRecord:
    def __init__(self, id: str, created: datetime.datetime, snapshot: tracemalloc.Snapshot):
        self.id = id
        self.created = created
        self.snapshot = snapshot
        self.size = snapshot_traces_size(snapshot)
        self.previous_snapshot_id = None


class MemoryManager:
    def __init__(self, n_frames=10):
        self.snapshots: List[SnapshotRecord] = []
        self.n_frames = n_frames

    def start_tracing(self, n_frames: Optional[int] = None):
        n_frames = n_frames or self.n_frames
        logger.info(f'Started tracemalloc tracing with n_frames={n_frames}')
        tracemalloc.start(n_frames)

    def stop_tracing(self):
        logger.info('Stopped tracemalloc tracing')
        tracemalloc.stop()

    def is_tracing(self):
        return tracemalloc.is_tracing()

    def get_traced_memory(self):
        return tracemalloc.get_traced_memory()

    def get_tracemalloc_memory(self):
        return tracemalloc.get_tracemalloc_memory()

    def take_snapshot(self, filters: Optional[List[tracemalloc.Filter]] = None) -> SnapshotRecord:
        logger.info('Taking snapshot...')
        snapshot = tracemalloc.take_snapshot()

        if filters:
            snapshot = snapshot.filter_traces(filters)

        snapshot_record = SnapshotRecord(
            id=str(uuid.uuid4())[:4],
            created=datetime.datetime.now(),
            snapshot=snapshot
        )
        self.snapshots.insert(0, snapshot_record)
        self.snapshots.sort(key=lambda s: s.created, reverse=True)
        self._recalc_prev_ids()
        logger.info(f'New snapshot: {snapshot_record.id}')
        return snapshot_record

    def _recalc_prev_ids(self):
        # assuming ordering by created
        prev: Optional[SnapshotRecord] = None
        for s in reversed(self.snapshots):
            if prev:
                s.previous_snapshot_id = prev.id
            prev = s

    def get_snapshot(self, snapshot_id: str) -> Optional[SnapshotRecord]:
        for snapshot_record in self.snapshots:
            if snapshot_record.id == snapshot_id:
                return snapshot_record
        return None

    def delete_snapshot(self, snapshot_id: str):
        for i, snapshot_record in enumerate(self.snapshots):
            if snapshot_record.id == snapshot_id:
                self.snapshots.pop(i)
        self._recalc_prev_ids()

    def clear(self):
        self.snapshots.clear()
        tracemalloc.clear_traces()

