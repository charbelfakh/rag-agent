"""Sprint A unit tests: asyncio.to_thread SSE iteration pattern."""
import asyncio

import pytest


def _make_stream_puller():
    """Mirror the /query SSE helper in api/main.py."""
    _STREAM_END = object()

    def _pull_next_event(iterator):
        try:
            return next(iterator)
        except StopIteration:
            return _STREAM_END

    async def collect_events(sync_iterator):
        collected = []
        while True:
            event = await asyncio.to_thread(_pull_next_event, sync_iterator)
            if event is _STREAM_END:
                break
            collected.append(event)
        return collected

    return collect_events


class TestAsyncStreamPuller:
    @pytest.mark.asyncio
    async def test_collects_all_sync_generator_events(self):
        events = [
            {"token": "Hello"},
            {"citations": [], "done": True, "meta": {}},
        ]
        collect = _make_stream_puller()
        result = await collect(iter(events))
        assert result == events

    @pytest.mark.asyncio
    async def test_uses_to_thread_per_event(self, monkeypatch):
        events = [{"token": "a"}, {"token": "b"}]
        call_count = {"n": 0}
        original = asyncio.to_thread

        async def counting_to_thread(func, *args, **kwargs):
            call_count["n"] += 1
            return await original(func, *args, **kwargs)

        monkeypatch.setattr(asyncio, "to_thread", counting_to_thread)
        collect = _make_stream_puller()
        await collect(iter(events))
        # Two events plus one StopIteration pull.
        assert call_count["n"] == 3
