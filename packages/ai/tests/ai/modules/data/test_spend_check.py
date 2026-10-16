"""
The task asks EaaS for credit right before getPipe().

DataConn.ask_spend sends the question over the data channel EaaS opened and
waits for its answer; require_credit, the engine's pipe gate, turns a refusal
into OutOfCreditsError.
"""

from __future__ import annotations

import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import ai.modules.data.data_server as data_server
from ai.account.base import OutOfCreditsError
from ai.modules.data.data_conn import DataConn

REFUSAL = 'Out of credits'


def _conn():
    conn = DataConn.__new__(DataConn)
    conn._asked = {}
    conn._seq = 0
    conn._server = MagicMock()
    conn.debug_message = MagicMock()
    conn.send = AsyncMock()
    return conn


@pytest.fixture
def eaas(monkeypatch):
    """A running loop in another thread, like the task's web server, with a connection on it."""
    loop = asyncio.new_event_loop()
    thread = threading.Thread(target=loop.run_forever, daemon=True)
    thread.start()
    conn = _conn()
    conn.ask_spend = AsyncMock(return_value=None)
    monkeypatch.setattr(data_server, '_eaas', (conn, loop))
    yield conn
    loop.call_soon_threadsafe(loop.stop)
    thread.join(5)


@pytest.mark.asyncio
async def test_ask_spend_returns_the_refusal_eaas_sends():
    conn = _conn()
    asking = asyncio.ensure_future(conn.ask_spend())
    await asyncio.sleep(0)

    request = conn.send.await_args.args[0]
    assert request['command'] == 'rrext_check_spend'
    await conn.on_receive({'type': 'response', 'request_seq': request['seq'], 'body': {'refusal': REFUSAL}})
    assert await asking == REFUSAL


def test_require_credit_refuses_when_eaas_refuses(eaas):
    eaas.ask_spend.return_value = REFUSAL
    with pytest.raises(OutOfCreditsError, match=REFUSAL) as exc_info:
        data_server.require_credit()
    assert exc_info.value.code == 'OUT_OF_CREDITS'


def test_require_credit_lets_the_pipe_through_when_eaas_allows(eaas):
    data_server.require_credit()
    eaas.ask_spend.assert_awaited_once()


def test_a_task_not_run_by_eaas_has_no_one_to_ask(monkeypatch):
    monkeypatch.setattr(data_server, '_eaas', None)
    data_server.require_credit()


@pytest.mark.asyncio
async def test_eaas_connecting_makes_require_credit_the_pipe_gate(monkeypatch):
    server = data_server.DataServer.__new__(data_server.DataServer)
    server._dapbase_on_connected = AsyncMock()
    server._dapbase_on_disconnected = AsyncMock()
    set_gate = MagicMock()
    monkeypatch.setattr(data_server.engLib, 'setPipeGate', set_gate)
    monkeypatch.setattr(data_server, 'DataConn', lambda **kwargs: _conn())
    seen = {}

    async def accept(websocket):
        seen['while_connected'] = data_server._eaas

    monkeypatch.setattr(data_server, 'TransportWebSocket', lambda: MagicMock(accept=accept))
    await server.listen(MagicMock())

    set_gate.assert_called_once_with(data_server.require_credit)
    assert seen['while_connected'] is not None
    assert data_server._eaas is None


@pytest.mark.asyncio
async def test_ask_spend_raises_the_error_eaas_answers():
    conn = _conn()
    asking = asyncio.ensure_future(conn.ask_spend())
    await asyncio.sleep(0)

    request = conn.send.await_args.args[0]
    failed = {'type': 'response', 'request_seq': request['seq'], 'success': False, 'message': 'database unavailable'}
    await conn.on_receive(failed)
    with pytest.raises(RuntimeError, match='database unavailable'):
        await asking
    assert conn._asked == {}


def test_a_question_eaas_never_answers_times_out_and_is_dropped(eaas, monkeypatch):
    monkeypatch.setattr(data_server, 'CONST_SPEND_CHECK_TIMEOUT', 0.05)
    dropped = threading.Event()

    async def never_answers():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            dropped.set()
            raise

    eaas.ask_spend = never_answers
    with pytest.raises(TimeoutError):
        data_server.require_credit()
    assert dropped.wait(1)


@pytest.mark.asyncio
async def test_a_cancelled_question_leaves_nothing_waiting():
    conn = _conn()
    asking = asyncio.ensure_future(conn.ask_spend())
    await asyncio.sleep(0)

    asking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asking
    assert conn._asked == {}
