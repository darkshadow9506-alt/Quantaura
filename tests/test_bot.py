"""Unit tests for the bot's safe Markdown sending (no network/token needed)."""
import asyncio

from telegram.error import BadRequest

from quantaura.bot import _plain, _reply


class FakeMessage:
    def __init__(self, fail_markdown: bool = False):
        self.fail_markdown = fail_markdown
        self.calls = []  # list of (text, parse_mode)

    async def reply_text(self, text, parse_mode=None):
        self.calls.append((text, parse_mode))
        if parse_mode is not None and self.fail_markdown:
            raise BadRequest("Can't parse entities")


def test_plain_strips_markdown():
    assert _plain("*bold* `code` x") == "bold code x"


def test_reply_markdown_ok():
    m = FakeMessage(fail_markdown=False)
    asyncio.run(_reply(m, "*hi*"))
    assert len(m.calls) == 1
    assert m.calls[0][1] is not None      # sent with parse_mode


def test_reply_falls_back_to_plain_on_badrequest():
    m = FakeMessage(fail_markdown=True)
    asyncio.run(_reply(m, "*hi* `x` mean_reversion"))
    # first attempt Markdown (raised), second attempt plain text
    assert len(m.calls) == 2
    assert m.calls[1][1] is None
    assert m.calls[1][0] == "hi x mean_reversion"


def test_status_strategy_line_has_no_raw_underscore_markdown():
    # the strategy list must be inside a code span so legacy Markdown is safe
    from quantaura import bot
    import inspect
    src = inspect.getsource(bot.cmd_status)
    assert "`trend, macd, dual_thrust" in src  # wrapped in backticks
