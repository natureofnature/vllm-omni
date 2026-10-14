# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One bounded summary of committed text, with recent turns kept verbatim."""

from dataclasses import dataclass

from vllm_omni.engine.duplex.plugin import duplex_history_digest

RECENT_MESSAGES = 8
SUMMARY_MAX_TOKENS = 512
SUMMARY_CHUNK_MESSAGES = 8


@dataclass
class JoyAISummaryMemory:
    text: str = ""
    covered_messages: int = 0
    covered_digest: str = ""

    def validate(self, history: tuple[tuple[str, str], ...]) -> None:
        """A playback correction or conversation edit revokes derived memory."""
        if self.covered_messages and (
            len(history) < self.covered_messages
            or duplex_history_digest(history[: self.covered_messages]) != self.covered_digest
        ):
            self.text = ""
            self.covered_messages = 0
            self.covered_digest = ""
