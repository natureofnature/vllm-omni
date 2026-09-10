# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Shared Realtime protocol translation for engine-owned sessions.

``commands`` maps wire input to the framework's typed commands. ``projection``
maintains conversation/response presentation state inside the engine session;
it does not own execution, scheduling, or model selection. Model integration
continues to use ``engine.duplex.plugin.DuplexModelPlugin``.
"""
