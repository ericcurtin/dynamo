# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

if "PYTHONHASHSEED" not in os.environ:
    os.environ["PYTHONHASHSEED"] = "0"

if __name__ == "__main__":
    # Opt-in (DYN_ENABLE_MEDIA_DECODERS, off by default): install this backend's
    # media-decoder package(s) before the first lazy cv2/av import at request
    # time. Omni is a separate entrypoint from `python -m dynamo.vllm`:
    # `-m pkg.sub` runs pkg/sub/__main__.py and never pkg/__main__.py, so the
    # call there does not cover this one and the switch was silently inert here.
    #
    # Ahead of importing omni.main so the install precedes any decoder import
    # that module may pull in, matching the ordering in vllm/__main__.py.
    from dynamo.common.utils.media_decoders import maybe_install_media_decoders

    maybe_install_media_decoders("vllm")

    from dynamo.vllm.omni.main import main

    main()
