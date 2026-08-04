// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package injection

const (
	agentBinDir = "/snapshot-binaries"

	// SnapshotBinDir is the destination path inside the placeholder namespace
	// where the agent's binary bundle is mounted.
	SnapshotBinDir = "/tmp" + agentBinDir
)
