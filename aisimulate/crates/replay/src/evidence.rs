// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use uuid::Uuid;

use aisimulate_engine::{NativePressureEvent, NativePressureKind, NativePressureState};

use crate::{ReplayCaptureOptions, TraceCollector};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerPool {
    Agg,
    Prefill,
    Decode,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerLifecycleTransitionKind {
    WorkerStarting,
    WorkerReady,
    WorkerDraining,
    WorkerRemoved,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct WorkerLifecycleTransition {
    pub worker_id: usize,
    pub transition: WorkerLifecycleTransitionKind,
    pub prior_state: Option<&'static str>,
    pub state: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub origin_operation_ordinal: Option<u64>,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
pub struct WorkerPoolState {
    pub active: Vec<usize>,
    pub starting: Vec<usize>,
    pub draining: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LifecycleOperation {
    pub operation_ordinal: u64,
    pub at_ms: f64,
    pub pool: WorkerPool,
    pub cause: &'static str,
    pub planner_tick_ordinal: Option<u64>,
    pub origin_operation_ordinal: Option<u64>,
    pub transitions: Vec<WorkerLifecycleTransition>,
    pub state_after_batch: WorkerPoolState,
    pub topology_released_request_uuids: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PressureKind {
    VllmPreemption,
    SglangRetraction,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Serialize)]
pub struct EnginePressureState {
    pub running_requests: usize,
    pub waiting_requests: Option<usize>,
    pub active_blocks: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PressureRecord {
    pub pressure_ordinal: u64,
    pub at_ms: f64,
    pub pool: WorkerPool,
    pub worker_id: u64,
    pub dp_rank: u32,
    pub kind: PressureKind,
    pub request_uuid: String,
    pub state_before: EnginePressureState,
    pub state_after: EnginePressureState,
    pub request_active_blocks_before: usize,
    pub logical_available_blocks_before: Option<usize>,
    pub required_blocks_before: Option<usize>,
    pub readmitted_at_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PressureEvidence {
    pub records: Vec<PressureRecord>,
    pub vllm_preemptions_total: u64,
    pub sglang_retractions_total: u64,
}

/// Evidence owned and returned by one replay execution.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct OfflineRuntimeEvidence {
    pub lifecycle_operations: Vec<LifecycleOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pressure: Option<PressureEvidence>,
}

/// Explicit execution-local evidence state. It deliberately is not stored in
/// a thread-local: two replays on one thread cannot observe or overwrite each
/// other's capture state.
#[derive(Debug)]
pub(crate) struct ReplayEvidenceCollector {
    options: ReplayCaptureOptions,
    lifecycle_operations: Vec<LifecycleOperation>,
    pressure_records: Vec<PressureRecord>,
    outstanding_pressure: BTreeMap<(Uuid, WorkerPool), Vec<u64>>,
    startup_origins: BTreeMap<(WorkerPool, usize), u64>,
    drain_origins: BTreeMap<(WorkerPool, usize), u64>,
}

impl ReplayEvidenceCollector {
    pub(crate) fn new(options: ReplayCaptureOptions) -> Self {
        Self {
            options,
            lifecycle_operations: Vec::new(),
            pressure_records: Vec::new(),
            outstanding_pressure: BTreeMap::new(),
            startup_origins: BTreeMap::new(),
            drain_origins: BTreeMap::new(),
        }
    }

    pub(crate) fn options(&self) -> ReplayCaptureOptions {
        self.options
    }

    pub(crate) fn startup_origin(&self, pool: WorkerPool, worker_id: usize) -> Option<u64> {
        self.startup_origins.get(&(pool, worker_id)).copied()
    }

    pub(crate) fn drain_origin(&self, pool: WorkerPool, worker_id: usize) -> Option<u64> {
        self.drain_origins.get(&(pool, worker_id)).copied()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_lifecycle_operation(
        &mut self,
        at_ms: f64,
        pool: WorkerPool,
        cause: &'static str,
        planner_tick_ordinal: Option<u64>,
        origin_operation_ordinal: Option<u64>,
        mut transitions: Vec<WorkerLifecycleTransition>,
        state_after_batch: WorkerPoolState,
        topology_released_request_uuids: Vec<Uuid>,
    ) -> Option<u64> {
        if !self.options.capture_lifecycle_evidence
            || (transitions.is_empty() && topology_released_request_uuids.is_empty())
        {
            return None;
        }

        let operation_ordinal = u64::try_from(self.lifecycle_operations.len()).ok()?;
        for transition in &mut transitions {
            if transition.origin_operation_ordinal.is_none() {
                transition.origin_operation_ordinal = Some(operation_ordinal);
            }
            let key = (pool, transition.worker_id);
            match transition.transition {
                WorkerLifecycleTransitionKind::WorkerStarting => {
                    self.startup_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerDraining => {
                    self.drain_origins.insert(key, operation_ordinal);
                }
                WorkerLifecycleTransitionKind::WorkerReady => {
                    self.startup_origins.remove(&key);
                }
                WorkerLifecycleTransitionKind::WorkerRemoved => {
                    self.startup_origins.remove(&key);
                    self.drain_origins.remove(&key);
                }
            }
        }
        let mut seen = BTreeSet::new();
        self.lifecycle_operations.push(LifecycleOperation {
            operation_ordinal,
            at_ms,
            pool,
            cause,
            planner_tick_ordinal,
            origin_operation_ordinal,
            transitions,
            state_after_batch,
            topology_released_request_uuids: topology_released_request_uuids
                .into_iter()
                .filter(|uuid| seen.insert(*uuid))
                .map(|uuid| uuid.to_string())
                .collect(),
        });
        Some(operation_ordinal)
    }

    /// Record one engine pressure transition and attach its stable ordinal to
    /// the matching per-request report record.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record_pressure(
        &mut self,
        collector: &mut TraceCollector,
        at_ms: f64,
        pool: WorkerPool,
        worker_id: u64,
        dp_rank: u32,
        kind: PressureKind,
        request_uuid: Uuid,
        state_before: EnginePressureState,
        state_after: EnginePressureState,
        request_active_blocks_before: usize,
        logical_available_blocks_before: Option<usize>,
        required_blocks_before: Option<usize>,
    ) -> Option<u64> {
        if !self.options.capture_canonical_evidence {
            return None;
        }
        let pressure_ordinal = u64::try_from(self.pressure_records.len()).ok()?;
        self.pressure_records.push(PressureRecord {
            pressure_ordinal,
            at_ms,
            pool,
            worker_id,
            dp_rank,
            kind,
            request_uuid: request_uuid.to_string(),
            state_before,
            state_after,
            request_active_blocks_before,
            logical_available_blocks_before,
            required_blocks_before,
            readmitted_at_ms: None,
        });
        self.outstanding_pressure
            .entry((request_uuid, pool))
            .or_default()
            .push(pressure_ordinal);
        collector.on_pressure_reference(request_uuid, pressure_ordinal);
        Some(pressure_ordinal)
    }

    pub(crate) fn record_native_pressure(
        &mut self,
        collector: &mut TraceCollector,
        pool: WorkerPool,
        worker_id: u64,
        dp_rank: u32,
        event: NativePressureEvent,
    ) -> Option<u64> {
        let kind = match event.kind {
            NativePressureKind::VllmPreemption => PressureKind::VllmPreemption,
            NativePressureKind::SglangRetraction => PressureKind::SglangRetraction,
        };
        self.record_pressure(
            collector,
            event.at_ms,
            pool,
            worker_id,
            dp_rank,
            kind,
            event.request_id,
            lower_pressure_state(event.state_before),
            lower_pressure_state(event.state_after),
            event.request_active_blocks_before,
            event.logical_available_blocks_before,
            event.required_blocks_before,
        )
    }

    pub(crate) fn record_pressure_readmission(
        &mut self,
        request_uuid: Uuid,
        pool: WorkerPool,
        at_ms: f64,
    ) {
        if !self.options.capture_canonical_evidence {
            return;
        }
        let key = (request_uuid, pool);
        let Some(ordinals) = self.outstanding_pressure.get_mut(&key) else {
            return;
        };
        let Some(pressure_ordinal) = ordinals.pop() else {
            return;
        };
        let remove_key = ordinals.is_empty();
        if remove_key {
            self.outstanding_pressure.remove(&key);
        }
        if let Some(record) = self
            .pressure_records
            .get_mut(usize::try_from(pressure_ordinal).expect("pressure ordinal must fit usize"))
        {
            record.readmitted_at_ms = Some(at_ms);
        }
    }

    pub(crate) fn finish(self) -> OfflineRuntimeEvidence {
        let pressure = self.options.capture_canonical_evidence.then(|| {
            let vllm_preemptions_total = self
                .pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::VllmPreemption)
                .count() as u64;
            let sglang_retractions_total = self
                .pressure_records
                .iter()
                .filter(|record| record.kind == PressureKind::SglangRetraction)
                .count() as u64;
            PressureEvidence {
                records: self.pressure_records,
                vllm_preemptions_total,
                sglang_retractions_total,
            }
        });
        OfflineRuntimeEvidence {
            lifecycle_operations: self.lifecycle_operations,
            pressure,
        }
    }
}

fn lower_pressure_state(state: NativePressureState) -> EnginePressureState {
    EnginePressureState {
        running_requests: state.running_requests,
        waiting_requests: state.waiting_requests,
        active_blocks: state.active_blocks,
    }
}

impl Default for ReplayEvidenceCollector {
    fn default() -> Self {
        Self::new(ReplayCaptureOptions::default())
    }
}

pub(crate) fn common_origin(mut origins: impl Iterator<Item = u64>) -> Option<u64> {
    let first = origins.next()?;
    origins.all(|origin| origin == first).then_some(first)
}

#[cfg(test)]
mod tests {
    use aisimulate_engine::{NativePressureEvent, NativePressureKind, NativePressureState};
    use uuid::Uuid;

    use crate::{ReplayCaptureOptions, ReplayDeterminism, TraceCollector};

    use super::{
        ReplayEvidenceCollector, WorkerLifecycleTransition, WorkerLifecycleTransitionKind,
        WorkerPool, WorkerPoolState,
    };

    #[test]
    fn execution_local_pressure_ordinals_link_to_request_records() {
        let uuid = Uuid::from_u128(42);
        let mut trace = TraceCollector::default();
        trace.set_capture_per_request(true);
        trace.on_arrival(uuid, 0.0, 4, 2);
        trace.on_admit(uuid, 0.0, 0);
        let mut evidence = ReplayEvidenceCollector::new(ReplayCaptureOptions {
            capture_per_request: true,
            capture_canonical_evidence: true,
            determinism: ReplayDeterminism::CanonicalV1,
            ..Default::default()
        });
        evidence.record_native_pressure(
            &mut trace,
            WorkerPool::Agg,
            7,
            1,
            NativePressureEvent {
                at_ms: 2.0,
                kind: NativePressureKind::VllmPreemption,
                request_id: uuid,
                state_before: NativePressureState {
                    running_requests: 1,
                    waiting_requests: Some(0),
                    active_blocks: 4,
                },
                state_after: NativePressureState {
                    running_requests: 0,
                    waiting_requests: Some(1),
                    active_blocks: 0,
                },
                request_active_blocks_before: 4,
                logical_available_blocks_before: None,
                required_blocks_before: None,
            },
        );
        evidence.record_pressure_readmission(uuid, WorkerPool::Agg, 3.0);
        trace.on_terminal(uuid, 4.0, crate::ReplayTerminalStatus::Completed);
        trace.set_runtime_evidence(evidence.finish());

        let report = trace.finish();
        assert_eq!(report.per_request[0].pressure_record_ordinals, vec![0]);
        let pressure = report.runtime_evidence.pressure.unwrap();
        assert_eq!(pressure.vllm_preemptions_total, 1);
        assert_eq!(pressure.records[0].readmitted_at_ms, Some(3.0));
    }

    #[test]
    fn lifecycle_origins_are_owned_by_one_collector() {
        let mut evidence = ReplayEvidenceCollector::new(ReplayCaptureOptions {
            capture_lifecycle_evidence: true,
            ..Default::default()
        });
        evidence.record_lifecycle_operation(
            1.0,
            WorkerPool::Decode,
            "planner_scale",
            Some(0),
            None,
            vec![WorkerLifecycleTransition {
                worker_id: 3,
                transition: WorkerLifecycleTransitionKind::WorkerStarting,
                prior_state: None,
                state: "starting",
                reason: None,
                origin_operation_ordinal: None,
            }],
            WorkerPoolState {
                starting: vec![3],
                ..Default::default()
            },
            Vec::new(),
        );
        let origin = evidence.startup_origin(WorkerPool::Decode, 3);
        evidence.record_lifecycle_operation(
            2.0,
            WorkerPool::Decode,
            "worker_ready_event",
            None,
            origin,
            vec![WorkerLifecycleTransition {
                worker_id: 3,
                transition: WorkerLifecycleTransitionKind::WorkerReady,
                prior_state: Some("starting"),
                state: "active",
                reason: None,
                origin_operation_ordinal: origin,
            }],
            WorkerPoolState {
                active: vec![3],
                ..Default::default()
            },
            Vec::new(),
        );

        let evidence = evidence.finish();
        assert_eq!(evidence.lifecycle_operations.len(), 2);
        assert_eq!(
            evidence.lifecycle_operations[1].origin_operation_ordinal,
            Some(0)
        );
    }
}
