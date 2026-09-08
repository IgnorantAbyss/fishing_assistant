"""Bounded passive right target, serviced AFTER action dispatch. No I/O."""
from collections import Counter, deque
from time import perf_counter

from src.hook_right_geometry import EpisodeRightTarget


class HookRightTelemetry:
    def __init__(self, *, enabled=True, max_segments=256):
        self.enabled = enabled
        self.target = EpisodeRightTarget()
        self.active = None
        self.segments = deque(maxlen=max_segments)
        self.costs = deque(maxlen=512)
        self.total_segments = 0
        self.failure_count = 0
        self.latest = {}
        self.completed_states = Counter()
        self.invalidation_reasons = Counter()
        self.invalidation_confirmed = Counter()
        self.representatives = {}
        self.near_full_observed = 0
        self.near_full_missing = Counter()
        self.near_full_invalidated_frames = 0

    def finish(self, reason):
        if self.active is not None:
            self.active['terminal_reason'] = reason
            self.completed_states[self.active.get('target_state','unconfirmed')] += 1
            self.segments.append(dict(self.active))
        self.active = None
        self.target = EpisodeRightTarget()
        self.latest = {}

    def observe(self, *, episode, observation, state_before, state_after):
        """Consume canonical output only, not pixels or a detector callable.

        Terminal/action frame is sampled before clearing. SYNC boundaries reset
        confirmation even within the same physical episode; summary segments
        retain physical IDs so analysis can group without double-counting.
        """
        if not self.enabled:
            return {}
        try:
            return self._observe(episode,observation,state_before,state_after)
        except Exception:
            self.failure_count += 1
            self.finish('passive_error')
            return {}

    def _observe(self, episode, observation, state_before, state_after):
        hook_states = {'HOOK_PENDING','HOOK'}
        if episode is None or state_before not in hook_states:
            self.finish('lifecycle_reset_or_sync')
            return {}
        episode_id = episode.hook_episode_id
        if self.active is not None and self.active['hook_episode_id'] != episode_id:
            self.finish('new_physical_episode')
        if self.active is None:
            self.total_segments += 1
            self.active = dict(hook_episode_id=episode_id,cycle_id=episode.cycle_id,
                               segment_index=self.total_segments,frames=0,
                               confirmation_timestamp=None,confirmation_frame=None,
                               confirmation_latency_from_START_HOOK=None,
                               confirmed_fillable_right_x=None,invalidation_reason=None,
                               first_invalidation_reason=None,first_invalidation_frame=None,
                               first_invalidation_timestamp=None,first_invalidation_deltas=None,
                               confirmed_before_invalidation=None,confirmed_target_x=None,
                               current_target_x=None,confirmation_count_before_invalidation=None,
                               first_near_full_frame=None,confirmed_before_near_full=None)
        result = {}
        if observation is not None:
            passive = observation.evidence.get('passive_right')
            identity = getattr(passive,'identity',None)
            if not identity:
                if passive and str(passive.get('fillable_right_reason','')).startswith('passive_error:'):
                    self.failure_count += 1
                self.finish('canonical_passive_observation_unavailable')
                return {}
            row = dict(identity, frame_id=observation.frame_index,timestamp=observation.timestamp)
            start = perf_counter()
            snapshot = self.target.update(episode_id,row)
            update_ms = (perf_counter()-start)*1000
            self.costs.append((passive.get('measurement_cost_ms',0),update_ms))
            self.active['frames'] += 1
            x = snapshot['confirmed_x']
            x = row['roi_bounds'][0]+x if x is not None else None
            result = dict(target_state=snapshot['status'],confirmed_fillable_right_x=x,
                          observation_count=self.target.observation_count,
                          confirmation_frame=snapshot['confirmation_frame'],
                          confirmation_timestamp=snapshot['confirmation_timestamp'],
                          confirmation_latency_from_START_HOOK=(
                              snapshot['confirmation_timestamp']-episode.started_at
                              if snapshot['confirmation_timestamp'] is not None and episode.start_hook_applied else None),
                          invalidation_reason=self.target.invalidation_reason,
                          current_measurement_agrees=(passive['fillable_right_x']==x
                              if passive['fillable_right_x'] is not None and x is not None else None),
                          update_cost_ms=update_ms,hook_episode_id=episode_id,
                          segment_index=self.total_segments)
            self.active.update(result)
            if self.target.invalidated and self.active['first_invalidation_reason'] is None:
                detail = self.target.first_invalidation
                reason = self.target.invalidation_reason
                self.active.update(
                    first_invalidation_reason=reason,
                    first_invalidation_frame=observation.frame_index,
                    first_invalidation_timestamp=observation.timestamp,
                    first_invalidation_deltas=detail,
                    confirmed_before_invalidation=(detail['confirmed_before_invalidation'] if detail else None),
                    confirmed_target_x=(detail['confirmed_target_x'] if detail else None),
                    current_target_x=row.get('fillable_right_x'),
                    confirmation_count_before_invalidation=(detail['confirmation_count_before_invalidation'] if detail else None))
                self.invalidation_reasons[reason] += 1
                self.invalidation_confirmed[
                    'confirmed' if detail and detail['confirmed_before_invalidation'] else
                    'unconfirmed' if detail else 'unknown'] += 1
                examples = self.representatives.setdefault(reason, [])
                if len(examples) < 3:
                    examples.append(dict(segment_index=self.total_segments,
                                         hook_episode_id=episode_id, details=detail))
            right = snapshot['confirmed_x'] if snapshot['confirmed_x'] is not None else row['fillable_right_x']
            fill, divider = row.get('fill_endpoint_x'), row.get('divider_x')
            # Missing fields are overlapping per-observation counters. Being
            # invalidated is context, NOT a prerequisite: current right may work.
            for value, reason in ((fill,'no_fill_endpoint'), (divider,'no_divider'),
                                  (right,'no_confirmed_or_current_right')):
                if value is None:
                    self.near_full_missing[reason] += 1
            if self.target.invalidated:
                self.near_full_invalidated_frames += 1
            if (right is not None and fill is not None and divider is not None
                    and 0 <= right-fill <= .1*(right-divider)
                    and self.active['first_near_full_frame'] is None):
                self.active['first_near_full_frame'] = observation.frame_index
                self.near_full_observed += 1
                self.active['confirmed_before_near_full'] = bool(
                    x is not None and snapshot['confirmation_timestamp'] < observation.timestamp)
            # Public scalar telemetry only; packed masks remain private RAM data.
            passive['confirmation'] = result
            self.latest = result
        if state_after not in hook_states:
            self.finish('lifecycle_terminal_or_sync')
        return result

    def summary(self):
        try:
            return self._summary()
        except Exception:
            return dict(failure_count=self.failure_count+1,summary_reason='passive_summary_error',
                        segments=[],total_segments=self.total_segments)

    def _summary(self):
        import numpy as np
        timings = {}
        if self.costs:
            values = np.asarray(self.costs)
            for key,column in [('measurement',values[:,0]),('confirmation',values[:,1]),
                               ('combined',values.sum(axis=1))]:
                timings[key+'_ms'] = dict(zip(('p50','p95','max'),np.percentile(column,[50,95,100]).tolist()))
        states = self.completed_states.copy()
        if self.active is not None:
            states[self.active.get('target_state','unconfirmed')] += 1
        return dict(total_segments=self.total_segments,retained_segment_cap=self.segments.maxlen,
                    dropped_segments=max(0,self.total_segments-len(self.segments)-(self.active is not None)),
                    segments=list(self.segments)+([dict(self.active)] if self.active else []),
                    failure_count=self.failure_count,cost_sample_count=len(self.costs),
                    cost_sample_cap=self.costs.maxlen,costs=timings,
                    hook_right_final_state_counts={name:states[name] for name in ('confirmed','invalidated','unconfirmed')},
                    hook_right_invalidation_reason_counts=dict(self.invalidation_reasons),
                    hook_right_invalidation_confirmed_before_counts=dict(self.invalidation_confirmed),
                    hook_right_invalidation_representatives=self.representatives,
                    hook_right_representatives_per_reason_cap=3,
                    hook_right_near_full_observed_count=self.near_full_observed,
                    hook_right_near_full_missing_reason_counts=dict(self.near_full_missing),
                    hook_right_near_full_target_invalidated_frame_count=self.near_full_invalidated_frames,
                    aggregate_scope='session_lifetime_segments_including_active',
                    near_full_missing_count_unit='overlapping_observation_counts_not_episodes',
                    near_full_definition='last_10_percent_of_divider_to_right_span_telemetry_only')
