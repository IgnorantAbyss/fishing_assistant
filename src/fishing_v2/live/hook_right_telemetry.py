"""Bounded passive right target, serviced AFTER action dispatch. No I/O."""
from collections import deque
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

    def finish(self, reason):
        if self.active is not None:
            self.active['terminal_reason'] = reason
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
            right = snapshot['confirmed_x'] if snapshot['confirmed_x'] is not None else row['fillable_right_x']
            fill, divider = row.get('fill_endpoint_x'), row.get('divider_x')
            if (right is not None and fill is not None and divider is not None
                    and 0 <= right-fill <= .1*(right-divider)
                    and self.active['first_near_full_frame'] is None):
                self.active['first_near_full_frame'] = observation.frame_index
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
        return dict(total_segments=self.total_segments,retained_segment_cap=self.segments.maxlen,
                    dropped_segments=max(0,self.total_segments-len(self.segments)-(self.active is not None)),
                    segments=list(self.segments)+([dict(self.active)] if self.active else []),
                    failure_count=self.failure_count,cost_sample_count=len(self.costs),
                    cost_sample_cap=self.costs.maxlen,costs=timings,
                    near_full_definition='last_10_percent_of_divider_to_right_span_telemetry_only')
