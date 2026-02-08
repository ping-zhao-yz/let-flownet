import numpy as np
import os


class EventParser(object):
    def __init__(self, event_dir, gray_dir, width=346, height=260):
        self.event_dir = event_dir
        self.gray_dir = gray_dir
        self.width = width
        self.height = height

    def generate_event_frames(self, events=0, gray_image=0, event_inds=0, gray_image_ts=0):
        group_count = gray_image_ts.shape[0]
        sub_group_count = 10

        event_frames = np.zeros((2, self.height, self.width,
                                 sub_group_count), dtype=np.uint8)

        #  group index
        for g_idx in range(group_count):
            if g_idx == 0:
                group_events = events[0: event_inds[g_idx], :]
            else:
                group_events = events[event_inds[g_idx - 1]: event_inds[g_idx], :]

            #
            # for each group of events:
            # - first subdivide into sub groups,
            # - then, for each sub group, accrue events from neightbouring sequences to form an aggregated event frame
            #
            if group_events.size > 0:
                event_frames.fill(0)
                sub_group_size = group_events.shape[0] / sub_group_count

                # sub group index
                for sub_g_idx in range(sub_group_count):

                    # frame index (relative within sub group)
                    for f_idx_rel in range(int(sub_group_size)):
                        #  frame index (absolute within group)
                        f_idx = int(sub_group_size) * sub_g_idx + f_idx_rel

                        # blend all the event frames from the same sub-group into one event frame, hence losing time granularity
                        if group_events[f_idx, 3].item() == 1:
                            event_frames[0,
                                         group_events[f_idx, 1].astype(int),
                                         group_events[f_idx, 0].astype(int),
                                         sub_g_idx] += 1

                        elif group_events[f_idx, 3].item() == -1:
                            event_frames[1,
                                         group_events[f_idx, 1].astype(int),
                                         group_events[f_idx, 0].astype(int),
                                         sub_g_idx] += 1

                np.save(os.path.join(self.event_dir, str(g_idx)), event_frames)
                np.save(os.path.join(self.gray_dir, str(g_idx)),
                        gray_image[g_idx, :, :])
