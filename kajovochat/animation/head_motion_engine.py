from __future__ import annotations

import math

from .types import HeadMotionFrame


class HeadMotionEngine:
    def update(
        self,
        *,
        now: float,
        state: str,
        speech_energy: float,
        speaking_attack: float,
    ) -> HeadMotionFrame:
        breathing_x = math.sin(now * 0.43) * 0.028
        breathing_y = math.sin(now * 0.86) * 0.050
        breathing_rot = math.sin(now * 0.57) * 0.030

        tx = breathing_x
        ty = breathing_y
        rot = breathing_rot
        neck = 0.18

        if state == "listening":
            tx += 0.060 + math.sin(now * 0.92) * 0.010
            ty -= 0.020
            rot += 0.050
            neck = 0.32
        elif state == "thinking":
            tx -= 0.045
            ty -= 0.030 + abs(math.sin(now * 0.62)) * 0.020
            rot -= 0.070
            neck = 0.36
        elif state == "speaking":
            nod = speech_energy * 0.11 + speaking_attack * 0.28
            attack_nod = max(0.0, speaking_attack - 0.10) * 0.18
            asym = math.sin(now * 2.1) * (0.010 + speech_energy * 0.008)
            ty += math.sin(now * (1.7 + speech_energy * 2.8)) * (0.042 + speech_energy * 0.040)
            ty -= nod + attack_nod
            tx += math.sin(now * 1.35) * (0.016 + speech_energy * 0.014) + asym
            rot += math.sin(now * 3.0) * (0.016 + speech_energy * 0.022) + asym * 0.70
            neck = min(1.0, 0.26 + speech_energy * 0.30 + speaking_attack * 0.22)
        elif state == "error":
            tx *= 0.25
            ty *= 0.25
            rot *= 0.20
            neck = 0.08

        return HeadMotionFrame(
            timestamp_s=now,
            head_tx=max(-1.0, min(1.0, tx)),
            head_ty=max(-1.0, min(1.0, ty)),
            head_rot=max(-1.0, min(1.0, rot)),
            neck_compensation=max(0.0, min(1.0, neck)),
        )
