# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Scheduler Classes"""

from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple, Type

import numpy as np
from torch.optim import Optimizer, lr_scheduler

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    # Backwards compatibility for PyTorch 1.x
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.schedulers import SchedulerConfig, Scheduler

@dataclass
class WarmupMultiStepSchedulerConfig(SchedulerConfig):
    """Config for multi step scheduler where lr decays by gamma every milestone"""

    _target: Type = field(default_factory=lambda: WarmupMultiStepScheduler)
    """target class to instantiate"""
    max_steps: int = 1000000
    """The maximum number of steps."""
    gamma: float = 0.33
    """The learning rate decay factor."""
    milestones: Tuple[int, ...] = (500000, 750000, 900000)
    """The milestone steps at which to decay the learning rate."""
    warmup_steps: Optional[int] = None
    """"""


class WarmupMultiStepScheduler(Scheduler):
    """Multi step scheduler where lr decays by gamma every milestone"""

    config: WarmupMultiStepSchedulerConfig

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        scheduler = lr_scheduler.ChainedScheduler(
            [
                # warmup
                lr_scheduler.LinearLR(
                    optimizer, start_factor=0.01, total_iters=self.config.warmup_steps
                ),
                # Linear decay
                lr_scheduler.MultiStepLR(
                    optimizer,
                    milestones=self.config.milestones,
                    gamma=0.33,
                ),
            ]
        )
        return scheduler

