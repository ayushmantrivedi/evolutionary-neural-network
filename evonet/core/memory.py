import numpy as np
import logging
from typing import Dict, Any, Union, Optional

# Try importing torch, but handle if missing
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)

class DirectionalMemory:
    """
    Post-hoc memory recovery for evolutionary neural networks.
    Stores task-specific parameter directions and enables on-demand recall
    without gradients, replay, or retraining.
    """

    def __init__(self, theta_init: Union[np.ndarray, Any]):
        """
        Initialize memory with the starting parameter state (theta_init).
        
        Args:
            theta_init: The initial parameter vector (numpy array or torch tensor).
        """
        if HAS_TORCH and isinstance(theta_init, torch.Tensor):
            self.theta_init = theta_init.clone().detach()
            self.is_torch = True
        else:
            self.theta_init = np.array(theta_init, copy=True)
            self.is_torch = False
            
        self.task_directions: Dict[str, Any] = {}  # task_id -> direction vector

    def store_task(self, task_id: str, theta_star: Union[np.ndarray, Any]):
        """
        Called once when a task is successfully solved.
        Calculates and stores the displacement vector v = theta_star - theta_init.
        """
        if self.is_torch:
            if not isinstance(theta_star, torch.Tensor):
                 theta_star = torch.tensor(theta_star, device=self.theta_init.device)
            v = theta_star - self.theta_init
            self.task_directions[task_id] = v.clone().detach()
        else:
            v = np.array(theta_star) - self.theta_init
            self.task_directions[task_id] = v.copy()
            
        logger.info(f"💾 Memory Stored: Task '{task_id}' vector captured.")

    def recover(self, theta_current: Union[np.ndarray, Any], task_id: str, alpha: float = 0.1, steps: int = 1, from_origin: bool = False) -> Union[np.ndarray, Any]:
        """
        On-demand recovery.
        
        Args:
            theta_current: The current (possibly damaged/forgotten) parameters.
            task_id: The ID of the task to recover.
            alpha: Step size.
            steps: Number of steps.
            from_origin: If True, applies vector to theta_init (Regrowth) instead of theta_current (Correction).
                         Use True for handling Massive Trauma (weights wiped).
                         Use False for handling Drift (solving new tasks).
        """
        if task_id not in self.task_directions:
            logger.warning(f"⚠️ Memory Miss: No solution found for task '{task_id}'")
            return theta_current

        v = self.task_directions[task_id]
        
        # Ensure consistency
        if self.is_torch:
            target_device = theta_current.device if isinstance(theta_current, torch.Tensor) else v.device
            
            if from_origin:
                theta = self.theta_init.clone().to(target_device)
            else:
                theta = theta_current.clone() if isinstance(theta_current, torch.Tensor) else torch.tensor(theta_current)
            
            if theta.device != v.device:
                v = v.to(theta.device)
        else:
            if from_origin:
                theta = np.array(self.theta_init, copy=True)
            else:
                theta = np.array(theta_current, copy=True)

        # Apply displacement
        total_shift = alpha * steps
        theta = theta + total_shift * v
        
        if from_origin:
            logger.info(f"🧠 Memory Recall: REGROWTH from Origin for '{task_id}' (Strength: {total_shift:.2f})")
        else:
            logger.info(f"🧠 Memory Recall: Correction applied for '{task_id}' (Strength: {total_shift:.2f})")
            
        return theta
