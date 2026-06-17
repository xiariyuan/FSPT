from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Optional

import torch
import torch.distributed as dist


def _project_root() -> Path:
    # utils/ is directly under the project root
    return Path(__file__).resolve().parents[1]


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        try:
            return int(dist.get_rank()) == 0
        except Exception:
            return False
    return True


class DiskTensorCache:
    """
    Minimal on-disk tensor cache with atomic writes.

    Intended use:
    - Cache expensive per-video features (e.g., CLIP global/spatial features) during evaluation.
    - Keep tensors on CPU on disk; caller moves to the desired device.
    """

    def __init__(
        self,
        root_dir: str,
        enabled: bool = True,
        write: bool = True,
        keep_in_memory: bool = True,
    ) -> None:
        self.enabled = bool(enabled)
        self.write = bool(write)
        self.keep_in_memory = bool(keep_in_memory)
        self._mem: dict[str, torch.Tensor] = {}

        root = Path(str(root_dir))
        if not root.is_absolute():
            root = (_project_root() / root).resolve()
        self.root_dir = root
        if self.enabled:
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for_key(self, key: str) -> Path:
        digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
        return self.root_dir / f"{digest}.pt"

    def get(self, key: str) -> Optional[torch.Tensor]:
        if not self.enabled:
            return None
        if self.keep_in_memory and key in self._mem:
            return self._mem[key]

        path = self._path_for_key(key)
        if not path.exists() or path.stat().st_size == 0:
            return None

        try:
            obj: Any = torch.load(str(path), map_location="cpu", weights_only=False)
            tensor = obj.get("tensor") if isinstance(obj, dict) else obj
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.detach().cpu()
                if self.keep_in_memory:
                    self._mem[key] = tensor
                return tensor
        except Exception:
            return None
        return None

    def set(self, key: str, tensor: torch.Tensor) -> None:
        if not self.enabled or not self.write or not _is_main_process():
            return
        if not isinstance(tensor, torch.Tensor):
            return

        path = self._path_for_key(key)
        if path.exists() and path.stat().st_size > 0:
            return

        try:
            payload = {"key": key, "tensor": tensor.detach().cpu()}
            tmp_path = path.with_suffix(".tmp")
            torch.save(payload, str(tmp_path))
            os.replace(str(tmp_path), str(path))
            if self.keep_in_memory:
                self._mem[key] = payload["tensor"]
        except Exception:
            try:
                tmp_path = path.with_suffix(".tmp")
                if tmp_path.exists():
                    tmp_path.unlink()
            except Exception:
                pass
