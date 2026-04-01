import hashlib
import json
from omegaconf import OmegaConf

def config_hash(cfg) -> str:
    d = OmegaConf.to_container(cfg, resolve=True)
    payload = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
