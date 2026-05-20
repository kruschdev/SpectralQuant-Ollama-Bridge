import torch
from transformers.cache_utils import DynamicCache
from typing import Any, Dict, List, Optional, Tuple

class SpectralQuantEngineManager:
    """Manager that holds SpectralQuantEngine instances for each layer, head, and modality."""
    def __init__(self, calibration_data: dict, avg_bits: float = 3.0, device: str = "cuda"):
        # Import engine lazily so it works when spectralquant-src is added to sys.path
        import sys
        from pathlib import Path
        PROJECT_ROOT = Path(__file__).parent
        if str(PROJECT_ROOT / "spectralquant-src" / "src") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "spectralquant-src" / "src"))
            
        # Try to import from integration test directory or core engine
        try:
            from experiments.phase2_integration import SpectralQuantEngine
        except ImportError:
            # Fallback if we need to use the canonical engine (assuming it matches)
            # but usually it's in spectralquant.engine?
            try:
                from spectralquant.engine import SpectralQuantEngine
            except ImportError:
                raise ImportError("Could not import SpectralQuantEngine")
        
        self.key_engines = {}
        self.val_engines = {}
        
        for l in calibration_data.keys():
            n_kv_heads = len(calibration_data[l])
            for h in range(n_kv_heads):
                c = calibration_data[l][h]
                
                key_eng = SpectralQuantEngine(
                    eigenvectors=c["key_eigenvectors"].to(device),
                    eigenvalues=c["key_eigenvalues"].to(device),
                    d_eff=int(c["key_d_eff"]),
                    head_dim=c["key_eigenvectors"].shape[0],
                    total_bits=int(avg_bits),
                    device=device
                )
                if hasattr(torch, "compile"):
                    try:
                        key_eng._quantize_regime = torch.compile(
                            key_eng._quantize_regime,
                            mode="reduce-overhead",
                            dynamic=True
                        )
                        key_eng.decompress_keys_pytorch = torch.compile(
                            key_eng.decompress_keys_pytorch,
                            mode="reduce-overhead",
                            dynamic=True
                        )
                    except Exception as compile_err:
                        import logging
                        logging.getLogger("spectralquant").warning(f"Could not compile key engine: {compile_err}")
                self.key_engines[(l, h)] = key_eng
                
                val_eng = SpectralQuantEngine(
                    eigenvectors=c["val_eigenvectors"].to(device),
                    eigenvalues=c["val_eigenvalues"].to(device),
                    d_eff=int(c["val_d_eff"]),
                    head_dim=c["val_eigenvectors"].shape[0],
                    total_bits=int(avg_bits),
                    device=device
                )
                if hasattr(torch, "compile"):
                    try:
                        val_eng._quantize_regime = torch.compile(
                            val_eng._quantize_regime,
                            mode="reduce-overhead",
                            dynamic=True
                        )
                        val_eng.decompress_values_pytorch = torch.compile(
                            val_eng.decompress_values_pytorch,
                            mode="reduce-overhead",
                            dynamic=True
                        )
                    except Exception as compile_err:
                        import logging
                        logging.getLogger("spectralquant").warning(f"Could not compile value engine: {compile_err}")
                self.val_engines[(l, h)] = val_eng
    
    def get_engine(self, layer_idx: int, head_idx: int, modality: str):
        if modality == "key":
            return self.key_engines.get((layer_idx, head_idx))
        else:
            return self.val_engines.get((layer_idx, head_idx))


class SpectralQuantCache(DynamicCache):
    """
    A custom Cache subclass that integrates SpectralQuant KV compression.
    Instead of storing uncompressed tensors, it compresses keys and values
    into dictionaries and decompresses them during the update() call.
    """
    def __init__(self, engine_manager: SpectralQuantEngineManager, config: Optional[Any] = None):
        super().__init__(config=config)
        self.engine_manager = engine_manager
        
        num_layers = len(self.layers) if hasattr(self, "layers") else 0
        self.sq_key_cache: List[Any] = [[] for _ in range(num_layers)]
        self.sq_value_cache: List[Any] = [[] for _ in range(num_layers)]
        self._seen_tokens = 0
        
    def _compress_states(self, states: torch.Tensor, layer_idx: int, modality: str) -> List[List[Dict[str, torch.Tensor]]]:
        """
        states: [batch_size, num_heads, seq_len, head_dim]
        Returns: list of list of compressed dicts, shape [batch_size, num_heads]
        """
        batch_size, num_heads, seq_len, head_dim = states.shape
        compressed_batches = []
        for b in range(batch_size):
            compressed_heads = []
            for h in range(num_heads):
                state_h = states[b, h, :, :]  # [seq_len, head_dim]
                engine = self.engine_manager.get_engine(layer_idx, h, modality)
                if engine is None:
                    raise ValueError(f"No engine found for layer {layer_idx}, head {h}, modality {modality}")
                    
                # Move engine tensors to the same device as the layer state
                if getattr(engine, '_current_device', None) != state_h.device:
                    engine.eigenvalues = engine.eigenvalues.to(state_h.device)
                    engine.Pi = engine.Pi.to(state_h.device)
                    engine.PiT = engine.PiT.to(state_h.device)
                    engine.S = engine.S.to(state_h.device)
                    engine.ST = engine.ST.to(state_h.device)
                    if hasattr(engine, '_centroids_key_high'):
                        engine._centroids_key_high = engine._centroids_key_high.to(state_h.device)
                        engine._centroids_key_low = engine._centroids_key_low.to(state_h.device)
                        engine._centroids_val_high = engine._centroids_val_high.to(state_h.device)
                        engine._centroids_val_low = engine._centroids_val_low.to(state_h.device)
                    engine._current_device = state_h.device
                
                if modality == "key":
                    comp = engine.compress_keys_pytorch(state_h)
                    # k_mse is omitted from the compressed dict to save massive VRAM
                else:
                    comp = engine.compress_values_pytorch(state_h)
                
                compressed_heads.append(comp)
            compressed_batches.append(compressed_heads)
        return compressed_batches

    def _decompress_states(self, compressed_batches: List[List[Dict[str, torch.Tensor]]], layer_idx: int, modality: str) -> torch.Tensor:
        """
        Reconstructs the full uncompressed state tensor from compressed head dicts.
        Returns: [batch_size, num_heads, seq_len, head_dim]
        """
        uncompressed_batches = []
        for b, compressed_heads in enumerate(compressed_batches):
            uncompressed_heads = []
            for h, comp in enumerate(compressed_heads):
                engine = self.engine_manager.get_engine(layer_idx, h, modality)
                
                # Move engine tensors to the correct device first
                state_device = comp["indices"].device
                if getattr(engine, '_current_device', None) != state_device:
                    engine.eigenvalues = engine.eigenvalues.to(state_device)
                    engine.Pi = engine.Pi.to(state_device)
                    engine.PiT = engine.PiT.to(state_device)
                    engine.S = engine.S.to(state_device)
                    engine.ST = engine.ST.to(state_device)
                    if hasattr(engine, '_centroids_key_high'):
                        engine._centroids_key_high = engine._centroids_key_high.to(state_device)
                        engine._centroids_key_low = engine._centroids_key_low.to(state_device)
                    if hasattr(engine, '_centroids_val_high'):
                        engine._centroids_val_high = engine._centroids_val_high.to(state_device)
                        engine._centroids_val_low = engine._centroids_val_low.to(state_device)
                    engine._current_device = state_device

                if modality == "key":
                    recon_h = engine.decompress_keys_pytorch(comp)
                else:
                    recon_h = engine.decompress_values_pytorch(comp)
                
                uncompressed_heads.append(recon_h)
            uncompressed_batches.append(torch.stack(uncompressed_heads, dim=0))
        return torch.stack(uncompressed_batches, dim=0)  # [batch_size, num_heads, seq_len, head_dim]

    def _concat_compressed(self, existing: List[List[Dict[str, torch.Tensor]]], new: List[List[Dict[str, torch.Tensor]]]) -> List[List[Dict[str, torch.Tensor]]]:
        """
        Concatenates new compressed representations onto existing ones along the seq_len dimension.
        """
        result = []
        for e_batch, n_batch in zip(existing, new):
            batch_result = []
            for e, n in zip(e_batch, n_batch):
                merged = {}
                for k in e.keys():
                    if isinstance(e[k], torch.Tensor) and k in n:
                        # Concatenate along seq_len which is dim 0 since we removed batch_size
                        merged[k] = torch.cat([e[k], n[k]], dim=0)
                    else:
                        merged[k] = e[k]
                batch_result.append(merged)
            result.append(batch_result)
        return result

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Ensure lists are large enough
        if len(self.sq_key_cache) <= layer_idx:
            self.sq_key_cache.extend([[] for _ in range(layer_idx + 1 - len(self.sq_key_cache))])
            self.sq_value_cache.extend([[] for _ in range(layer_idx + 1 - len(self.sq_value_cache))])

        # Check if this layer has a compression engine (e.g. key engine for head 0 exists)
        has_engine = self.engine_manager.get_engine(layer_idx, 0, "key") is not None
        
        if not has_engine:
            # Delegate directly to parent DynamicCache/Cache class so that
            # uncalibrated or linear layers handle their state updates correctly.
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        # Compress new states
        new_comp_keys = self._compress_states(key_states, layer_idx, modality="key")
        new_comp_vals = self._compress_states(value_states, layer_idx, modality="value")
        
        if len(self.sq_key_cache[layer_idx]) == 0 or isinstance(self.sq_key_cache[layer_idx], torch.Tensor):
            self.sq_key_cache[layer_idx] = new_comp_keys
            self.sq_value_cache[layer_idx] = new_comp_vals
        else:
            self.sq_key_cache[layer_idx] = self._concat_compressed(self.sq_key_cache[layer_idx], new_comp_keys)
            self.sq_value_cache[layer_idx] = self._concat_compressed(self.sq_value_cache[layer_idx], new_comp_vals)

        # Decompress full history to return to the model
        full_keys = self._decompress_states(self.sq_key_cache[layer_idx], layer_idx, modality="key")
        full_vals = self._decompress_states(self.sq_value_cache[layer_idx], layer_idx, modality="value")
        
        # Update seen tokens for the first layer
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]
            
        return full_keys, full_vals

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.sq_key_cache) <= layer_idx or len(self.sq_key_cache[layer_idx]) == 0:
            # Fallback to super/underlying layers
            if layer_idx < len(self.layers):
                layer = self.layers[layer_idx]
                if hasattr(layer, "get_seq_length"):
                    return layer.get_seq_length()
            return 0
        cache_item = self.sq_key_cache[layer_idx]
        if isinstance(cache_item, torch.Tensor):
            return cache_item.shape[-2]
        if isinstance(cache_item, list) and len(cache_item) > 0:
            # The seq_len is dim=0 of the "indices" tensor of the first batch and head
            return cache_item[0][0]["indices"].shape[0]
        return 0

    def get_max_length(self) -> Optional[int]:
        return None
        
    def to(self, device):
        # Implementation of moving to device if needed
        # Just return self as DynamicCache's to is often not strictly implemented for full depth dict structures
        return self

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self.layers):
            has_engine = self.engine_manager.get_engine(layer_idx, 0, "key") is not None
            if has_engine:
                k_cache = self.sq_key_cache[layer_idx]
                v_cache = self.sq_value_cache[layer_idx]
                k_decomp = k_cache if isinstance(k_cache, torch.Tensor) else self._decompress_states(k_cache, layer_idx, "key")
                v_decomp = v_cache if isinstance(v_cache, torch.Tensor) else self._decompress_states(v_cache, layer_idx, "value")
                return k_decomp, v_decomp
            else:
                layer = self.layers[layer_idx]
                keys = getattr(layer, "keys", None)
                values = getattr(layer, "values", None)
                return keys, values
        else:
            raise KeyError(f"Cache only has {len(self.layers)} layers")

    def __iter__(self):
        for layer_idx in range(len(self.layers)):
            yield self[layer_idx]

    def __len__(self):
        return len(self.layers)

    @property
    def key_cache(self) -> List[torch.Tensor]:
        res = []
        for i in range(len(self.layers)):
            res.append(self[i][0])
        return res

    @property
    def value_cache(self) -> List[torch.Tensor]:
        res = []
        for i in range(len(self.layers)):
            res.append(self[i][1])
        return res
