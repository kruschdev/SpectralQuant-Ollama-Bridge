import os
import sys
import torch

# Add backend directory to path
PROJECT_ROOT = "/home/kruschdev/homelab/projects/spectralquant-ollama-bridge/backend"
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "spectralquant-src", "src"))

try:
    from spectralquant.engine import SpectralQuantEngine
    from spectralquant_cache import SpectralQuantCache, SpectralQuantEngineManager
    print("✅ Successfully imported SpectralQuant engine and cache wrapper!")
except ImportError as e:
    print(f"❌ Failed to import engine/cache: {e}")
    sys.exit(1)

# Set seed for reproducibility
torch.manual_seed(42)

head_dim = 128
d_eff = 64
total_bits = 3

# Generate dummy eigenvectors and eigenvalues
eigenvalues = torch.sort(torch.rand(head_dim), descending=True)[0]
raw_matrix = torch.randn(head_dim, head_dim)
q, r = torch.linalg.qr(raw_matrix)
eigenvectors = q

# Mock calibration data dictionary for 1 layer, 2 heads
calibration_data = {
    0: {
        0: {
            "key_eigenvectors": eigenvectors.clone(),
            "key_eigenvalues": eigenvalues.clone(),
            "key_d_eff": d_eff,
            "val_eigenvectors": eigenvectors.clone(),
            "val_eigenvalues": eigenvalues.clone(),
            "val_d_eff": d_eff,
        },
        1: {
            "key_eigenvectors": eigenvectors.clone(),
            "key_eigenvalues": eigenvalues.clone(),
            "key_d_eff": d_eff,
            "val_eigenvectors": eigenvectors.clone(),
            "val_eigenvalues": eigenvalues.clone(),
            "val_d_eff": d_eff,
        }
    }
}

print("Initializing SpectralQuantEngineManager...")
manager = SpectralQuantEngineManager(calibration_data, avg_bits=total_bits, device="cpu")

print("\n=== Testing Batch Size = 1 ===")
cache_single = SpectralQuantCache(manager)

# Mock K/V tensors with shape [batch, heads, seq_len, head_dim]
batch_size = 1
num_heads = 2
seq_len = 8
K = torch.randn(batch_size, num_heads, seq_len, head_dim)
V = torch.randn(batch_size, num_heads, seq_len, head_dim)

print(f"Input shape: {K.shape}")
recon_k, recon_v = cache_single.update(K, V, layer_idx=0)
print(f"Reconstructed K shape: {recon_k.shape}")
print(f"Reconstructed V shape: {recon_v.shape}")

assert recon_k.shape == K.shape, f"Shape mismatch: {recon_k.shape} vs {K.shape}"
assert recon_v.shape == V.shape, f"Shape mismatch: {recon_v.shape} vs {V.shape}"
print("✅ Batch size 1 shape match verified.")

# Test appending new token step (autoregressive)
K_new = torch.randn(batch_size, num_heads, 1, head_dim)
V_new = torch.randn(batch_size, num_heads, 1, head_dim)
recon_k_next, recon_v_next = cache_single.update(K_new, V_new, layer_idx=0)
print(f"Appended step: new history shape is {recon_k_next.shape}")
assert recon_k_next.shape == (batch_size, num_heads, seq_len + 1, head_dim)
print("✅ Autoregressive concatenation shape verified.")


print("\n=== Testing Batch Size = 3 (Multi-Batch concurrency test) ===")
cache_multi = SpectralQuantCache(manager)

batch_size_m = 3
K_m = torch.randn(batch_size_m, num_heads, seq_len, head_dim)
V_m = torch.randn(batch_size_m, num_heads, seq_len, head_dim)

print(f"Multi-batch input shape: {K_m.shape}")
recon_k_m, recon_v_m = cache_multi.update(K_m, V_m, layer_idx=0)
print(f"Reconstructed multi K shape: {recon_k_m.shape}")
print(f"Reconstructed multi V shape: {recon_v_m.shape}")

assert recon_k_m.shape == K_m.shape, f"Shape mismatch: {recon_k_m.shape} vs {K_m.shape}"
assert recon_v_m.shape == V_m.shape, f"Shape mismatch: {recon_v_m.shape} vs {V_m.shape}"
print("✅ Batch size 3 shape match verified.")

# Test appending new step on batch=3
K_m_new = torch.randn(batch_size_m, num_heads, 1, head_dim)
V_m_new = torch.randn(batch_size_m, num_heads, 1, head_dim)
recon_k_m_next, recon_v_m_next = cache_multi.update(K_m_new, V_m_new, layer_idx=0)
print(f"Appended step on batch=3: new history shape is {recon_k_m_next.shape}")
assert recon_k_m_next.shape == (batch_size_m, num_heads, seq_len + 1, head_dim)
print("✅ Multi-batch autoregressive concatenation verified.")

print(f"Checking cache sequence length: {cache_multi.get_seq_length(0)}")
assert cache_multi.get_seq_length(0) == seq_len + 1, "Sequence length tracking incorrect!"

print("\n🎉 ALL MULTI-BATCH CACHE TESTS PASSED SUCCESSFULLY! PARITY SECURED.")
sys.exit(0)
