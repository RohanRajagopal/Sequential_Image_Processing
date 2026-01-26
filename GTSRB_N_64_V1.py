import jax
import jax.numpy as jnp
import equinox as eqx
import equinox.nn as enn
import diffrax as dfx
import optax
import time
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from functools import partial
import os
import shutil

# --- CRITICAL: Enable 64-bit precision ---
jax.config.update("jax_enable_x64", True)

# ==========================================
# 0. Global Settings & Device Setup (UPDATED)
# ==========================================

DATASET_NAME = "GTSRB"

if DATASET_NAME == "GTSRB":
    # Approximate Mean/Std for GTSRB
    DATA_MEAN = (0.3337, 0.3064, 0.3171)
    DATA_STD  = (0.2672, 0.2564, 0.2629)
    # GTSRB has 43 classes.
    CLASS_NAMES = [f"Sign {i}" for i in range(43)]

# <--- FEATURE: Multi-Device Detection --->
devices = jax.local_devices()
num_devices = len(devices)
print(f"🚀 Running {DATASET_NAME} on {num_devices} Device(s): {devices}")

class ModelConfig:
    # Architecture
    LAYER_SIZES = [64] # Increased capacity for RGB data
    
    # <--- CHANGED: Input is 28 (width) * 3 (RGB) = 84 --->
    INPUT_DIM = 84      
    ENCODER_DIM = 64    
    DECODER_DIM = 64    
    
    # <--- CHANGED: GTSRB has 43 classes --->
    OUTPUT_DIM = 43     

    # Physics / Time
    DT0 = 1e-8          
    MAX_STEPS = 100000   
    T_START = 0.0
    T_END = 5e-6        
    SEQ_LEN = 28  # We resize images to 28x28 (Height becomes Time)
    
    # Training
    BATCH_SIZE = 64     
    LR = 1.5e-3           
    NUM_EPOCHS = 60 
    
    # <--- FEATURE: Checkpoint Paths --->
    LATEST_CKPT = f"{DATASET_NAME.lower()}_latest.eqx"
    BEST_CKPT   = f"{DATASET_NAME.lower()}_best.eqx"

# Ensure batch size divides evenly across devices for pmap
if ModelConfig.BATCH_SIZE % num_devices != 0:
    raise ValueError(f"Batch size {ModelConfig.BATCH_SIZE} must be divisible by {num_devices} devices.")

# ==========================================
# 1. Utilities (Sharding & Checkpointing)
# ==========================================

def shard(x):
    """Split data into chunks for each device."""
    # (Batch, ...) -> (Num_Devices, Batch_per_Device, ...)
    return x.reshape((num_devices, -1) + x.shape[1:])

def save_checkpoint(path, model, opt_state, epoch, best_val_acc):
    """Saves model state to disk."""
    eqx.tree_serialise_leaves(path, (model, opt_state, epoch, best_val_acc))
    print(f"  💾 Saved checkpoint: {path}")

def load_checkpoint(path, model_skeleton, opt_skeleton):
    """Loads state from disk if available."""
    if not os.path.exists(path):
        print("  ⚠️ No checkpoint found. Starting fresh.")
        return model_skeleton, opt_skeleton, 0, 0.0
    
    try:
        restored = eqx.tree_deserialise_leaves(path, (model_skeleton, opt_skeleton, 0, 0.0))
        print(f"  🔄 Resumed from Epoch {restored[2]}")
        return restored
    except Exception as e:
        print(f"  ❌ Load failed ({e}). Starting fresh.")
        return model_skeleton, opt_skeleton, 0, 0.0

# ==========================================
# 2. Data Loading (UPDATED)
# ==========================================

def get_dataloaders():
    transform = transforms.Compose([
        transforms.Resize((28, 28)), # Force fixed size
        transforms.ToTensor(),
        transforms.Normalize(DATA_MEAN, DATA_STD)
    ])

    print("📥 Downloading/Loading GTSRB (this may take a moment)...")
    
    # GTSRB uses 'split' argument ('train' or 'test')
    train_set = torchvision.datasets.GTSRB(root='./data', split='train', download=True, transform=transform)
    test_set  = torchvision.datasets.GTSRB(root='./data', split='test',  download=True, transform=transform)

    def collate_fn(batch):
        # Batch item: (3, 28, 28) RGB Tensor
        # We need to treat Height (28) as Time.
        # Input at every time step is Width (28) * Channels (3) = 84.
        
        # 1. Permute to (H, W, C) -> (28, 28, 3)
        # 2. Reshape to (H, W*C)  -> (28, 84)
        processed_imgs = []
        labels = []
        for item in batch:
            img = item[0] # (3, 28, 28)
            label = item[1]
            
            # Permute C,H,W -> H,W,C
            img = img.permute(1, 2, 0).numpy()
            
            # Flatten Width and Channel: (28, 28, 3) -> (28, 84)
            img_flat = img.reshape(28, -1)
            
            processed_imgs.append(img_flat)
            labels.append(label)

        return np.stack(processed_imgs), np.array(labels)

    train_loader = DataLoader(train_set, batch_size=ModelConfig.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=ModelConfig.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, drop_last=True)
    
    return train_loader, test_loader

# ==========================================
# 3. Physics Model (High Precision)
# ==========================================

class CircuitOscillatorLayer(eqx.Module):
    V: enn.Linear 
    params: dict = eqx.field(static=True)
    num_nodes: int = eqx.field(static=True)

    def __init__(self, num_nodes, input_dim, key=None):
        self.num_nodes = num_nodes
        self.V = enn.Linear(input_dim, num_nodes, use_bias=False, key=key)
        
        R_distribution = np.linspace(1e5, 4e5, num_nodes)
        self.params = {
            "V_mid": 1.65, "I_0": 150e-9, "I_cc": 1.304e-12,
            "beta_cc": 18.9089, "C_s": 50e-15, "C_d": 150e-15,
            "C_eff": 250e-15, "R_val": R_distribution
        }

    def __call__(self, t, flat_state, u_t):
        state = flat_state.reshape(3, self.num_nodes)
        u1, u2, u3 = state[0], state[1], state[2]
        p = self.params

        K = p["beta_cc"] * (u1 * p["R_val"] + u2 / 2.0)
        J = p["I_cc"] * jnp.exp(p["beta_cc"] * (p["V_mid"] - u3 / 2.0))
        forcing = p["I_0"] * jax.nn.tanh((u_t @ self.V.weight.T)/0.11)

        du1 = (J * jnp.sinh(K) - u1) / (p["R_val"] * p["C_d"])
        du2 = -(J * jnp.sinh(K) + forcing) / p["C_eff"] 
        du3 = (2.0 / p["C_s"]) * (J * jnp.cosh(K) - 2.0 * p["I_0"])

        return jnp.stack([du1, du2, du3]).flatten()

class CircuitOscillatorFlow(eqx.Module):
    layers: list
    control: dfx.AbstractPath

    def __call__(self, t, flat_state, args=None):
        u_t = self.control.evaluate(t)
        layer_states, curr_idx, d_states = [], 0, []
        
        for i, layer in enumerate(self.layers):
            dim = 3 * layer.num_nodes 
            s_i = flat_state[curr_idx : curr_idx + dim]
            curr_idx += dim
            layer_states.append(s_i)

            inp = u_t if i == 0 else layer_states[i-1].reshape(3, -1)[1]
            d_states.append(layer(t, s_i, inp))

        return jnp.concatenate(d_states, axis=-1)

class DeepCircuitGraph(eqx.Module):
    encoder: enn.Linear
    decoder: enn.Linear
    readout: enn.Linear
    layers: list
    u3_star: float 

    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.encoder = enn.Linear(ModelConfig.INPUT_DIM, ModelConfig.ENCODER_DIM, key=keys[0])
        self.decoder = enn.Linear(ModelConfig.LAYER_SIZES[-1], ModelConfig.DECODER_DIM, key=keys[1])
        self.readout = enn.Linear(ModelConfig.DECODER_DIM, ModelConfig.OUTPUT_DIM, key=keys[2])

        self.layers = []
        prev_dim = ModelConfig.ENCODER_DIM 
        for size in ModelConfig.LAYER_SIZES:
            self.layers.append(CircuitOscillatorLayer(size, prev_dim, key=keys[3]))
            prev_dim = size

        p = self.layers[0].params
        self.u3_star = 2 * p["V_mid"] - (2 / p["beta_cc"]) * jnp.log(2 * p["I_0"] / p["I_cc"])

    def get_trajectory(self, u_seq, t_points):
        encoded_u_seq = jax.vmap(lambda u: jax.nn.tanh(self.encoder(u)))(u_seq)
        control = dfx.LinearInterpolation(ts=t_points, ys=encoded_u_seq)

        y0 = jnp.concatenate([
            jnp.concatenate([jnp.zeros(l.num_nodes), jnp.zeros(l.num_nodes), jnp.ones(l.num_nodes) * self.u3_star])
            for l in self.layers
        ])

        rhs = CircuitOscillatorFlow(self.layers, control)
        
        # <--- FEATURE: High Precision Solver --->
        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs), dfx.Tsit5(), 
            t0=t_points[0], t1=t_points[-1], dt0=ModelConfig.DT0, 
            y0=y0, 
            stepsize_controller=dfx.PIDController(rtol=1e-8, atol=1e-9), # Strict tolerances
            max_steps=ModelConfig.MAX_STEPS,
            saveat=dfx.SaveAt(ts=t_points) 
        )
        return sol.ys

    def __call__(self, u_seq, t_points):
        trajectory = self.get_trajectory(u_seq, t_points)
        
        # <--- FEATURE: Energy Penalty Calculation --->
        energy = jnp.mean(trajectory ** 2)

        last_nodes = self.layers[-1].num_nodes
        u2_seq = 10.0 * trajectory[:, -3*last_nodes:].reshape(len(t_points), 3, last_nodes)[:, 1, :]
        
        seq_logits = jax.vmap(lambda x: self.readout(jax.nn.relu(self.decoder(x))))(u2_seq)
        
        weights = jnp.linspace(0.1, 1.0, len(t_points))[:, None]
        final_logits = jnp.sum(seq_logits * weights, axis=0) / jnp.sum(weights)
        
        return final_logits, energy

# ==========================================
# 4. Training Loop (Parallel + Checkpoint)
# ==========================================

def train_and_evaluate():
    key = jax.random.PRNGKey(42)
    train_loader, test_loader = get_dataloaders()
    
    # 1. Init Skeleton
    model_skeleton = DeepCircuitGraph(key)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=ModelConfig.LR)
    )
    opt_skeleton = optimizer.init(eqx.filter(model_skeleton, eqx.is_array))
    
    # 2. Load Checkpoint (Resume logic)
    model, opt_state, start_epoch, best_val_acc = load_checkpoint(
        ModelConfig.LATEST_CKPT, model_skeleton, opt_skeleton
    )

    # 3. Replicate for Parallelism
    print(f"🔄 Replicating model to {num_devices} devices...")
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    t_points = jnp.linspace(ModelConfig.T_START, ModelConfig.T_END, ModelConfig.SEQ_LEN)

    # 4. Define Parallel Step
    def compute_loss(model, x, y):
        # x: (Batch, Rows, Cols)
        logits, energy = jax.vmap(model, in_axes=(0, None))(x, t_points)
        
        # <--- FEATURE: Cross Entropy + Regularization --->
        cls_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
        return cls_loss + 1e-4 * jnp.mean(energy)

    @partial(jax.pmap, axis_name='batch')
    def step(model, opt_state, x, y):
        loss, grads = eqx.filter_value_and_grad(compute_loss)(model, x, y)
        
        # Aggregate gradients across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        loss = jax.lax.pmean(loss, axis_name='batch')
        
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    # 5. Training Loop
    print(f"--- Starting Training ({DATASET_NAME}) ---")
    
    for epoch in range(start_epoch, ModelConfig.NUM_EPOCHS):
        total_loss = 0
        count = 0
        start_t = time.time()
        
        for batch_x, batch_y in train_loader:
            # Shard data: (64, ...) -> (Num_Devices, Batch_Per_Device, ...)
            x_sharded = shard(np.array(batch_x))
            y_sharded = shard(np.array(batch_y))
            
            model, opt_state, loss_val = step(model, opt_state, x_sharded, y_sharded)
            
            # loss_val is replicated, just take the first one
            total_loss += loss_val[0].item() 
            count += 1
            print(f"  Epoch {epoch+1} | Batch {count}/{len(train_loader)} | Loss: {loss_val[0].item():.4f}", end='\r')
        
        # 6. Validation & Save
        # Use first replica for validation
        model_single = jax.tree.map(lambda x: x[0], model)
        
        correct = 0
        total = 0
        for i, (vx, vy) in enumerate(test_loader):
            if i > 20: break # Quick check
            vx = jnp.array(vx)
            vy = jnp.array(vy)
            logits, _ = jax.vmap(model_single, in_axes=(0, None))(vx, t_points)
            preds = jnp.argmax(logits, axis=1)
            correct += jnp.sum(preds == vy)
            total += len(vy)
            
        val_acc = correct / total
        avg_loss = total_loss / count
        
        print(f"\nEpoch {epoch+1:02d} | Time: {time.time()-start_t:.0f}s | Avg Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2%}")
        
        # Save Checkpoint
        opt_single = jax.tree.map(lambda x: x[0], opt_state)
        save_checkpoint(ModelConfig.LATEST_CKPT, model_single, opt_single, epoch + 1, max(val_acc, best_val_acc))
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            shutil.copyfile(ModelConfig.LATEST_CKPT, ModelConfig.BEST_CKPT)

    # Return un-replicated model for visualization
    return jax.tree.map(lambda x: x[0], model), test_loader

# ==========================================
# 5. Advanced Visualization (UPDATED)
# ==========================================

def visualize_dynamics(model, loader):
    print("\n🎨 Generating Dashboard...")
    bx, by = next(iter(loader))
    
    idx = np.random.randint(0, bx.shape[0])
    # bx is (Batch, 28, 84)
    img_flat_sample = jnp.array(bx[idx]) 
    label = by[idx]
    
    t_points = jnp.linspace(ModelConfig.T_START, ModelConfig.T_END, ModelConfig.SEQ_LEN)
    
    # Run Inference
    enc_out = jax.vmap(lambda u: jax.nn.tanh(model.encoder(u)))(img_flat_sample)
    trajectory = model.get_trajectory(img_flat_sample, t_points)
    
    last_nodes = model.layers[-1].num_nodes
    u2_trace = 1 * trajectory[:, -3*last_nodes:].reshape(len(t_points), 3, last_nodes)[:, 1, :]
    
    # Plotting
    fig = plt.figure(figsize=(15, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)

    # 1. Image Reconstruction
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Reshape back: (28, 84) -> (28, 28, 3)
    disp_img = np.array(img_flat_sample).reshape(28, 28, 3)
    
    # Un-normalize for display
    mean = np.array(DATA_MEAN).reshape(1, 1, 3)
    std = np.array(DATA_STD).reshape(1, 1, 3)
    disp_img = disp_img * std + mean
    disp_img = np.clip(disp_img, 0, 1)
    
    ax1.imshow(disp_img)
    ax1.set_title(f"Input: {CLASS_NAMES[label]}")
    ax1.axis('off')

    # 2. Control Signal
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(enc_out.T, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_title("Encoder Control Signal")
    ax2.set_xlabel("Time (Image Row)")
    ax2.set_ylabel("Feature")
    plt.colorbar(im2, ax=ax2)

    # 3. Traces
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t_points, u2_trace[:, ::5]) # Show every 5th node
    ax3.set_title("Voltage Traces (Sampled)")
    ax3.set_xlabel("Time (s)")

    # 4. Dynamics Heatmap
    ax4 = fig.add_subplot(gs[1, :])
    im4 = ax4.imshow(u2_trace.T, aspect='auto', cmap='inferno', interpolation='nearest')
    ax4.set_title("Full Circuit Dynamics (Nodes vs Time)")
    ax4.set_ylabel("Node Index")
    ax4.set_xlabel("Time Step")
    plt.colorbar(im4, ax=ax4)

    plt.suptitle(f"Physics-Informed Neural Circuit ({DATASET_NAME})", fontsize=16)
    plt.show()

if __name__ == "__main__":
    final_model, loader = train_and_evaluate()
    visualize_dynamics(final_model, loader)
