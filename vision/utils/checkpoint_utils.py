import orbax.checkpoint
from pathlib import Path

def save_checkpoint(train_state, step, checkpoint_dir):
    """Save training state checkpoint"""
    checkpoint_dir = Path(checkpoint_dir).resolve()  # Convert to absolute path
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_path = checkpoint_dir / f'step_{step}'
    checkpointer.save(save_path, train_state, force=True)  # Overwrite existing
    print(f"Saved checkpoint at step {step}")

def load_checkpoint(checkpoint_path, train_state_template):
    """Load training state checkpoint"""
    checkpoint_path = Path(checkpoint_path).resolve()  # Convert to absolute path
    checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored_state = checkpointer.restore(checkpoint_path, item=train_state_template, partial_restore=True)
    print(f"Loaded checkpoint from step {restored_state.step}")
    return restored_state