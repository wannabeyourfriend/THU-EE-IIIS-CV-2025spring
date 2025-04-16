import os
import subprocess
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
import glob

# Define learning rates to test
learning_rates = [1e-3, 1e-4, 1e-5]
exp_names = [f"cifar10_lr_{lr}" for lr in learning_rates]

# Train models with different learning rates
for lr, exp_name in zip(learning_rates, exp_names):
    print(f"Training with learning rate: {lr}")
    subprocess.run([
        "python", "train.py", 
        "--exp_name", exp_name,
        "--lr", str(lr),
        "--total_epoch", "10",
        "--batchsize", "64"
    ])

# Function to extract data from TensorBoard logs
def extract_tensorboard_data(log_dir):
    event_files = glob.glob(os.path.join(log_dir, "events.out.tfevents.*"))
    if not event_files:
        print(f"No event files found in {log_dir}")
        return None, None, None, None
    
    ea = event_accumulator.EventAccumulator(event_files[0])
    ea.Reload()
    
    train_loss = [(s.step, s.value) for s in ea.Scalars('Train/Loss')]
    train_acc = [(s.step, s.value) for s in ea.Scalars('Train/Acc@1')]
    val_acc = [(s.step, s.value) for s in ea.Scalars('Validation/Acc@1')]
    
    return train_loss, train_acc, val_acc

# Create plots
plt.figure(figsize=(15, 10))

# Training Loss
plt.subplot(2, 2, 1)
for lr, exp_name in zip(learning_rates, exp_names):
    log_dir = f"../experiments/{exp_name}/log"
    train_loss, _, _ = extract_tensorboard_data(log_dir)
    if train_loss:
        steps, values = zip(*train_loss)
        plt.plot(steps, values, label=f"lr={lr}")
plt.title('Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()

# Training Accuracy
plt.subplot(2, 2, 2)
for lr, exp_name in zip(learning_rates, exp_names):
    log_dir = f"../experiments/{exp_name}/log"
    _, train_acc, _ = extract_tensorboard_data(log_dir)
    if train_acc:
        steps, values = zip(*train_acc)
        plt.plot(steps, values, label=f"lr={lr}")
plt.title('Training Accuracy')
plt.xlabel('Iterations')
plt.ylabel('Accuracy')
plt.legend()

# Validation Accuracy
plt.subplot(2, 2, 3)
for lr, exp_name in zip(learning_rates, exp_names):
    log_dir = f"../experiments/{exp_name}/log"
    _, _, val_acc = extract_tensorboard_data(log_dir)
    if val_acc:
        epochs = range(len(val_acc))
        _, values = zip(*val_acc)
        plt.plot(epochs, values, label=f"lr={lr}")
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('../results/learning_rate_comparison.png')
plt.show()