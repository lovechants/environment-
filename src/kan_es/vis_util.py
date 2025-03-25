import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import torch


def plot_training_loss(train_losses, val_losses, save_path=None):
    """
    Plot the training and validation losses
    
    Args:
        train_losses (list): List of training losses per epoch
        val_losses (list): List of validation losses per epoch
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(confusion_matrix, labels, save_path=None):
    """
    Plot the confusion matrix
    
    Args:
        confusion_matrix (dict): Dictionary with true labels as keys and dict of predicted labels as values
        labels (list): List of label names
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    matrix = np.zeros((len(labels), len(labels)))
    
    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            matrix[i, j] = confusion_matrix[true_label][pred_label]
    
    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.zeros_like(matrix)
    for i in range(matrix.shape[0]):
        if row_sums[i] > 0:
            normalized_matrix[i] = matrix[i] / row_sums[i]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_transition_matrix(sequences, labels, save_path=None):
    """
    Analyze and plot transitions between events in the sequences
    
    Args:
        sequences (dict): Dictionary of sequences
        labels (list): List of label names
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    transitions = defaultdict(lambda: defaultdict(int))
    
    for seq_id, seq in sequences.items():
        for i in range(len(seq) - 1):
            current = seq[i]
            next_event = seq[i + 1]
            transitions[current][next_event] += 1
    
    # Create transition matrix
    matrix = np.zeros((len(labels), len(labels)))
    
    for i, from_label in enumerate(labels):
        for j, to_label in enumerate(labels):
            matrix[i, j] = transitions[from_label][to_label]
    
    # Normalize by row
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    normalized_matrix = matrix / row_sums
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(normalized_matrix, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Next Event")
    plt.ylabel("Current Event")
    plt.title("Event Transition Probabilities")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
    
    return transitions


def plot_attention_weights(model, sequence, label_to_idx, idx_to_label, save_path=None):
    """
    Visualize the attention weights for a sequence
    
    Args:
        model: The trained KAN model
        sequence (list): Input event sequence
        label_to_idx (dict): Mapping from labels to indices
        idx_to_label (dict): Mapping from indices to labels
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    device = model.device
    
    # Convert sequence to indices
    seq_indices = [label_to_idx[label] for label in sequence]
    
    # Pad if necessary
    if len(seq_indices) < model.sequence_length:
        seq_indices = [0] * (model.sequence_length - len(seq_indices)) + seq_indices
    else:
        seq_indices = seq_indices[-model.sequence_length:]
        
    seq_tensor = torch.LongTensor(seq_indices).unsqueeze(0).to(device)
    
    # Get model output with attention weights
    try:
        logits, attention_weights = model(seq_tensor, return_attention=True)
        
        # Extract attention weights
        if isinstance(attention_weights, tuple):
            attention_weights = attention_weights[0]  # Take first element if it's a tuple
        
        # Process attention weights for visualization
        attn = attention_weights.squeeze().cpu().detach().numpy()
        
        # If multidimensional, average across one dimension for visualization
        if len(attn.shape) > 2:
            attn = np.mean(attn, axis=0)
        
        # Get the actual sequence labels for the x-axis
        if len(sequence) < model.sequence_length:
            x_labels = ["<pad>"] * (model.sequence_length - len(sequence)) + sequence
        else:
            x_labels = sequence[-model.sequence_length:]
        
        # Plot heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(attn, cmap="viridis", xticklabels=x_labels)
        plt.title("Attention Weights")
        plt.xlabel("Sequence Position")
        plt.ylabel("Attention Head/Feature")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    except Exception as e:
        print(f"Could not get attention weights: {e}")
        print("This model may not support returning attention weights.")


def plot_multiscale_weights(model, save_path=None):
    """
    Plot the learned weights for multi-scale models
    
    Args:
        model: The multi-scale KAN model
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    try:
        # Get the scale weights
        if hasattr(model, 'scale_weights'):
            weights = model.scale_weights.cpu().detach().numpy()
            weights = torch.softmax(torch.tensor(weights), dim=0).numpy()
            
            # Scale names
            scale_names = ['Local', 'Mid-range', 'Global']
            
            plt.figure(figsize=(10, 6))
            plt.bar(scale_names, weights)
            plt.ylim(0, 1.0)
            plt.ylabel('Weight')
            plt.title('Learned Weights for Different Temporal Scales')
            plt.grid(True, alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(weights):
                plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path)
                plt.close()
            else:
                plt.show()
        else:
            print("Model does not have scale_weights attribute")
    except Exception as e:
        print(f"Could not plot multi-scale weights: {e}")


def visualize_predictions(model, test_sequences, idx_to_label, label_to_idx, num_examples=5, save_path=None):
    """
    Visualize predictions for a few test sequences
    
    Args:
        model: The trained KAN model
        test_sequences (dict): Dictionary of test sequences
        idx_to_label (dict): Mapping from indices to labels
        label_to_idx (dict): Mapping from labels to indices
        num_examples (int): Number of examples to visualize
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    device = model.device
    model.eval()
    
    # Select some test sequences
    example_ids = list(test_sequences.keys())[:num_examples]
    
    plt.figure(figsize=(15, num_examples * 3))
    
    for i, seq_id in enumerate(example_ids):
        sequence = test_sequences[seq_id]
        
        # Take a prefix as input
        prefix_length = len(sequence) // 2
        prefix = sequence[:prefix_length]
        
        # Generate the rest of the sequence
        generated, probabilities = model.generate_sequence(
            prefix, len(sequence), idx_to_label, label_to_idx
        )
        
        # Plot the original and generated sequence side by side
        ax = plt.subplot(num_examples, 1, i + 1)
        
        # Mark prefix in both sequences
        for j in range(len(sequence)):
            if j < prefix_length:
                color = 'blue'  # Prefix (given)
                alpha = 0.7
            else:
                if j < len(generated):
                    # Check if prediction matches original
                    if sequence[j] == generated[j]:
                        color = 'green'  # Correct prediction
                    else:
                        color = 'red'  # Incorrect prediction
                    alpha = 1.0
                else:
                    color = 'gray'  # Original beyond generated length
                    alpha = 0.5
            
            # Plot original sequence
            plt.text(j, 0.7, sequence[j], 
                    horizontalalignment='center',
                    color=color, alpha=alpha,
                    fontsize=12)
            
            # Plot probabilities for generated sequence
            if j >= prefix_length and j < len(generated):
                prob = probabilities[j - prefix_length][1]  # Get the probability
                plt.text(j, 0.3, f"{prob:.2f}", 
                        horizontalalignment='center',
                        color=color, alpha=alpha,
                        fontsize=10)
        
        # Add labels
        plt.text(-0.5, 0.7, "Original:", horizontalalignment='right')
        plt.text(-0.5, 0.3, "Prob:", horizontalalignment='right')
        
        # Remove axes
        plt.axis('off')
        
        # Add title for this example
        plt.title(f"Sequence {seq_id}: Prefix (blue) + Actual vs Predicted (green=correct, red=wrong)")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_sequence_likelihood(model, sequences, idx_to_label, label_to_idx, save_path=None):
    """
    Plot the likelihood of different sequences
    
    Args:
        model: The trained KAN model
        sequences (dict): Dictionary of sequences
        idx_to_label (dict): Mapping from indices to labels
        label_to_idx (dict): Mapping from labels to indices
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    device = model.device
    model.eval()
    
    sequence_probs = []
    
    # Calculate log-likelihood for each sequence
    for seq_id, sequence in sequences.items():
        log_likelihood = 0
        for i in range(1, len(sequence)):
            # Predict next event based on prefix
            prefix = sequence[:i]
            _, prob, prob_dist = model.predict_next_event(
                prefix, idx_to_label, label_to_idx
            )
            
            # Get probability of actual next event
            actual_next = sequence[i]
            if actual_next in prob_dist:
                event_prob = prob_dist[actual_next]
                # Add log probability
                log_likelihood += np.log(max(event_prob, 1e-10))
        
        # Store sequence ID and average log likelihood
        sequence_probs.append((seq_id, log_likelihood / (len(sequence) - 1)))
    
    # Sort by likelihood
    sequence_probs.sort(key=lambda x: x[1], reverse=True)
    
    # Plot top and bottom 10 sequences
    num_to_plot = min(10, len(sequence_probs) // 2)
    plot_sequences = sequence_probs[:num_to_plot] + sequence_probs[-num_to_plot:]
    
    plt.figure(figsize=(12, 6))
    
    seq_ids = [str(seq_id) for seq_id, _ in plot_sequences]
    likelihoods = [likelihood for _, likelihood in plot_sequences]
    
    colors = ['green'] * num_to_plot + ['red'] * num_to_plot
    
    bars = plt.bar(seq_ids, likelihoods, color=colors)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.xlabel('Sequence ID')
    plt.ylabel('Average Log Likelihood')
    plt.title('Sequence Likelihoods (Green = Most Likely, Red = Least Likely)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
