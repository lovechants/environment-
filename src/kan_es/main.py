import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import os
import json
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.backends.mps

# Import the event generator utility
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
from utils.generating_topology_based_events import gen_synth_data_from_topology, gen_mult_seq_from_topology_example
# Import our KAN implementation
from model import EventSequenceKAN, MultiScaleEventSequenceKAN, EventSequenceDataset


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='KAN for Event Sequence Prediction')
    
    # Data parameters
    parser.add_argument('--num_sequences', type=int, default=500, 
                        help='Number of sequences to generate')
    parser.add_argument('--seq_length', type=int, default=15, 
                        help='Length of each sequence')
    parser.add_argument('--mult_factor', type=float, default=2.0, 
                        help='Strength of topological relationships')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='standard', 
                        choices=['standard', 'multiscale'], 
                        help='Type of KAN model to use')
    parser.add_argument('--hidden_dim', type=int, default=64, 
                        help='Hidden layer dimension')
    parser.add_argument('--embedding_dim', type=int, default=32, 
                        help='Event embedding dimension')
    parser.add_argument('--num_layers', type=int, default=2, 
                        help='Number of KAN layers')
    parser.add_argument('--spline_points', type=int, default=5, 
                        help='Number of points for B-spline basis')
    parser.add_argument('--spline_order', type=int, default=3, 
                        help='Order of B-splines')
    parser.add_argument('--input_window', type=int, default=10, 
                        help='Input sequence window length')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=50, 
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, 
                        help='Weight decay for regularization')
    parser.add_argument('--patience', type=int, default=10, 
                        help='Patience for early stopping')
    
    # Generation parameters
    parser.add_argument('--temperature', type=float, default=1.0, 
                        help='Temperature for sampling (higher = more random)')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed')
    parser.add_argument('--device', type=str, default=None, 
                        help='Device to use (cpu, cuda, auto, mps)')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Directory to save results')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg):
    """Determine which device to use"""
    if device_arg == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_arg == 'cpu':
        return torch.device('cpu')
    else:  # 'auto' or None
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def generate_sequences(args):
    """Generate synthetic event sequences"""
    # Define labels and topology
    labels = ['A', 'B', 'C', 'D', 'E']
    topology = {
        'A': [],
        'B': ['A'],
        'C': ['A', 'B'],
        'D': ['A'],
        'E': ['A', 'B']
    }
    
    # Generate sequences
    sequences = {}
    for i in range(args.num_sequences):
        sequences[i] = gen_synth_data_from_topology(
            labels, topology, args.seq_length, args.mult_factor
        )
    
    # Create label mappings
    label_to_idx = {'<pad>': 0}
    for label in labels:
        label_to_idx[label] = len(label_to_idx)
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}
    
    return sequences, labels, topology, label_to_idx, idx_to_label


def prepare_datasets(sequences, label_to_idx, args):
    """Prepare training, validation, and test datasets"""
    # Split into train, validation, and test sets
    all_ids = list(sequences.keys())
    train_ids, temp_ids = train_test_split(all_ids, test_size=0.3, random_state=args.seed)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=args.seed)
    
    # Create separate sequence dictionaries
    train_sequences = {k: sequences[k] for k in train_ids}
    val_sequences = {k: sequences[k] for k in val_ids}
    test_sequences = {k: sequences[k] for k in test_ids}
    
    # Create datasets
    train_dataset = EventSequenceDataset(train_sequences, label_to_idx, seq_length=args.input_window)
    val_dataset = EventSequenceDataset(val_sequences, label_to_idx, seq_length=args.input_window)
    test_dataset = EventSequenceDataset(test_sequences, label_to_idx, seq_length=args.input_window)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    return train_loader, val_loader, test_loader, train_sequences, val_sequences, test_sequences


def create_model(num_events, args, device):
    """Create the KAN model based on arguments"""
    if args.model_type == 'standard':
        model = EventSequenceKAN(
            num_events=num_events,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            sequence_length=args.input_window,
            num_splines=args.spline_points,
            spline_order=args.spline_order,
            device=device
        )
    else:  # 'multiscale'
        model = MultiScaleEventSequenceKAN(
            num_events=num_events,
            hidden_dim=args.hidden_dim,
            embedding_dim=args.embedding_dim,
            num_layers=args.num_layers,
            sequence_length=args.input_window,
            num_splines=args.spline_points,
            spline_order=args.spline_order,
            device=device
        )
    
    print(f"Created {args.model_type} KAN model with {sum(p.numel() for p in model.parameters())} parameters")
    return model


def train(model, train_loader, val_loader, args, device):
    """Train the model"""
    # Setup
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_state = None
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    
    # Training loop
    for epoch in range(args.num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss = train_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = val_correct / val_total
        history['val_loss'].append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}: "
              f"Train loss: {train_loss:.4f}, "
              f"Val loss: {val_loss:.4f}, "
              f"Val accuracy: {val_accuracy:.4f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model state
    print(f"Best model found at epoch {best_epoch+1} with validation loss: {best_val_loss:.4f}")
    model.load_state_dict(best_state)
    
    return model, history


def evaluate(model, test_loader, idx_to_label, device):
    """Evaluate the model on test data"""
    model.eval()
    
    # Setup
    criterion = nn.CrossEntropyLoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    # Confusion matrix (using dictionaries for labels)
    confusion = {}
    for label in idx_to_label.values():
        if label != '<pad>':
            confusion[label] = {other: 0 for other in idx_to_label.values() if other != '<pad>'}
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Update metrics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
            # Update confusion matrix
            for true, pred in zip(targets.cpu().numpy(), predicted.cpu().numpy()):
                if true in idx_to_label and pred in idx_to_label:
                    true_label = idx_to_label[true]
                    pred_label = idx_to_label[pred]
                    if true_label != '<pad>' and pred_label != '<pad>':
                        confusion[true_label][pred_label] += 1
    
    # Calculate metrics
    accuracy = correct / total
    test_loss = test_loss / len(test_loader.dataset)
    
    # Print results
    print(f"Test accuracy: {accuracy:.4f}, Test loss: {test_loss:.4f}")
    
    return accuracy, test_loss, confusion


def predict_next_events(model, test_sequences, idx_to_label, label_to_idx, args):
    """Demonstrate next event prediction"""
    model.eval()
    
    results = []
    
    # Select a few test sequences
    test_ids = list(test_sequences.keys())[:5]
    
    for seq_id in test_ids:
        sequence = test_sequences[seq_id]
        
        # Try predicting at different points in the sequence
        prediction_points = [
            len(sequence) // 4,
            len(sequence) // 2,
            3 * len(sequence) // 4
        ]
        
        sequence_results = []
        
        for point in prediction_points:
            prefix = sequence[:point]
            
            # Get next event prediction
            next_event, prob, prob_dist = model.predict_next_event(
                prefix, idx_to_label, label_to_idx, temperature=args.temperature
            )
            
            # Get actual next event
            actual_next = sequence[point] if point < len(sequence) else None
            is_correct = (next_event == actual_next) if actual_next else None
            
            prediction = {
                'prefix_length': point,
                'prefix': prefix,
                'predicted': next_event,
                'probability': prob,
                'distribution': {k: float(v) for k, v in prob_dist.items()},
                'actual': actual_next,
                'correct': is_correct
            }
            
            sequence_results.append(prediction)
            
            # Print result
            print(f"Sequence {seq_id}, Prefix length {point}:")
            print(f"  Prefix: {prefix}")
            print(f"  Predicted next: {next_event} (p={prob:.4f})")
            print(f"  Actual next: {actual_next}")
            print(f"  Correct: {is_correct}")
            print(f"  Top 3 probabilities:")
            for label, p in sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)[:3]:
                print(f"    {label}: {p:.4f}")
            print()
        
        results.append({
            'sequence_id': seq_id,
            'sequence': sequence,
            'predictions': sequence_results
        })
    
    return results


def generate_sequences_from_prefix(model, test_sequences, idx_to_label, label_to_idx, args):
    """Generate complete sequences from prefixes"""
    model.eval()
    
    results = []
    
    # Select a few test sequences
    test_ids = list(test_sequences.keys())[:5]
    
    for seq_id in test_ids:
        original_sequence = test_sequences[seq_id]
        
        # Use first third as prefix
        prefix_length = len(original_sequence) // 3
        prefix = original_sequence[:prefix_length]
        
        # Generate the rest of the sequence
        generated_sequence, probabilities = model.generate_sequence(
            prefix, len(original_sequence), idx_to_label, label_to_idx, 
            temperature=args.temperature
        )
        
        # Calculate accuracy of generated continuation
        correct = 0
        for i in range(prefix_length, len(original_sequence)):
            if i < len(generated_sequence) and original_sequence[i] == generated_sequence[i]:
                correct += 1
        
        total = len(original_sequence) - prefix_length
        accuracy = correct / total if total > 0 else 0
        
        result = {
            'sequence_id': seq_id,
            'original': original_sequence,
            'prefix': prefix,
            'generated': generated_sequence,
            'probabilities': [(event, float(prob)) for event, prob, _ in probabilities],
            'accuracy': accuracy
        }
        
        results.append(result)
        
        # Print result
        print(f"Sequence {seq_id}:")
        print(f"  Original: {original_sequence}")
        print(f"  Prefix: {prefix}")
        print(f"  Generated: {generated_sequence}")
        print(f"  Accuracy: {accuracy:.4f}")
        print()
    
    return results


def save_results(model, history, metrics, predictions, generations, args, output_dir):
    """Save model, metrics, and results"""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    
    # Save metrics and results
    results = {
        'args': vars(args),
        'history': {k: [float(v) for v in vals] for k, vals in history.items()},
        'metrics': metrics,
        'next_event_predictions': predictions,
        'sequence_generations': generations
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()
    
    print(f"Results saved to {output_dir}")


def main():
    # Parse arguments
    args = parse_arguments()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = get_device(args.device)

    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                  "and/or you do not have an MPS-enabled device on this machine.")
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.model_type}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate sequences
    print("Generating event sequences")
    sequences, labels, topology, label_to_idx, idx_to_label = generate_sequences(args)
    
    # Prepare datasets and loaders
    print("Preparing datasets")
    train_loader, val_loader, test_loader, train_sequences, val_sequences, test_sequences = prepare_datasets(
        sequences, label_to_idx, args
    )
    
    # Create model
    print("Creating model")
    model = create_model(len(label_to_idx), args, device)
    
    # Train model
    print("Training model")
    model, history = train(model, train_loader, val_loader, args, device)
    
    # Evaluate model
    print("Evaluating model")
    accuracy, test_loss, confusion = evaluate(model, test_loader, idx_to_label, device)
    metrics = {
        'test_accuracy': accuracy,
        'test_loss': test_loss,
        'confusion_matrix': confusion
    }
    
    # Predict next events
    print("Predicting next events...")
    predictions = predict_next_events(model, test_sequences, idx_to_label, label_to_idx, args)
    
    # Generate sequences from prefixes
    print("Generating sequences from prefixes")
    generations = generate_sequences_from_prefix(model, test_sequences, idx_to_label, label_to_idx, args)
    
    # Save results
    save_results(model, history, metrics, predictions, generations, args, output_dir)
    
    print("Done")


if __name__ == "__main__":
    main()
