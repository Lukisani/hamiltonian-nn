import torch
import argparse
import numpy as np
import os, sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PARENT_DIR)

from nn_models import MLP
from hnn import HNN
from data3d import get_dataset  # Ensure dataset is structured for 3D
from utils import L2_loss, to_pickle, from_pickle

def get_args():
    parser = argparse.ArgumentParser(description="Train Baseline & HNN in 3D")
    parser.add_argument('--input_dim', default=3*6, type=int, help='Input dimensions for a 3-body system in 3D')
    parser.add_argument('--hidden_dim', default=256, type=int, help='Hidden layer size for MLP')
    parser.add_argument('--learn_rate', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--batch_size', default=512, type=int, help='Batch size')
    parser.add_argument('--total_steps', default=15000, type=int, help='Total gradient steps')
    parser.add_argument('--print_every', default=500, type=int, help='Steps between logs')
    parser.add_argument('--name', default='3body_3D', type=str, help='Dataset name')
    parser.add_argument('--baseline', action='store_true', help='Train baseline instead of HNN')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    parser.add_argument('--save_dir', default=THIS_DIR, type=str, help='Directory to save models')
    return parser.parse_args()

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    output_dim = args.input_dim if args.baseline else 3  # 3 for dH/dq, dH/dp
    nn_model = MLP(args.input_dim, args.hidden_dim, output_dim, 'tanh')
    model = HNN(args.input_dim, differentiable_model=nn_model, field_type='solenoidal', baseline=args.baseline)
    optim = torch.optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=1e-4)
    
    data = get_dataset(args.name, args.save_dir, verbose=args.verbose)
    x = torch.tensor(data['coords'], requires_grad=True, dtype=torch.float32)
    dxdt = torch.tensor(data['dcoords'], dtype=torch.float32)
    test_x = torch.tensor(data['test_coords'], requires_grad=True, dtype=torch.float32)
    test_dxdt = torch.tensor(data['test_dcoords'], dtype=torch.float32)
    
    stats = {'train_loss': [], 'test_loss': []}
    for step in range(args.total_steps + 1):
        ixs = torch.randperm(x.shape[0])[:args.batch_size]
        dxdt_hat = model.time_derivative(x[ixs])
        loss = L2_loss(dxdt[ixs], dxdt_hat)
        loss.backward()
        optim.step()
        optim.zero_grad()
        
        test_ixs = torch.randperm(test_x.shape[0])[:args.batch_size]
        test_dxdt_hat = model.time_derivative(test_x[test_ixs])
        test_loss = L2_loss(test_dxdt[test_ixs], test_dxdt_hat)
        
        stats['train_loss'].append(loss.item())
        stats['test_loss'].append(test_loss.item())
        
        if args.verbose and step % args.print_every == 0:
            print(f"Step {step}, Train Loss: {loss.item():.4e}, Test Loss: {test_loss.item():.4e}")
    
    return model, stats

def run_model(args, label):
    model, stats = train(args)
    os.makedirs(args.save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), f'{args.save_dir}/{args.name}-{label}.tar')
    to_pickle(stats, f'{args.save_dir}/{args.name}-{label}.pkl')
    
if __name__ == "__main__":
    args = get_args()
    print("Training Baseline...")
    args.baseline = True
    run_model(args, 'baseline')
    
    print("Training HNN...")
    args.baseline = False
    run_model(args, 'hnn')
    
    print("Training complete.")
