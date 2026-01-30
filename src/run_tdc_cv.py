import argparse
import os
import random
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tqdm import tqdm

from .dataset_map import infer_task_type
from .metric import get_metric
from .deep_models import GCNN, GraphFeaturizer, GraphDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import tempfile
import pathlib

def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Required for some PyTorch operations to be deterministic (e.g. scatter_add)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Avoid errors if a specific op doesn't have a deterministic implementation,
    # or use warn_only=True
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except AttributeError:
        # Older pytorch versions might not support warn_only
        pass

def parse_args():
    parser = argparse.ArgumentParser(description="Run 5-fold CV on TDC processed datasets")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset (e.g., AMES)")
    parser.add_argument("--target_col", type=str, required=True, help="Target column name (e.g., AMES)")
    parser.add_argument("--task_type", type=str, choices=['classification', 'regression', 'auto'], default='auto', help="Task type")
    parser.add_argument("--model_type", type=str, choices=['rf', 'xgboost', 'gcnn'], default='rf', help="Model type to use")
    parser.add_argument("--base_path", type=str, default="./../tdc/data/processed", help="Base path to data")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output Directory")
    parser.add_argument("--id_col", type=str, default=None, help="Column name for unique ID")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    if args.task_type == 'auto':
        found = False
        # 1. Try mapping by dataset_name
        try:
            args.task_type = infer_task_type(args.dataset_name)
            print(f"Auto-detected task type from dataset name '{args.dataset_name}': {args.task_type}")
            found = True
        except KeyError:
            pass
            
        # 2. If failed, try mapping by target_col
        if not found:
            try:
                args.task_type = infer_task_type(args.target_col)
                print(f"Auto-detected task type from target col '{args.target_col}': {args.task_type}")
                found = True
            except KeyError:
                pass
                
        if not found:
             print(f"Warning: Could not infer task type for {args.dataset_name}/{args.target_col}. Defaulting to classification.")
             args.task_type = 'classification'
             
    return args

def smile_to_fp(smile, n_bits=2048, radius=2):
    try:
        mol = Chem.MolFromSmiles(smile)
        if mol is None:
            return np.zeros(n_bits)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp)
    except:
        return np.zeros(n_bits)

def load_and_featurize(path, target_col, id_col=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path)
    print(f"Loading {path}: {len(df)} samples")
    
    # Featurize
    tqdm.pandas(desc="Calculating Fingerprints")
    X = np.stack(df['smiles'].progress_apply(smile_to_fp).values)
    y = df[target_col].values
    
    if id_col and id_col in df.columns:
        ids = df[id_col].values
    else:
        ids = df.index.values
        
    return X, y, ids

def evaluate(y_true, y_pred, y_prob, task_type):
    return get_metric(y_true, y_pred, task_type, y_prob)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    all_metrics = []
        
    for fold in range(1, 6):
        
        te_pred = []

        print(f"\n{'='*20} Fold {fold} {'='*20}")
        fold_dir = os.path.join(args.base_path, args.dataset_name, args.target_col, f"fold{fold}")
        
        # # Load Data
        # print("Processing Train/Val...")
        # # X_train, y_train = load_and_featurize(os.path.join(fold_dir, 'train.csv'), args.target_col)
        # # X_val, y_val = load_and_featurize(os.path.join(fold_dir, 'val.csv'), args.target_col)
        
        # print("Processing Test...")
        # # X_test, y_test = load_and_featurize(os.path.join(fold_dir, 'test.csv'), args.target_col)
        
        # # Merge Train + Val for full training
        # # X_full_train = np.vstack([X_train, X_val])
        # # y_full_train = np.concatenate([y_train, y_val])
        
        if args.model_type == 'gcnn':
             # GCNN Data Preparation
             print("Featurizing for GCNN...")
             # Load raw dataframes first
             df_train = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
             df_val = pd.read_csv(os.path.join(fold_dir, 'val.csv'))
             df_test = pd.read_csv(os.path.join(fold_dir, 'test.csv'))
             
             y_train = df_train[args.target_col].values
             y_val = df_val[args.target_col].values
             y_test = df_test[args.target_col].values
             
             if args.id_col and args.id_col in df_test.columns:
                 ids_test = df_test[args.id_col].values
             else:
                 ids_test = df_test.index.values
             
             featurizer = GraphFeaturizer(y_column='target') # Dummy y_column, not used in logic but required by init
             
             X_train_graphs = featurizer(df_train['smiles'])
             X_val_graphs = featurizer(df_val['smiles'])
             X_test_graphs = featurizer(df_test['smiles'])
             
             # Create temp dir for InMemoryDataset
             temp_dir = tempfile.mkdtemp()
             try:
                 # Train + Val for full training
                 train_val_graphs = X_train_graphs + X_val_graphs
                 train_val_y = np.concatenate([y_train, y_val])
                 
                 train_dataset = GraphDataset(train_val_graphs, train_val_y, root=pathlib.Path(temp_dir)/'train')
                 test_dataset = GraphDataset(X_test_graphs, y_test, root=pathlib.Path(temp_dir)/'test')
                 
                 train_loader = GraphDataLoader(train_dataset, batch_size=64, shuffle=True)
                 test_loader = GraphDataLoader(test_dataset, batch_size=len(y_test), shuffle=False)
                 
                 # Model Init
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                 model = GCNN(n_layers=5, hidden_size=150, dp=0.1).to(device)
                 optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                 criterion = nn.MSELoss() if args.task_type == 'regression' else nn.BCEWithLogitsLoss()
                 
                 # Train Loop
                 print(f"Training GCNN on {device}...")
                 model.train()
                 for epoch in range(50):
                     pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/50", leave=True)
                     for batch in pbar:
                         batch = batch.to(device)
                         optimizer.zero_grad()
                         out = model(batch.x, batch.edge_index, batch.batch)
                         
                         if args.task_type == 'classification':
                             loss = criterion(out.view(-1), batch.y.float())
                         else:
                             loss = criterion(out.view(-1), batch.y.float())
                             
                         loss.backward()
                         optimizer.step()
                         pbar.set_postfix({'loss': loss.item()})
                 
                 # Inference
                 model.eval()
                 y_pred_list = []
                 y_prob_list = []
                 with torch.no_grad():
                     for batch in test_loader:
                         batch = batch.to(device)
                         out = model(batch.x, batch.edge_index, batch.batch)
                         if args.task_type == 'classification':
                              probs = torch.sigmoid(out).view(-1).cpu().numpy()
                              preds = (probs > 0.5).astype(int)
                              y_prob_list.extend(probs)
                              y_pred_list.extend(preds)
                         else:
                              preds = out.view(-1).cpu().numpy()
                              y_pred_list.extend(preds)
                 
                 y_pred = np.array(y_pred_list)
                 y_prob = np.array(y_prob_list) if args.task_type == 'classification' else None
                 
             finally:
                 shutil.rmtree(temp_dir, ignore_errors=True)

        else:
            # Shallow Models (RF/XGB)
            # Load Data (using fingerprints)
            print("Processing Train/Val/Test with Morgan Fingerprints...")
            X_train, y_train, _ = load_and_featurize(os.path.join(fold_dir, 'train.csv'), args.target_col, args.id_col)
            X_val, y_val, _ = load_and_featurize(os.path.join(fold_dir, 'val.csv'), args.target_col, args.id_col)
            X_test, y_test, ids_test = load_and_featurize(os.path.join(fold_dir, 'test.csv'), args.target_col, args.id_col)

            # Merge Train + Val for full training
            X_full_train = np.vstack([X_train, X_val])
            y_full_train = np.concatenate([y_train, y_val])
            
            # Train Model
            print(f"Training {args.model_type}...")
            if args.task_type == 'classification':
                if args.model_type == 'rf':
                    model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=args.seed)
                elif args.model_type == 'xgboost':
                    from xgboost import XGBClassifier
                    model = XGBClassifier(n_estimators=100, n_jobs=-1, random_state=args.seed, use_label_encoder=False, eval_metric='logloss')
                else:
                    raise ValueError(f"Unknown model type: {args.model_type}")
                    
                model.fit(X_full_train, y_full_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]
            else:
                if args.model_type == 'rf':
                    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=args.seed)
                elif args.model_type == 'xgboost':
                    from xgboost import XGBRegressor
                    model = XGBRegressor(n_estimators=100, n_jobs=-1, random_state=args.seed)
                else:
                    raise ValueError(f"Unknown model type: {args.model_type}")

                model.fit(X_full_train, y_full_train)
                y_pred = model.predict(X_test)
                y_prob = None
            
        # Evaluate
        fold_output_dir = os.path.join(args.output_dir, args.dataset_name, args.target_col, f"fold{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)

        pred_data = {
            'id': ids_test,
            'y_true': y_test,
            'y_pred': y_pred
        }
        if y_prob is not None:
             pred_data['y_prob'] = y_prob
             
        df_pred = pd.DataFrame(pred_data)
        df_pred.to_csv(os.path.join(fold_output_dir, "predictions.csv"), index=False)

        
        fold_metrics = evaluate(y_test, y_pred, y_prob, args.task_type)
        print(f"Fold {fold} Metrics: {fold_metrics}")
        
        pd.DataFrame([fold_metrics]).to_csv(os.path.join(fold_output_dir, "metrics.csv"), index=False)
        
        all_metrics.append(fold_metrics)
        
    # Aggregate Results
    print(f"\n{'='*20} Final Results {'='*20}")
    df_metrics = pd.DataFrame(all_metrics)
    summary = df_metrics.describe().loc[['mean', 'std']]
    
    print(df_metrics)
    print("\nAverage Performance:")
    print(summary)
    
    # Create flattened dictionary for single row output
    flat_metrics = {}
    for col in df_metrics.columns:
        flat_metrics[f"{col}_mean"] = summary.loc['mean', col]
        flat_metrics[f"{col}_std"] = summary.loc['std', col]
    
    df_save = pd.DataFrame([flat_metrics])
    
    # Save results
    output_file = os.path.join(args.output_dir, args.dataset_name, args.target_col, "metrics.csv")
    df_save.to_csv(output_file, index=False)
    print(f"\nSaved aggregated results to {output_file}")

if __name__ == "__main__":
    main()
