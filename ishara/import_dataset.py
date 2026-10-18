"""Import and convert international datasets (BdSLW60, WLASL) to Ishara format.

Usage:
  python import_dataset.py --source bdslw60 --data bdslw60_processed
  python import_dataset.py --source wlasl --data wlasl_processed
  python train.py --data bdslw60_processed --out models/ishara_bdslw60.joblib
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np


def convert_bdslw60(source_path, output_path):
    """Convert BdSLW60 dataset to (30, 144) format.
    
    BdSLW60 assumed structure:
    - BdSLW60/
      - sign_name/
        - signer_id/
          - video_id_landmarks.npy (shape varies)
    """
    print(f"Converting BdSLW60 from {source_path} to {output_path}")
    
    source = Path(source_path)
    output = Path(output_path)
    output.mkdir(exist_ok=True)
    
    stats = {"total": 0, "converted": 0, "skipped": 0}
    
    # Iterate over signs
    for sign_dir in source.iterdir():
        if not sign_dir.is_dir():
            continue
        
        sign_name = sign_dir.name
        sign_output = output / sign_name
        sign_output.mkdir(exist_ok=True)
        
        # Iterate over signers
        for signer_dir in sign_dir.iterdir():
            if not signer_dir.is_dir():
                continue
            
            signer_id = signer_dir.name
            signer_output = sign_output / signer_id
            signer_output.mkdir(exist_ok=True)
            
            # Process landmark files
            for lm_file in signer_dir.glob("*_landmarks.npy"):
                try:
                    lm = np.load(lm_file)  # Assume shape (T, D)
                    
                    # Resample to (30, 144) - placeholder conversion
                    if lm.shape[0] != 30 or lm.shape[1] != 144:
                        # Simple temporal resampling
                        from scipy.interpolate import interp1d
                        t_old = np.linspace(0, 1, lm.shape[0])
                        t_new = np.linspace(0, 1, 30)
                        
                        resampled = np.zeros((30, 144), dtype=np.float32)
                        for d in range(min(lm.shape[1], 144)):
                            f = interp1d(t_old, lm[:, d], kind='linear', fill_value='extrapolate')
                            resampled[:, d] = f(t_new).astype(np.float32)
                    else:
                        resampled = lm.astype(np.float32)
                    
                    # Save
                    out_file = signer_output / lm_file.name.replace("_landmarks.npy", ".npy")
                    np.save(out_file, resampled)
                    stats["converted"] += 1
                
                except Exception as e:
                    print(f"  Skip {lm_file}: {e}")
                    stats["skipped"] += 1
                
                stats["total"] += 1
    
    print(f"✓ Conversion complete:")
    print(f"  Total: {stats['total']}, Converted: {stats['converted']}, Skipped: {stats['skipped']}")
    return stats


def convert_wlasl(source_path, output_path):
    """Convert WLASL dataset to (30, 144) format.
    
    WLASL assumed structure:
    - WLASL/
      - dictionary.json
      - videos/
        - {video_id}_landmarks.npy
    """
    print(f"Converting WLASL from {source_path} to {output_path}")
    
    source = Path(source_path)
    output = Path(output_path)
    output.mkdir(exist_ok=True)
    
    # Load video-to-sign mapping
    dict_file = source / "dictionary.json"
    if not dict_file.exists():
        print("✗ dictionary.json not found in WLASL directory")
        return {"total": 0, "converted": 0, "skipped": 0}
    
    with open(dict_file) as f:
        video_map = json.load(f)  # {video_id: {sign: ..., video: ...}}
    
    stats = {"total": 0, "converted": 0, "skipped": 0}
    
    # Create sign directories
    signs_seen = {}
    for video_id, meta in video_map.items():
        sign_name = meta.get("sign", "unknown")
        if sign_name not in signs_seen:
            signs_seen[sign_name] = 0
            (output / sign_name).mkdir(exist_ok=True)
        signs_seen[sign_name] += 1
    
    # Process landmark files
    for lm_file in source.glob("videos/*_landmarks.npy"):
        try:
            video_id = lm_file.stem.replace("_landmarks", "")
            meta = video_map.get(video_id, {})
            sign_name = meta.get("sign", "unknown")
            
            lm = np.load(lm_file)
            
            # Resample to (30, 144)
            if lm.shape[0] != 30 or lm.shape[1] != 144:
                from scipy.interpolate import interp1d
                t_old = np.linspace(0, 1, lm.shape[0])
                t_new = np.linspace(0, 1, 30)
                
                resampled = np.zeros((30, 144), dtype=np.float32)
                for d in range(min(lm.shape[1], 144)):
                    f = interp1d(t_old, lm[:, d], kind='linear', fill_value='extrapolate')
                    resampled[:, d] = f(t_new).astype(np.float32)
            else:
                resampled = lm.astype(np.float32)
            
            # Save to sign directory
            out_file = output / sign_name / f"{video_id}.npy"
            np.save(out_file, resampled)
            stats["converted"] += 1
        
        except Exception as e:
            print(f"  Skip {lm_file}: {e}")
            stats["skipped"] += 1
        
        stats["total"] += 1
    
    print(f"✓ Conversion complete:")
    print(f"  Total: {stats['total']}, Converted: {stats['converted']}, Skipped: {stats['skipped']}")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["bdslw60", "wlasl"], required=True)
    ap.add_argument("--input", required=True, help="Path to source dataset")
    ap.add_argument("--data", default="data_imported")
    a = ap.parse_args()
    
    if a.source == "bdslw60":
        convert_bdslw60(a.input, a.data)
    elif a.source == "wlasl":
        convert_wlasl(a.input, a.data)


if __name__ == "__main__":
    main()
