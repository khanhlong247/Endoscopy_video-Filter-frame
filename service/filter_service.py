import numpy as np

def run_global_vectorized_filtering(db_results, threshold: float):
    frames = []
    for emb, meta in zip(db_results['embeddings'], db_results['metadatas']):
        frames.append({
            "embedding": emb,
            "metadata": meta
        })
    
    frames.sort(key=lambda x: x['metadata']['timestamp'])
    
    if not frames:
        return []

    E_keep = np.array([frames[0]['embedding']])
    keep_frames_meta = [frames[0]['metadata']]

    for i in range(1, len(frames)):
        v_i = np.array(frames[i]['embedding'])
        
        S = np.dot(E_keep, v_i) 
        
        S_max = np.max(S)
        
        if S_max < threshold:
            E_keep = np.vstack([E_keep, v_i])
            keep_frames_meta.append(frames[i]['metadata'])
            
    return keep_frames_meta