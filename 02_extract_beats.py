# ==================== SETUP BEATs EMBEDDING EXTRACTION ====================
# [Keep Code 1's original cloning/setup code here]

# ==================== NEW ADDITIONS FROM CODE 2 ====================
# Reproducibility seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Validation utility
def validate_inputs(file_path, expected_shape=None):
    [Same as Code 2]

# ==================== MODIFIED EMBEDDING EXTRACTION ====================
def extract_beats_embedding(wav: np.ndarray, sr: int=16000, device="cpu") -> np.ndarray:
    [Code 1's version with Code 2's shape check added]
    
def process_machine_split(...):
    [Code 1's version with device parameter and validate_inputs call added]

# ==================== KEEP CODE 1'S MAIN EXECUTION FLOW ====================
if __name__ == "__main__":
    [Code 1's original main logic with device parameter propagation]
