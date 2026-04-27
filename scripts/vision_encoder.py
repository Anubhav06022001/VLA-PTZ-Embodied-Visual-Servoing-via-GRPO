import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image

class DINOv2Encoder:
    """
    Agent 1: The Perception Module.
    Extracts dense spatial features from MuJoCo camera views to calculate 
    camera drift without tracking gradients.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        # Load the lightweight ViT-Small version of DINOv2 (~21M parameters)
        model_id = "facebook/dinov2-small"
        self.device = device
        
        print(f"[VisionEncoder] Loading {model_id} on {self.device}...")
        
        # The processor automatically handles 224x224 resizing, center cropping, 
        # and standard ImageNet normalization required by DINOv2.
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        
        # Load model and immediately set to evaluation mode
        self.model = AutoModel.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # CRITICAL: Explicitly freeze all parameters. 
        # This prevents the RL backprop from accidentally updating the vision model,
        # which would instantly cause an Out-of-Memory (OOM) crash.
        for param in self.model.parameters():
            param.requires_grad = False

    def get_cls_token(self, image):
        """
        Processes a raw image and extracts the DINOv2 Class (CLS) token.
        """
        # MuJoCo usually returns numpy arrays. The HF processor prefers PIL.
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Prepare inputs and push to the correct GPU/CPU
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        # Forward pass without tracking gradients
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # The CLS token is the first token [:, 0, :] in the sequence of the last hidden state.
        # It acts as the global representation of the visual layout.
        # Shape: (Batch, Sequence_Length, Hidden_Size) -> (1, 384)
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    def compute_delta_and_score(self, ref_image, cur_image):
        """
        Calculates the drift vector for the LLM prompt and the alignment score for the reward.
        
        Args:
            ref_image: The perfect preset view.
            cur_image: The current drifted view.
            
        Returns:
            v_delta (torch.Tensor): The mathematical difference vector. Shape (384,).
            score (float): The cosine similarity between -1.0 and 1.0.
        """
        cls_ref = self.get_cls_token(ref_image)
        cls_cur = self.get_cls_token(cur_image)
        
        # 1. Compute the Visual Delta (The Observation State for the LLM)
        # Squeeze removes the batch dimension, leaving a flat 1D vector of shape (384,)
        v_delta = (cls_ref - cls_cur).squeeze(0)
        
        # 2. Compute the Cosine Similarity (The Reward Score for GRPO)
        # dim=1 because inputs are shaped (1, 384) before squeezing
        score = F.cosine_similarity(cls_ref, cls_cur, dim=1).item()
        
        return v_delta, score

# --- Local Testing Block ---
if __name__ == "__main__":
    # If you run `python rl_training/vision_encoder.py` directly, it will test the pipeline.
    print("Testing Vision Encoder Initialization...")
    encoder = DINOv2Encoder()
    
    # Create dummy images representing MuJoCo numpy outputs (Height, Width, Channels)
    dummy_ref = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_cur = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    v_delta, score = encoder.compute_delta_and_score(dummy_ref, dummy_cur)
    
    print("\n--- Test Results ---")
    print(f"Visual Delta Vector Shape: {v_delta.shape}")
    print(f"Cosine Similarity Score: {score:.4f}")
    print("Test passed successfully!")



# from rl_training.vision_encoder import DINOv2Encoder

# # Initialize globally so it stays loaded in VRAM
# vision_agent = DINOv2Encoder()