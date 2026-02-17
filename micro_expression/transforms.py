import torch
import torchvision.transforms as T
from torchvision.transforms import functional as F
import random
import numpy as np

def create_spatial_transforms(image_size=(128, 128), is_train=True):
    """
    Creates a composition of spatial transforms similar to the Keras ImageDataGenerator
    used in the original project.

    Args:
        image_size (tuple): Target image size (height, width).
        is_train (bool): If True, applies augmentation. Otherwise, only resizes and converts to tensor.

    Returns:
        callable: A torchvision transforms composition.
    """
    
    # Note: Input to these transforms should be PyTorch Tensor (C, H, W)
    # The MicroExpressionDataset ensures output is (1, H, W) float tensor

    if is_train:
        # Mimic Keras ImageDataGenerator: rotation, zoom, horizontal_flip
        # Keras zoom_range=0.1 -> Scale factor between [0.9, 1.1] (approx)
        # Keras rotation_range=15
        transform_list = [
            T.RandomAffine(degrees=15, 
                           scale=(0.9, 1.1), 
                           translate=(0.05, 0.05), # Added slight translation similar to width/height_shift
                           fill=0), # Fill with black for grayscale
            T.RandomHorizontalFlip(p=0.5),
            # Ensure output is still Tensor of correct shape and type
             T.ConvertImageDtype(torch.float32) # Ensure float32 after transforms
        ]
    else:
        # For validation/testing, just ensure correct type
         transform_list = [
             T.ConvertImageDtype(torch.float32)
         ]
         
    # Note: Resizing and Grayscale conversion happens in the Dataset's _load_frame method
    # Normalization (to [0,1]) also happens there.
    # If further normalization (e.g., mean/std) is needed, add T.Normalize here.

    return T.Compose(transform_list)


# --- Temporal Augmentation (Placeholder/Concept) ---

class ApplySameRandomTransformSequence:
    """
    A wrapper transform that applies the *same* randomly chosen parameters 
    of a given transform to all frames in a sequence tensor.
    This mimics the behavior of the augment_sequence function in the original LSTM script.
    
    Args:
        transform (callable): A torchvision transform (e.g., RandomAffine, RandomHorizontalFlip) 
                              that determines its parameters randomly on call.
    """
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sequence_tensor):
        """
        Args:
            sequence_tensor (torch.Tensor): Input sequence tensor (T, C, H, W).
        Returns:
            torch.Tensor: Transformed sequence tensor (T, C, H, W).
        """
        # Determine the random parameters using the first frame
        # This works for transforms like RandomAffine, RandomHorizontalFlip
        # We need to be careful with stateful transforms or ones dependent on input values.
        
        # Hacky way: Call transform once to get state (e.g., for RandomHorizontalFlip)
        # Or, rely on functional interface if available and parameters can be generated once.
        
        # Example for RandomHorizontalFlip:
        if isinstance(self.transform, T.RandomHorizontalFlip):
            apply_flip = torch.rand(1) < self.transform.p
            transformed_frames = []
            for i in range(sequence_tensor.size(0)):
                frame = sequence_tensor[i]
                if apply_flip:
                    frame = F.hflip(frame)
                transformed_frames.append(frame)
            return torch.stack(transformed_frames, dim=0)
            
        # Example for RandomAffine (more complex as parameters are generated internally)
        # A potentially cleaner way is to manually generate parameters once and use functional API
        elif isinstance(self.transform, T.RandomAffine):
            # 1. Get random parameters
            # Note: T.RandomAffine.get_params is protected, we might need to reimplement
            # or find another way. Let's simulate getting the params:
            ret = T.RandomAffine.get_params(
                self.transform.degrees, self.transform.translate, self.transform.scale,
                self.transform.shear, sequence_tensor.shape[-2:] # Get H, W
            )
            angle, translations, scale, shear = ret
            
            # 2. Apply the *same* parameters to all frames using functional API
            transformed_frames = []
            for i in range(sequence_tensor.size(0)):
                frame = sequence_tensor[i]
                transformed_frame = F.affine(frame, angle=angle, translate=list(translations),
                                             scale=scale, shear=list(shear), 
                                             interpolation=self.transform.interpolation,
                                             fill=self.transform.fill)
                transformed_frames.append(transformed_frame)
            return torch.stack(transformed_frames, dim=0)

        else:
            # Fallback: Apply transform individually (loses temporal consistency)
            # This is NOT what we want for sequence augmentation usually.
            # logger.warning("Applying transform individually to frames - temporal consistency lost!")
            # return torch.stack([self.transform(sequence_tensor[i]) for i in range(sequence_tensor.size(0))], dim=0)
            # Or raise error if consistency is required
             raise TypeError(f"ApplySameRandomTransformSequence doesn't directly support {type(self.transform)}. Needs specific handling.")


# --- Function to create temporally consistent transforms (for LSTM stage) ---
def create_temporal_spatial_transforms(image_size=(128, 128), is_train=True):
    """
    Creates spatial transforms that are applied consistently across a sequence.
    """
    if is_train:
        # Define the base spatial transforms
        affine_transform = T.RandomAffine(degrees=15, 
                                        scale=(0.9, 1.1), 
                                        translate=(0.05, 0.05),
                                        fill=0)
        flip_transform = T.RandomHorizontalFlip(p=0.5)
        
        # Wrap them for consistent application across the sequence
        # Note: Order might matter. Apply affine first, then flip.
        transforms_list = [
            ApplySameRandomTransformSequence(affine_transform),
            ApplySameRandomTransformSequence(flip_transform),
            T.ConvertImageDtype(torch.float32) # Ensure type
        ]
        return T.Compose(transforms_list)
    else:
        # No augmentation for validation/test
        return T.Compose([T.ConvertImageDtype(torch.float32)])


# Example Usage (for testing)
if __name__ == '__main__':
    # --- Test Spatial Transforms (like for CNN stage) ---
    print("--- Testing Spatial Transforms ---")
    train_spatial_transform = create_spatial_transforms(is_train=True)
    test_spatial_transform = create_spatial_transforms(is_train=False)
    
    dummy_frame = torch.rand(1, 128, 128) # Example C, H, W tensor
    
    augmented_frame = train_spatial_transform(dummy_frame)
    print(f"Train spatial transform output shape: {augmented_frame.shape}, dtype: {augmented_frame.dtype}")
    assert augmented_frame.shape == dummy_frame.shape
    assert augmented_frame.dtype == torch.float32
    
    test_frame = test_spatial_transform(dummy_frame)
    print(f"Test spatial transform output shape: {test_frame.shape}, dtype: {test_frame.dtype}")
    assert test_frame.shape == dummy_frame.shape
    assert test_frame.dtype == torch.float32
    print("Spatial transforms tests passed.")

    # --- Test Temporal Spatial Transforms (like for LSTM stage) ---
    print("\n--- Testing Temporal Spatial Transforms ---")
    train_temporal_transform = create_temporal_spatial_transforms(is_train=True)
    test_temporal_transform = create_temporal_spatial_transforms(is_train=False)
    
    dummy_sequence = torch.rand(16, 1, 128, 128) # Example T, C, H, W tensor
    
    try:
        augmented_sequence = train_temporal_transform(dummy_sequence)
        print(f"Train temporal transform output shape: {augmented_sequence.shape}, dtype: {augmented_sequence.dtype}")
        assert augmented_sequence.shape == dummy_sequence.shape
        assert augmented_sequence.dtype == torch.float32
        # Check if transform was applied (hard to check consistency easily here)
        # print("Original frame 0 sum:", dummy_sequence[0].sum())
        # print("Augmented frame 0 sum:", augmented_sequence[0].sum())
        # print("Original frame 5 sum:", dummy_sequence[5].sum())
        # print("Augmented frame 5 sum:", augmented_sequence[5].sum())
        
        test_sequence = test_temporal_transform(dummy_sequence)
        print(f"Test temporal transform output shape: {test_sequence.shape}, dtype: {test_sequence.dtype}")
        assert test_sequence.shape == dummy_sequence.shape
        assert test_sequence.dtype == torch.float32
        
        print("Temporal spatial transforms tests passed.")
    except Exception as e:
         print(f"Error during Temporal Spatial Transforms test: {e}") 