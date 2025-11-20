import os
import tempfile

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.networks.nets import SwinUNETR
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    LoadImage,
    Resize,
    ScaleIntensity,
    Spacing,
)

# constants
MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f"]
CLASS_NAMES = ["Background", "Necrotic core", "Edema", "Enhancing tumor"]
CLASS_COLORS = [
    (0, 0, 0, 0),
    (1, 0, 0, 0.7),
    (0, 1, 0, 0.7),
    (0, 0, 1, 0.7),
]  # RGBA format for overlays
IMG_SIZE = (96, 96, 96)
IN_CHANNELS = 4
OUT_CHANNELS = 4
MODEL_PATH = "trials/swin_unetr_model.pth"


class Prediction:

    def __init__(self):
        pass

    def get_transform(
        self,
    ):
        return Compose(
            [
                LoadImage(image_only=True),
                EnsureChannelFirst(),
                Spacing(pixdim=(1.0, 1.0, 1.0), mode=("bilinear")),
                Resize(IMG_SIZE),
                ScaleIntensity(),
            ]
        )

    def is_valid_nifti(self, file):
        try:
            # Check file extension
            filename = file.name.lower()
            if not (filename.endswith(".nii") or filename.endswith(".nii.gz")):
                return False

            # Save to temp file and try loading with nibabel to verify it's a valid NIfTI
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as temp:
                temp.write(file.getvalue())
                temp_path = temp.name

            try:
                nib.load(temp_path)
                return True
            except:
                return False
            finally:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
        except:
            return False

    def load_model(
        self,
    ):
        if not os.path.exists(MODEL_PATH):
            return (
                None,
                "Model file not found at {MODEL_PATH}. Please ensure the model exists before proceeding.",
            )

        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SwinUNETR(
                img_size=IMG_SIZE,
                in_channels=IN_CHANNELS,
                out_channels=OUT_CHANNELS,
                feature_size=24,
                use_checkpoint=False,
            )

            # Load with device mapping
            model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            model.to(device)
            model.eval()
            return model, f"Model loaded successfully (using {device})"
        except Exception as e:
            return None, f"Error loading model: {str(e)}"

    def find_tumor_slices(self, pred_volume):
        """Find slices that contain tumor tissue in each dimension"""
        tumor_slices = {"Axial": [], "Coronal": [], "Sagittal": []}

        # Find tumor slices in each dimension
        for z in range(pred_volume.shape[2]):
            if np.any(pred_volume[:, :, z] > 0):
                tumor_slices["Axial"].append(z)

        for y in range(pred_volume.shape[1]):
            if np.any(pred_volume[:, y, :] > 0):
                tumor_slices["Coronal"].append(y)

        for x in range(pred_volume.shape[0]):
            if np.any(pred_volume[x, :, :] > 0):
                tumor_slices["Sagittal"].append(x)

        return tumor_slices

    def run_segmentation(self, modality_files):
        """
        Run tumor segmentation on the provided modality files

        Args:
            modality_files: Dictionary mapping modality names to file objects

        Returns:
            Dict containing segmentation results or error message
        """
        # Load model
        model, message = self.load_model()
        if model is None:
            return {"success": False, "message": message}

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                paths = []
                modality_paths = {}

                # Save uploaded files to temp directory
                for modality, file in modality_files.items():
                    path = os.path.join(tmpdir, file.name)
                    with open(path, "wb") as out:
                        out.write(file.read())
                    paths.append(path)
                    modality_paths[modality] = path

                # Process based on modality order expected by the model
                ordered_paths = [modality_paths.get(mod) for mod in MODALITY_NAMES]

                # Process images
                images = []
                original_images = []
                original_affines = []

                for i, p in enumerate(ordered_paths):
                    try:
                        # Load original image for reference
                        nib_img = nib.load(p)
                        original_images.append(nib_img.get_fdata())
                        original_affines.append(nib_img.affine)

                        # Process with MONAI transforms
                        img = self.get_transform()(p)
                        images.append(img)
                    except Exception as e:
                        return {
                            "success": False,
                            "message": f"Error processing {os.path.basename(p)}: {str(e)}",
                        }

                # Create input tensor
                image = torch.cat(images, dim=0).unsqueeze(0)  # Shape: (1, 4, H, W, D)

                # Get device
                device = next(model.parameters()).device
                image = image.to(device)

                # Run inference
                with torch.no_grad():
                    output = model(image)
                    pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

                # Check if tumor is present
                tumor_present = np.any(pred > 0)

                # Find tumor slices in each dimension
                tumor_slices = self.find_tumor_slices(pred)

                # Calculate tumor statistics
                tumor_voxels = np.sum(pred > 0)
                total_voxels = pred.size
                tumor_percentage = (tumor_voxels / total_voxels) * 100

                # Create class distribution
                classes = np.unique(pred)
                class_counts = {int(c): np.sum(pred == c) for c in classes}

                # Return results
                return {
                    "success": True,
                    "image": image.cpu(),
                    "prediction": pred,
                    "tumor_present": tumor_present,
                    "tumor_slices": tumor_slices,
                    "tumor_voxels": tumor_voxels,
                    "total_voxels": total_voxels,
                    "tumor_percentage": tumor_percentage,
                    "class_counts": class_counts,
                    "classes": classes,
                }
        except Exception as e:
            return {"success": False, "message": f"Error during segmentation: {str(e)}"}

    def create_slice_visualization(
        self,
        selected_img,
        pred_slice,
        slice_idx,
        view_type,
        modality_idx,
        class_visibility,
        overlay_opacity,
        show_overlay,
    ):
        """
        Create a visualization of a brain slice with tumor overlay

        Args:
            selected_img: Selected modality image
            pred_slice: Prediction slice
            slice_idx: Slice index
            view_type: View type (Axial, Coronal, Sagittal)
            modality_idx: Modality index
            class_visibility: Dictionary of class visibility settings
            overlay_opacity: Opacity of overlay
            show_overlay: Whether to show overlay

        Returns:
            Matplotlib figure
        """
        # Create visualization figure
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(selected_img, cmap="gray")

        # Apply overlay mask if enabled
        if show_overlay:
            # Create masked overlay for each class
            for class_idx in range(1, OUT_CHANNELS):  # Skip background
                if class_idx in class_visibility and class_visibility[class_idx]:
                    mask = pred_slice == class_idx
                    color = CLASS_COLORS[class_idx]
                    colored_mask = np.zeros((*mask.shape, 4))
                    colored_mask[mask] = color
                    colored_mask[..., 3] = colored_mask[..., 3] * overlay_opacity
                    ax.imshow(colored_mask)

        ax.set_title(
            f"{MODALITY_NAMES[modality_idx]} - {view_type} View (Slice {slice_idx})"
        )
        ax.axis("off")
        return fig

    def create_class_distribution_chart(self, classes, class_counts):
        """
        Create a pie chart showing class distribution

        Args:
            classes: Array of class indices
            class_counts: Dictionary of class counts

        Returns:
            Matplotlib figure
        """
        # Prepare data for pie chart
        class_labels = [
            CLASS_NAMES[int(c)] for c in classes if c > 0
        ]  # Exclude background
        class_values = [class_counts[int(c)] for c in classes if c > 0]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [CLASS_COLORS[i][:3] for i in classes if i > 0]  # RGB part of RGBA
        ax.pie(
            class_values,
            labels=class_labels,
            autopct="%1.1f%%",
            colors=colors,
            shadow=True,
            startangle=90,
        )
        ax.axis("equal")
        return fig

    def get_class_data(self, classes, class_counts, total_voxels):
        """
        Get class data for tables

        Args:
            classes: Array of class indices
            class_counts: Dictionary of class counts
            total_voxels: Total number of voxels

        Returns:
            List of class data dictionaries
        """
        class_data = []
        for c in classes:
            if int(c) < len(CLASS_NAMES):
                class_name = CLASS_NAMES[int(c)]
                count = class_counts[int(c)]
                percentage = (count / total_voxels) * 100
                class_data.append(
                    {
                        "Class": class_name,
                        "Voxel Count": f"{count:,}",
                        "Percentage": f"{percentage:.2f}%",
                    }
                )
        return class_data

    def extract_slice(self, volume, slice_idx, view_type):
        """
        Extract a slice from a volume based on view type

        Args:
            volume: 3D volume
            slice_idx: Slice index
            view_type: View type (Axial, Coronal, Sagittal)

        Returns:
            2D slice
        """
        if view_type == "Axial":
            return volume[:, :, slice_idx]
        elif view_type == "Coronal":
            return volume[:, slice_idx, :]
        else:  # Sagittal
            return volume[slice_idx, :, :]
