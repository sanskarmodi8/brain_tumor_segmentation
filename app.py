import streamlit as st

from src.BrainTumorSegmentation.pipeline.prediction import Prediction

# constants
MODALITY_NAMES = ["T1n", "T1c", "T2w", "T2f"]
CLASS_NAMES = ["Background", "Necrotic core", "Edema", "Enhancing tumor"]
CLASS_COLORS = [
    (0, 0, 0, 0),
    (1, 0, 0, 0.7),
    (0, 1, 0, 0.7),
    (0, 0, 1, 0.7),
]  # RGBA format for overlays

# -------------------- STREAMLIT UI --------------------
st.set_page_config(layout="wide", page_title="Brain Tumor Segmentation")
st.title("🧠 Brain Tumor Segmentation")
st.write("")
st.write("Upload MRI scans to detect and visualize brain tumors")
st.markdown("---")


@st.cache_resource
def initiate_prediction():
    prediction = Prediction()
    model, _ = prediction.load_model()
    if model is None:
        print("none model")
    transform = prediction.get_transform()
    return prediction, model, transform


# Initialize sidebar
st.sidebar.title("Instructions")

# Model initialization in sidebar
with st.sidebar:
    with st.spinner("Initializing model..."):
        prediction, model, transform = initiate_prediction()

    st.markdown(
        """
    1. Upload all 4 modality MRI scans
    2. Ensure files are in NIfTI format (.nii or .nii.gz)
    3. Required modalities:
       - T1n: T1-weighted without contrast
       - T1c: T1-weighted with contrast
       - T2w: T2-weighted 
       - T2f: T2-FLAIR
    """
    )

# File upload section
st.write("### Upload MRI Scans")
st.write("")
st.write("Please upload all 4 MRI modality files (T1n, T1c, T2w, T2f)")
st.write("")

# Map from modality name to file
modality_files = {}

# Create file uploaders for each modality
col1, col2 = st.columns(2)
with col1:
    t1n_file = st.file_uploader(
        "T1n (T1-weighted without contrast)",
        type=None,
        accept_multiple_files=False,
        help="Accept .nii or .nii.gz files",
    )
    if t1n_file is not None and prediction.is_valid_nifti(t1n_file):
        modality_files["T1n"] = t1n_file

    t2w_file = st.file_uploader(
        "T2w (T2-weighted)",
        type=None,
        accept_multiple_files=False,
        help="Accept .nii or .nii.gz files",
    )
    if t2w_file is not None and prediction.is_valid_nifti(t2w_file):
        modality_files["T2w"] = t2w_file

with col2:
    t1c_file = st.file_uploader(
        "T1c (T1-weighted with contrast)",
        type=None,
        accept_multiple_files=False,
        help="Accept .nii or .nii.gz files",
    )
    if t1c_file is not None and prediction.is_valid_nifti(t1c_file):
        modality_files["T1c"] = t1c_file

    t2f_file = st.file_uploader(
        "T2f (T2-FLAIR)",
        type=None,
        accept_multiple_files=False,
        help="Accept .nii or .nii.gz files",
    )
    if t2f_file is not None and prediction.is_valid_nifti(t2f_file):
        modality_files["T2f"] = t2f_file

# Show validation messages for uploaded files
for modality, file in list(modality_files.items()):
    if not prediction.is_valid_nifti(file):
        st.warning(
            f"File for {modality} is not a valid NIfTI file. Please upload a .nii or .nii.gz file."
        )
        del modality_files[modality]

# Check if we have all required modalities
if len(modality_files) == 4 and model is not None:
    st.success("All required files uploaded!")

    # Process files
    with st.spinner("Processing files..."):
        # Run segmentation
        results = prediction.run_segmentation(modality_files)

        if results["success"]:
            # Extract results
            image = results["image"]
            pred = results["prediction"]
            tumor_present = results["tumor_present"]
            tumor_slices = results["tumor_slices"]
            tumor_voxels = results["tumor_voxels"]
            total_voxels = results["total_voxels"]
            tumor_percentage = results["tumor_percentage"]
            class_counts = results["class_counts"]
            classes = results["classes"]

            # Create visualization
            if pred.shape[2] > 0:  # Make sure there's a valid Z dimension
                # Create tabs for different views and modalities
                view_tab, stats_tab = st.tabs(
                    ["Image Visualization", "Tumor Statistics"]
                )

                with view_tab:
                    # Add selector for modality
                    modality_idx = st.selectbox(
                        "Select MRI Modality",
                        range(len(MODALITY_NAMES)),
                        format_func=lambda i: MODALITY_NAMES[i],
                    )

                    # Get selected modality image
                    selected_img = image.squeeze(0)[modality_idx].numpy()

                    # Create columns for the view selection and controls
                    view_col, control_col = st.columns([3, 1])

                    with control_col:
                        st.markdown("### View Controls")
                        view_type = st.radio(
                            "Select View", ["Axial", "Coronal", "Sagittal"]
                        )
                        show_overlay = st.checkbox("Show Tumor Overlay", value=True)
                        overlay_opacity = st.slider("Overlay Opacity", 0.0, 1.0, 0.5)

                        # Get class visibility toggles
                        st.markdown("### Classes")
                        class_visibility = {}
                        for i, class_name in enumerate(CLASS_NAMES):
                            if i > 0:  # Skip background
                                class_visibility[i] = st.checkbox(
                                    f"{class_name}", value=True
                                )

                        # Display tumor slice information
                        if tumor_present:
                            st.markdown("### Tumor Slices")
                            st.write("Tumor detected in:")
                            if tumor_slices[view_type]:
                                st.write(
                                    f"- {view_type}: {len(tumor_slices[view_type])} slices"
                                )
                                if len(tumor_slices[view_type]) <= 10:
                                    st.write(
                                        f"  Slices: {', '.join(map(str, tumor_slices[view_type]))}"
                                    )
                                else:
                                    st.write(
                                        f"  Range: {min(tumor_slices[view_type])} - {max(tumor_slices[view_type])}"
                                    )

                                # Add a selector for tumor slices
                                if len(tumor_slices[view_type]) > 0:
                                    st.markdown("**Jump to tumor slice:**")
                                    tumor_slice_idx = st.selectbox(
                                        "Select tumor slice",
                                        tumor_slices[view_type],
                                        format_func=lambda i: f"Slice {i}",
                                    )
                        else:
                            st.info("No tumor detected in this scan.")

                    with view_col:
                        # If tumor is present and there are tumor slices in this view,
                        # default to showing a tumor slice instead of middle slice
                        default_slice = None
                        if tumor_present and tumor_slices[view_type]:
                            default_slice = tumor_slices[view_type][
                                len(tumor_slices[view_type]) // 2
                            ]

                        if view_type == "Axial":
                            if default_slice is None:
                                default_slice = selected_img.shape[2] // 2
                            slice_idx = st.slider(
                                "Axial Slice",
                                0,
                                selected_img.shape[2] - 1,
                                default_slice,
                            )
                            # Add indicator if tumor is present in this slice
                            if slice_idx in tumor_slices["Axial"]:
                                st.success("⚠️ Tumor present in this slice")
                        elif view_type == "Coronal":
                            if default_slice is None:
                                default_slice = selected_img.shape[1] // 2
                            slice_idx = st.slider(
                                "Coronal Slice",
                                0,
                                selected_img.shape[1] - 1,
                                default_slice,
                            )
                            # Add indicator if tumor is present in this slice
                            if slice_idx in tumor_slices["Coronal"]:
                                st.success("⚠️ Tumor present in this slice")
                        else:  # Sagittal
                            if default_slice is None:
                                default_slice = selected_img.shape[0] // 2
                            slice_idx = st.slider(
                                "Sagittal Slice",
                                0,
                                selected_img.shape[0] - 1,
                                default_slice,
                            )
                            # Add indicator if tumor is present in this slice
                            if slice_idx in tumor_slices["Sagittal"]:
                                st.success("⚠️ Tumor present in this slice")

                        # Extract the appropriate slice
                        img_slice = prediction.extract_slice(
                            selected_img, slice_idx, view_type
                        )
                        pred_slice = prediction.extract_slice(
                            pred, slice_idx, view_type
                        )

                        # Create visualization figure
                        fig = prediction.create_slice_visualization(
                            img_slice,
                            pred_slice,
                            slice_idx,
                            view_type,
                            modality_idx,
                            class_visibility,
                            overlay_opacity,
                            show_overlay,
                        )
                        st.pyplot(fig)

                with stats_tab:
                    st.subheader("Tumor Statistics")

                    # Only show detailed results if tumor is present
                    if tumor_present:
                        # Display statistics
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Total Tumor Voxels", f"{tumor_voxels:,}")
                            st.metric(
                                "Tumor Volume Percentage",
                                f"{tumor_percentage:.2f}%",
                            )

                        with col2:
                            # Calculate approximate volume in cm³ (assuming 1mm³ voxels)
                            # This is a rough estimate - would need actual voxel dimensions for accuracy
                            voxel_volume_mm3 = 1.0  # Placeholder
                            tumor_volume_mm3 = tumor_voxels * voxel_volume_mm3
                            tumor_volume_cm3 = tumor_volume_mm3 / 1000.0

                            st.metric(
                                "Estimated Tumor Volume",
                                f"{tumor_volume_cm3:.2f} cm³",
                            )

                        # Display class distribution chart
                        st.subheader("Tumor Class Distribution")
                        fig = prediction.create_class_distribution_chart(
                            classes, class_counts
                        )
                        st.pyplot(fig)

                        # Detailed class breakdown table
                        st.subheader("Detailed Class Breakdown")
                        class_data = prediction.get_class_data(
                            classes, class_counts, total_voxels
                        )
                        st.table(class_data)

                        # Tumor location summary
                        st.subheader("Tumor Location")
                        st.write("Tumor slices by view:")

                        for view_name, slices in tumor_slices.items():
                            if slices:
                                if len(slices) <= 10:
                                    st.write(
                                        f"- **{view_name}**: Slices {', '.join(map(str, slices))}"
                                    )
                                else:
                                    st.write(
                                        f"- **{view_name}**: {len(slices)} slices, range {min(slices)}-{max(slices)}"
                                    )
                    else:
                        st.info("No tumor detected in the scan.")
            else:
                st.error("Invalid image dimensions detected")
        else:
            st.error(results["message"])

elif len(modality_files) > 0:
    missing = set(MODALITY_NAMES) - set(modality_files.keys())
    if missing:
        st.warning(f"Missing required modalities: {', '.join(missing)}")


# Add a footer
st.write("")
st.markdown("---")

st.write("")
# Add explanations at the bottom
with st.expander("About this app"):
    st.write(
        """
    ## Brain Tumor Segmentation App
    
    This application performs automatic brain tumor segmentation using deep learning. It analyzes MRI scans and identifies different types of tumor tissue.
    
    ### How it works
    
    The app uses a Swin UNETR deep learning model, which is a state-of-the-art architecture for medical image segmentation. The model has been trained on the BraTS (Brain Tumor Segmentation) challenge dataset.
    
    ### Required inputs
    
    Four MRI modalities are required for accurate segmentation:
    
    - **T1n**: T1-weighted MRI without contrast
    - **T1c**: T1-weighted MRI with contrast enhancement
    - **T2w**: T2-weighted MRI
    - **T2f**: T2-FLAIR (Fluid Attenuated Inversion Recovery)
    
    ### Output classes
    
    The model segments brain tumors into multiple tissue classes:
    
    - **Necrotic core**: Areas of dead tumor tissue
    - **Edema**: Swelling around the tumor
    - **Enhancing tumor**: Active tumor regions that enhance with contrast
    
    ### Disclaimer
    
    This is a research tool and should not be used for clinical diagnosis. Always consult with a qualified medical professional.
    """
    )
st.markdown(
    "Made by **Sanskar Modi** • Report any [issues](https://github.com/sanskarmodi8/brain_tumor_segmentation/issues)"
)
