"""
Image upload components for multi-image support.
"""

import streamlit as st
from pathlib import Path
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
from PIL import Image
import io

from app.services.file_handler import save_uploaded_file, validate_image
from app.services.session_manager import get_state, set_state, update_state
from utils.config import config


def render_multi_image_upload_zone() -> List[Any]:
    """
    Render multi-image drag-and-drop upload zone.
    
    Returns:
        List of uploaded files (Streamlit UploadedFile objects)
    """
    st.subheader("📤 Upload Images")
    st.markdown("Upload one or multiple images for batch inspection")
    
    uploaded_files = st.file_uploader(
        "Choose image files",
        type=config.allowed_extensions_list,
        accept_multiple_files=True,
        help=f"Maximum file size: {config.max_file_size_mb}MB per file. You can select multiple files.",
        label_visibility="collapsed"
    )
    
    return uploaded_files if uploaded_files else []


def render_image_preview_card(image_id: str, filename: str, filepath: Path, status: str = "uploaded"):
    """
    Render a preview card for a single image in the gallery.
    
    Args:
        image_id: Unique image identifier
        filename: Original filename
        filepath: Path to saved image
        status: Current status (uploaded, processing, complete, failed)
    """
    col1, col2, col3 = st.columns([3, 2, 1])
    
    with col1:
        # Try to load and display thumbnail
        try:
            if filepath.exists():
                img = Image.open(filepath)
                # Resize for thumbnail (max 150px width)
                img.thumbnail((150, 150))
                st.image(img, width='content')
        except Exception:
            st.info("📷 Image")
    
    with col2:
        st.markdown(f"**{filename[:30]}...**" if len(filename) > 30 else f"**{filename}**")
        status_colors = {
            "uploaded": "🔵",
            "processing": "🟡",
            "complete": "🟢",
            "failed": "🔴"
        }
        emoji = status_colors.get(status, "⚪")
        st.caption(f"{emoji} {status.title()}")
    
    with col3:
        if status == "uploaded":
            if st.button("🗑️", key=f"remove_{image_id}", help="Remove image"):
                # Remove from uploaded_images list
                uploaded_images = get_state("uploaded_images", [])
                uploaded_images = [img for img in uploaded_images if img.get("image_id") != image_id]
                set_state("uploaded_images", uploaded_images)
                st.rerun()


def render_image_preview_gallery():
    """Render gallery of uploaded images with thumbnails."""
    uploaded_images = get_state("uploaded_images", [])
    
    if not uploaded_images:
        st.info("📷 No images uploaded yet. Use the upload zone above.")
        return
    
    st.subheader(f"📷 Uploaded Images ({len(uploaded_images)})")
    
    # Display in a grid (3 columns)
    cols = st.columns(3)
    for idx, img_info in enumerate(uploaded_images):
        with cols[idx % 3]:
            render_image_preview_card(
                img_info.get("image_id"),
                img_info.get("filename"),
                Path(img_info.get("filepath")),
                img_info.get("status", "uploaded")
            )


def render_batch_config_form() -> Dict[str, Any]:
    """
    Render session-level configuration form.
    
    Returns:
        Dictionary with configuration (criticality, domain, user_notes, batch_name)
    """
    st.subheader("⚙️ Session Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        criticality = st.selectbox(
            "🎚️ Criticality Level",
            options=["low", "medium", "high"],
            index=1,
            help="Safety criticality level for all images in this session"
        )
        
        batch_name = st.text_input(
            "📦 Batch Name (optional)",
            placeholder="e.g., Batch-001, Production-Run-2024",
            help="Optional name to group this inspection session"
        )
    
    with col2:
        domain = st.text_input(
            "🏷️ Domain (optional)",
            placeholder="e.g., mechanical_fasteners",
            help="Provide context about the inspection domain"
        )
    
    user_notes = st.text_area(
        "📝 Additional Notes (optional)",
        placeholder="Any specific concerns or context about this inspection session...",
        height=100
    )
    
    return {
        "criticality": criticality,
        "domain": domain or None,
        "user_notes": user_notes or None,
        "batch_name": batch_name or None
    }


def process_uploaded_files(uploaded_files: List[Any], metadata: Dict[str, Any]):
    """
    Process uploaded files and add them to session state.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
        metadata: Session metadata (criticality, domain, etc.)
    """
    if not uploaded_files:
        return
    
    # Initialize session if needed
    if not get_state("current_session_id"):
        set_state("current_session_id", str(uuid.uuid4())[:8])
        set_state("session_start_time", datetime.now())
        set_state("session_status", "uploading")
    
    # Get existing uploaded images
    uploaded_images = get_state("uploaded_images", [])
    
    # Process each uploaded file
    for uploaded_file in uploaded_files:
        # Validate
        is_valid, error_msg = validate_image(uploaded_file)
        
        if not is_valid:
            st.error(f"❌ {uploaded_file.name}: {error_msg}")
            continue
        
        # Save file
        filepath = save_uploaded_file(uploaded_file)
        
        if filepath:
            image_id = str(uuid.uuid4())
            
            # Create thumbnail (optional - for now just store path)
            thumbnail_path = filepath  # TODO: Create actual thumbnail
            
            # Check if this file already exists (avoid duplicates)
            existing = any(
                Path(img.get("filepath", "")).samefile(filepath) if Path(img.get("filepath", "")).exists() else False
                for img in uploaded_images
            )
            
            if not existing:
                image_info = {
                    "image_id": image_id,
                    "filepath": str(filepath),
                    "filename": uploaded_file.name,
                    "upload_time": datetime.now().isoformat(),
                    "thumbnail_path": str(thumbnail_path),
                    "status": "uploaded",
                    "inspection_result": None
                }
                
                uploaded_images.append(image_info)
    
    # Update session state
    set_state("uploaded_images", uploaded_images)
    
    # Store metadata
    set_state("session_metadata", metadata)
    
    # Update session status
    if uploaded_images:
        set_state("session_status", "uploading")
