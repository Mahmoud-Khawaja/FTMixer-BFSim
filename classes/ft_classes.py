"""
Fourier Transform Classes - Proper OOP Encapsulation
Contains: ImageProcessor, ImageViewer, FTMixer
All display and mathematical logic encapsulated in classes.
"""

import numpy as np
from scipy import fft
from PIL import Image
import io
import base64
import threading
from typing import Optional, Tuple, Dict, Any
import plotly.graph_objects as go


class ImageProcessor:
    """Handles image loading, FFT computation, and component extraction."""
    
    def __init__(self):
        self.image: Optional[np.ndarray] = None
        self.fft_result: Optional[np.ndarray] = None
        self.shape: Optional[Tuple[int, int]] = None
        
    def load_image(self, content: str) -> np.ndarray:
        """Load and convert image to grayscale."""
        content_string = content.split(',')[1]
        decoded = base64.b64decode(content_string)
        img = Image.open(io.BytesIO(decoded))
        
        # Convert to grayscale
        if img.mode != 'L':
            img = img.convert('L')
        
        self.image = np.array(img, dtype=np.float64)
        self.shape = self.image.shape
        self.fft_result = None  # Reset FFT
        return self.image
    
    def load_from_array(self, array: np.ndarray) -> np.ndarray:
        """Load image from numpy array."""
        self.image = array.astype(np.float64)
        self.shape = self.image.shape
        self.fft_result = None
        return self.image
    
    def resize_to(self, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize image to target shape."""
        if self.image is None:
            return None
        
        img_pil = Image.fromarray(self.image.astype(np.uint8))
        # target_shape is (height, width)
        img_pil = img_pil.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
        self.image = np.array(img_pil, dtype=np.float64)
        self.shape = self.image.shape
        self.fft_result = None  # Reset FFT after resize
        return self.image
    
    def compute_fft(self) -> np.ndarray:
        """Compute 2D FFT and cache result."""
        if self.image is None:
            return None
        if self.fft_result is None:
            self.fft_result = fft.fftshift(fft.fft2(self.image))
        return self.fft_result
    
    def get_magnitude(self) -> np.ndarray:
        """Get FFT magnitude spectrum."""
        fft_data = self.compute_fft()
        if fft_data is None:
            return None
        return np.abs(fft_data)
    
    def get_phase(self) -> np.ndarray:
        """Get FFT phase spectrum."""
        fft_data = self.compute_fft()
        if fft_data is None:
            return None
        return np.angle(fft_data)
    
    def get_real(self) -> np.ndarray:
        """Get real component of FFT."""
        fft_data = self.compute_fft()
        if fft_data is None:
            return None
        return np.real(fft_data)
    
    def get_imaginary(self) -> np.ndarray:
        """Get imaginary component of FFT."""
        fft_data = self.compute_fft()
        if fft_data is None:
            return None
        return np.imag(fft_data)
    
    @staticmethod
    def adjust_brightness_contrast(image: np.ndarray, brightness: float, 
                                   contrast: float, level: float = 128.0) -> np.ndarray:
        """Apply brightness and contrast adjustment using window/level transform."""
        if image is None:
            return None
        adjusted = (image - level) * contrast + brightness
        return np.clip(adjusted, 0, 255)
    
    @staticmethod
    def normalize_for_display(data: np.ndarray, log_scale: bool = False) -> np.ndarray:
        """Normalize data for display (0-255 range)."""
        if data is None:
            return None
        
        if log_scale:
            # Use log scale for magnitude (better visualization)
            data = np.log1p(np.abs(data))
        
        data_min, data_max = data.min(), data.max()
        if data_max - data_min > 0:
            normalized = 255 * (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        
        return normalized.astype(np.uint8)


class ImageViewer:
    """Encapsulates all display logic for a single image viewer."""
    
    def __init__(self, viewer_id: str, viewer_type: str, colors: dict):
        """
        Initialize image viewer.
        
        Args:
            viewer_id: Unique identifier (e.g., 'input_0', 'output_0')
            viewer_type: 'input' or 'output'
            colors: Color scheme dictionary
        """
        self.viewer_id = viewer_id
        self.viewer_type = viewer_type
        self.colors = colors
        self.processor = ImageProcessor()
        
        # Display state
        self.selected_component = 'magnitude'
        self.brightness = 128.0
        self.contrast = 1.0
    
    def load_image(self, content: str) -> bool:
        """Load image from upload content."""
        try:
            self.processor.load_image(content)
            return True
        except Exception as e:
            print(f"Error loading image: {e}")
            return False
    
    def load_from_array(self, array: np.ndarray) -> bool:
        """Load image from numpy array."""
        try:
            self.processor.load_from_array(array)
            return True
        except Exception as e:
            print(f"Error loading from array: {e}")
            return False
    
    def resize_to(self, target_shape: Tuple[int, int]) -> bool:
        """Resize image to target shape."""
        try:
            self.processor.resize_to(target_shape)
            return True
        except Exception as e:
            print(f"Error resizing image: {e}")
            return False
    
    def has_image(self) -> bool:
        """Check if viewer has an image loaded."""
        return self.processor.image is not None
    
    def get_image_info(self) -> str:
        """Get image dimension info string."""
        if not self.has_image():
            return ""
        h, w = self.processor.shape
        return f"📐 {w} × {h} pixels"
    
    def update_brightness_contrast(self, brightness: float, contrast: float):
        """Update brightness and contrast values."""
        self.brightness = brightness
        self.contrast = contrast
    
    def update_component_selection(self, component: str):
        """Update selected FT component."""
        self.selected_component = component
    
    def get_original_figure(self, title: str = None) -> go.Figure:
        """Get figure for original image."""
        if not self.has_image():
            return self._create_empty_figure("No image loaded")
        
        if title is None:
            title = f"Original ({self.processor.shape[1]}×{self.processor.shape[0]})"
        
        return self._create_image_figure(self.processor.image, title, show_axes=False)
    
    def get_component_figure(self, rect: Optional[Dict] = None, 
                           region_mode: str = 'inner') -> go.Figure:
        """Get figure for selected FT component with adjustments."""
        if not self.has_image():
            return self._create_empty_figure("Upload image first")
        
        # Get component data
        data = self._get_component_data()
        if data is None:
            return self._create_empty_figure("Error processing component")
        
        # Apply brightness/contrast
        data = ImageProcessor.adjust_brightness_contrast(
            data, self.brightness, self.contrast
        )
        
        # Get title
        title = self._get_component_title()
        
        # Create figure
        fig = self._create_image_figure(data, title, show_axes=True)
        
        # Add region rectangle overlay (only for input viewers)
        if self.viewer_type == 'input' and rect is not None:
            self._add_region_overlay(fig, rect, region_mode)
        
        return fig
    
    def _get_component_data(self) -> Optional[np.ndarray]:
        """Get data for selected component."""
        if self.selected_component == 'magnitude':
            data = self.processor.get_magnitude()
            return ImageProcessor.normalize_for_display(data, log_scale=True)
        elif self.selected_component == 'phase':
            data = self.processor.get_phase()
            return ImageProcessor.normalize_for_display(data, log_scale=False)
        elif self.selected_component == 'real':
            data = self.processor.get_real()
            return ImageProcessor.normalize_for_display(data, log_scale=False)
        elif self.selected_component == 'imaginary':
            data = self.processor.get_imaginary()
            return ImageProcessor.normalize_for_display(data, log_scale=False)
        return None
    
    def _get_component_title(self) -> str:
        """Get title for selected component."""
        titles = {
            'magnitude': '🔍 FT Magnitude',
            'phase': '🌀 FT Phase',
            'real': '➕ FT Real',
            'imaginary': '➖ FT Imaginary'
        }
        return titles.get(self.selected_component, 'FT Component')
    
    def _add_region_overlay(self, fig: go.Figure, rect: Dict, region_mode: str):
        """Add region selection rectangle overlay to figure."""
        h, w = self.processor.shape
        
        # Convert normalized coordinates to pixel coordinates
        x0_px = int(rect['x0'] * w)
        y0_px = int(rect['y0'] * h)
        x1_px = int(rect['x1'] * w)
        y1_px = int(rect['y1'] * h)
        
        # Flip y coordinates because image is displayed flipped
        y0_display = h - y1_px
        y1_display = h - y0_px
        
        # Choose color based on region mode
        use_inner = (region_mode == 'inner')
        rect_color = self.colors['primary'] if use_inner else self.colors['error']
        
        # Add rectangle shape
        fig.add_shape(
            type="rect",
            x0=x0_px, y0=y0_display,
            x1=x1_px, y1=y1_display,
            line=dict(color=rect_color, width=2),
            fillcolor=rect_color,
            opacity=0.25,
            layer='above'
        )
        
        # Add label
        label_text = "LOW FREQ" if use_inner else "HIGH FREQ"
        fig.add_annotation(
            x=(x0_px + x1_px) / 2,
            y=y0_display - 10,
            text=f"<b>{label_text}</b>",
            showarrow=False,
            font=dict(size=10, color=rect_color, family="Courier New, monospace"),
            bgcolor='rgba(15, 23, 42, 0.8)',
            borderpad=4
        )
    
    def _create_empty_figure(self, text: str) -> go.Figure:
        """Create an empty placeholder figure."""
        fig = go.Figure()
        fig.add_annotation(
            text=text,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color=self.colors['text_secondary'])
        )
        fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor=self.colors['surface'],
            plot_bgcolor=self.colors['surface']
        )
        return fig
    
    def _create_image_figure(self, image: np.ndarray, title: str, 
                            show_axes: bool = False) -> go.Figure:
        """Create Plotly figure from image array."""
        fig = go.Figure(data=go.Heatmap(
            z=image[::-1],  # Flip vertically for correct orientation
            colorscale='gray',
            showscale=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(color=self.colors['text'], size=12)),
            xaxis=dict(visible=show_axes, showgrid=False, zeroline=False),
            yaxis=dict(visible=show_axes, showgrid=False, zeroline=False, scaleanchor='x'),
            margin=dict(l=0, r=0, t=30 if title else 0, b=0),
            paper_bgcolor=self.colors['surface'],
            plot_bgcolor=self.colors['surface'],
            dragmode='pan'
        )
        
        return fig


class FTMixer:
    """Handles weighted mixing of Fourier Transform components with region selection."""
    
    def __init__(self):
        self.cancel_flag = threading.Event()
    
    def create_region_mask(self, shape: Tuple[int, int], rect: Dict[str, float], 
                          use_inner: bool) -> np.ndarray:
        """Create binary mask for region selection."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.float64)
        
        # Convert normalized coordinates to pixels
        x0 = int(rect['x0'] * w)
        y0 = int(rect['y0'] * h)
        x1 = int(rect['x1'] * w)
        y1 = int(rect['y1'] * h)
        
        # Ensure valid bounds
        x0, x1 = max(0, min(x0, x1)), min(w, max(x0, x1))
        y0, y1 = max(0, min(y0, y1)), min(h, max(y0, y1))
        
        if use_inner:
            mask[y0:y1, x0:x1] = 1.0
        else:
            mask[:, :] = 1.0
            mask[y0:y1, x0:x1] = 0.0
        
        return mask
    
    def mix_components(self, processors: list, weights: list, mode: str,
                      rect: Optional[Dict] = None, use_inner: bool = True) -> np.ndarray:
        """
        Mix FFT components from multiple ImageProcessor objects.
        
        Args:
            processors: List of ImageProcessor objects
            weights: List of weights for each processor
            mode: 'mag_phase' or 'real_imag'
            rect: Rectangle coordinates for region selection
            use_inner: If True, use inner region; else use outer
        
        Returns:
            Mixed image (inverse FFT result)
        """
        # Get valid processors and weights
        valid_data = [(p, w) for p, w in zip(processors, weights) 
                      if p is not None and p.image is not None]
        
        if not valid_data:
            return None
        
        # Get reference shape
        ref_shape = valid_data[0][0].shape
        
        # Mix based on mode
        if mode == 'mag_phase':
            mixed_fft = self._mix_magnitude_phase(valid_data, ref_shape)
        else:  # real_imag
            mixed_fft = self._mix_real_imaginary(valid_data, ref_shape)
        
        # Apply region selection if specified
        if rect is not None and len(valid_data) > 0:
            mask = self.create_region_mask(ref_shape, rect, use_inner)
            # Use first image as base for non-selected region
            base_fft = valid_data[0][0].compute_fft()
            mixed_fft = mask * mixed_fft + (1 - mask) * base_fft
        
        # Check cancellation before expensive iFFT
        if self.cancel_flag.is_set():
            return None
        
        # Inverse FFT
        result = fft.ifft2(fft.ifftshift(mixed_fft))
        result = np.real(result)
        result = np.clip(result, 0, 255)
        
        return result.astype(np.uint8)
    
    def _mix_magnitude_phase(self, valid_data: list, shape: Tuple[int, int]) -> np.ndarray:
        """Mix using magnitude and phase components."""
        mixed_magnitude = np.zeros(shape, dtype=np.float64)
        mixed_phase = np.zeros(shape, dtype=np.float64)
        total_weight = 0
        
        for processor, weight in valid_data:
            if self.cancel_flag.is_set():
                return None
            
            magnitude = processor.get_magnitude()
            phase = processor.get_phase()
            
            mixed_magnitude += weight * magnitude
            mixed_phase += weight * phase
            total_weight += weight
        
        if total_weight > 0:
            mixed_magnitude /= total_weight
            mixed_phase /= total_weight
        
        # Reconstruct complex FFT
        mixed_fft = mixed_magnitude * np.exp(1j * mixed_phase)
        return mixed_fft
    
    def _mix_real_imaginary(self, valid_data: list, shape: Tuple[int, int]) -> np.ndarray:
        """Mix using real and imaginary components."""
        mixed_real = np.zeros(shape, dtype=np.float64)
        mixed_imag = np.zeros(shape, dtype=np.float64)
        total_weight = 0
        
        for processor, weight in valid_data:
            if self.cancel_flag.is_set():
                return None
            
            real_part = processor.get_real()
            imag_part = processor.get_imaginary()
            
            mixed_real += weight * real_part
            mixed_imag += weight * imag_part
            total_weight += weight
        
        if total_weight > 0:
            mixed_real /= total_weight
            mixed_imag /= total_weight
        
        # Reconstruct complex FFT
        mixed_fft = mixed_real + 1j * mixed_imag
        return mixed_fft
    
    def cancel(self):
        """Cancel current mixing operation."""
        self.cancel_flag.set()
    
    def reset_cancel(self):
        """Reset cancellation flag for new operation."""
        self.cancel_flag.clear()