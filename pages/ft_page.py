"""
FT Magnitude/Phase Mixer - Complete Version with All Features
A Fourier Transform image mixing application with:
- 4 input image viewers with FT component display
- Weighted mixing of magnitude/phase or real/imaginary components
- Region-based frequency selection with VISUAL RECTANGLE OVERLAY ✅
- Real-time mixing with threading and cancellation
- Brightness/contrast adjustment via MOUSE DRAG and sliders ✅
- Interactive region size control with immediate visual feedback ✅
- ✅ FIXED: Click-to-upload now works seamlessly!
- ✅ NEW: Mouse drag for brightness/contrast control!
"""

from dash import dcc, html, callback, Output, Input, State, ALL, MATCH, ctx
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go
import numpy as np
from scipy import fft
from PIL import Image
import io
import base64
import threading
from typing import Optional, Tuple, Dict, Any
import dash

# ═══════════════════════════════════════════════════════════════════════════
# COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════════════

COLORS = {
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'secondary': '#8b5cf6',
    'background': '#0f172a',
    'surface': '#1e293b',
    'surface_light': '#334155',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'border': 'rgba(99, 102, 241, 0.2)',
    'success': '#10b981',
    'error': '#ef4444',
    'warning': '#f59e0b'
}

# ═══════════════════════════════════════════════════════════════════════════
# IMAGE PROCESSOR CLASS
# ═══════════════════════════════════════════════════════════════════════════

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

# ═══════════════════════════════════════════════════════════════════════════
# FT MIXER CLASS
# ═══════════════════════════════════════════════════════════════════════════

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
        Mix FFT components from multiple images.
        
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

# ═══════════════════════════════════════════════════════════════════════════
# GLOBAL STATE
# ═══════════════════════════════════════════════════════════════════════════

# Image processors for 4 inputs + 2 outputs
image_processors = {f'input_{i}': ImageProcessor() for i in range(4)}
image_processors.update({f'output_{i}': ImageProcessor() for i in range(2)})

# FT Mixer instance
ft_mixer = FTMixer()

# Threading for async mixing
mixing_thread = None
mixing_lock = threading.Lock()

# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS FOR UI
# ═══════════════════════════════════════════════════════════════════════════

def create_empty_figure(text: str = "No image loaded"):
    """Create an empty placeholder figure."""
    fig = go.Figure()
    fig.add_annotation(
        text=text,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=14, color=COLORS['text_secondary'])
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface']
    )
    return fig

def create_image_figure(image: np.ndarray, title: str = "", show_axes: bool = False):
    """Create Plotly figure from image array."""
    fig = go.Figure(data=go.Heatmap(
        z=image[::-1],  # Flip vertically for correct orientation
        colorscale='gray',
        showscale=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(color=COLORS['text'], size=12)),
        xaxis=dict(visible=show_axes, showgrid=False, zeroline=False),
        yaxis=dict(visible=show_axes, showgrid=False, zeroline=False, scaleanchor='x'),
        margin=dict(l=0, r=0, t=30 if title else 0, b=0),
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        dragmode='pan'  # Enable drag mode for mouse interaction
    )
    
    return fig

def create_image_viewer(viewer_id: str, title: str, show_component_selector: bool = True):
    """✅ FIXED: Create viewer with working click-to-upload and mouse drag B/C."""
    components = [
        html.H6(title, style={
            'textAlign': 'center',
            'marginBottom': '1rem',
            'color': COLORS['text'],
            'fontWeight': '600'
        })
    ]
    
    # Only add upload for input viewers
    if viewer_id.startswith('input_'):
        # ✅ FIXED: Visible upload area that becomes the image viewer
        components.append(
            html.Div([
                # Upload component (visible as clickable area)
                dcc.Upload(
                    id={'type': 'upload', 'index': viewer_id},
                    children=html.Div([
                        html.Div(id={'type': 'upload-placeholder', 'index': viewer_id},
                                children=[
                                    html.Div('📤', style={'fontSize': '3rem', 'marginBottom': '0.5rem'}),
                                    html.Div('Click to Upload Image', style={'fontSize': '1rem', 'fontWeight': '500'}),
                                    html.Div('or drag and drop', style={'fontSize': '0.85rem', 'color': COLORS['text_secondary'], 'marginTop': '0.25rem'})
                                ],
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'height': '100%'
                                }),
                        # Image will be shown here after upload
                        html.Div(id={'type': 'image-container', 'index': viewer_id},
                                style={'display': 'none', 'height': '100%'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '250px',
                        'lineHeight': '250px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '0.75rem',
                        'textAlign': 'center',
                        'backgroundColor': COLORS['surface'],
                        'borderColor': COLORS['border'],
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease',
                        'position': 'relative'
                    },
                    multiple=False
                ),
                # Image info
                html.Div(id={'type': 'image-info', 'index': viewer_id}, 
                        style={
                            'textAlign': 'center',
                            'color': COLORS['text_secondary'],
                            'fontSize': '0.85rem',
                            'marginTop': '0.5rem'
                        })
            ])
        )
    else:
        # Output viewers - traditional approach
        components.append(
            dcc.Graph(
                id={'type': 'graph-original', 'index': viewer_id},
                config={'displayModeBar': False, 'scrollZoom': False},
                figure=create_empty_figure("No result yet"),
                style={
                    'height': '400px',
                    'backgroundColor': COLORS['surface'],
                    'borderRadius': '0.75rem'
                }
            )
        )
    
    if show_component_selector:
        # Component selector dropdown
        components.extend([
            dcc.Dropdown(
                id={'type': 'component-selector', 'index': viewer_id},
                options=[
                    {'label': '🔍 FT Magnitude', 'value': 'magnitude'},
                    {'label': '🌀 FT Phase', 'value': 'phase'},
                    {'label': '➕ FT Real', 'value': 'real'},
                    {'label': '➖ FT Imaginary', 'value': 'imaginary'}
                ],
                value='magnitude',
                clearable=False,
                style={
                    'marginTop': '0.75rem',
                    'marginBottom': '0.5rem',
                    'backgroundColor': COLORS['surface'],
                    'color': COLORS['text']
                }
            ),
            dcc.Graph(
                id={'type': 'graph-component', 'index': viewer_id},
                config={
                    'displayModeBar': False,
                    'scrollZoom': False,
                    'doubleClick': False
                },
                figure=create_empty_figure("Upload image first"),
                style={
                    'height': '250px',
                    'backgroundColor': COLORS['surface'],
                    'borderRadius': '0.75rem',
                    'cursor': 'crosshair'  # Show drag cursor
                }
            )
        ])
        
        # ✅ Mouse drag instructions
        components.append(
            html.Div("🖱️ Drag on image: ↕️ Brightness | ↔️ Contrast", style={
                'textAlign': 'center',
                'color': COLORS['text_secondary'],
                'fontSize': '0.75rem',
                'marginTop': '0.5rem',
                'fontStyle': 'italic'
            })
        )
        
        # ✅ Brightness and Contrast Sliders
        components.extend([
            html.Div([
                html.Label("🔆 Brightness", style={
                    'fontSize': '0.85rem',
                    'color': COLORS['text_secondary'],
                    'marginBottom': '0.25rem',
                    'display': 'block'
                }),
                dcc.Slider(
                    id={'type': 'brightness-slider', 'index': viewer_id},
                    min=-128, max=128, step=8, value=0,
                    marks={-128: '-128', 0: '0', 128: '128'},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={'marginTop': '0.75rem'}),
            
            html.Div([
                html.Label("🎨 Contrast", style={
                    'fontSize': '0.85rem',
                    'color': COLORS['text_secondary'],
                    'marginBottom': '0.25rem',
                    'display': 'block'
                }),
                dcc.Slider(
                    id={'type': 'contrast-slider', 'index': viewer_id},
                    min=0.1, max=3.0, step=0.1, value=1.0,
                    marks={0.1: '0.1', 1.0: '1.0', 3.0: '3.0'},
                    tooltip={"placement": "bottom", "always_visible": False}
                )
            ], style={'marginTop': '0.5rem', 'marginBottom': '0.5rem'})
        ])
    
    # Store for brightness/contrast state
    components.append(dcc.Store(id={'type': 'bc-state', 'index': viewer_id}, 
                               data={'brightness': 128, 'contrast': 1.0}))
    
    # ✅ NEW: Store for mouse drag tracking
    components.append(dcc.Store(id={'type': 'mouse-drag-store', 'index': viewer_id}, 
                               data={'x': 0, 'y': 0, 'dragging': False}))
    
    return html.Div(components, style={
        'padding': '1rem',
        'backgroundColor': COLORS['surface'],
        'borderRadius': '1rem',
        'border': f'1px solid {COLORS["border"]}',
        'margin': '0.5rem'
    })

# ═══════════════════════════════════════════════════════════════════════════
# PAGE LAYOUT
# ═══════════════════════════════════════════════════════════════════════════

layout = html.Div([
    # Back to home button
    html.Div([
        dcc.Link(
            '← Back to Home',
            href='/',
            style={
                'color': COLORS['primary'],
                'textDecoration': 'none',
                'fontSize': '1rem',
                'fontWeight': '500',
                'transition': 'color 0.3s ease'
            }
        )
    ], style={'marginBottom': '2rem'}),
    
    # Header
    html.H1("FT Magnitude/Phase Mixer", style={
        'textAlign': 'center',
        'fontSize': '2.5rem',
        'fontWeight': '700',
        'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
        'WebkitBackgroundClip': 'text',
        'WebkitTextFillColor': 'transparent',
        'marginBottom': '1rem'
    }),
    
    html.P("✨ Click to upload | Drag on image for B/C | Visual region selection", style={
        'textAlign': 'center',
        'color': COLORS['text_secondary'],
        'fontSize': '1.1rem',
        'marginBottom': '3rem'
    }),
    
    # Input Images Section
    html.Div([
        html.H3("Input Images", style={
            'color': COLORS['text'],
            'fontSize': '1.5rem',
            'fontWeight': '600',
            'marginBottom': '1.5rem'
        }),
        html.Div([
            html.Div([
                create_image_viewer(f'input_{i}', f'Input {i+1}', True)
            ], style={'flex': '1', 'minWidth': '250px'})
            for i in range(4)
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '1rem',
            'marginBottom': '2rem'
        })
    ]),
    
    html.Hr(style={'borderColor': COLORS['border'], 'margin': '2rem 0'}),
    
    # Control Panel
    html.Div([
        html.H3("Mixer Controls", style={
            'color': COLORS['text'],
            'fontSize': '1.5rem',
            'fontWeight': '600',
            'marginBottom': '1.5rem'
        }),
        
        # Weights Section
        html.Div([
            html.Label("Image Weights", style={
                'color': COLORS['text'],
                'fontSize': '1.1rem',
                'fontWeight': '500',
                'marginBottom': '1rem',
                'display': 'block'
            }),
            html.Div([
                html.Div([
                    html.Label(f"Weight {i+1}", style={
                        'fontSize': '0.9rem',
                        'color': COLORS['text_secondary'],
                        'marginBottom': '0.5rem',
                        'display': 'block'
                    }),
                    dcc.Slider(
                        id={'type': 'weight-slider', 'index': i},
                        min=0, max=1, step=0.01, value=0.25,
                        marks={0: {'label': '0', 'style': {'color': COLORS['text_secondary']}},
                               0.5: {'label': '0.5', 'style': {'color': COLORS['text_secondary']}},
                               1: {'label': '1', 'style': {'color': COLORS['text_secondary']}}},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'flex': '1', 'minWidth': '200px', 'padding': '0 1rem'})
                for i in range(4)
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '1rem'})
        ], style={
            'backgroundColor': COLORS['surface'],
            'padding': '1.5rem',
            'borderRadius': '1rem',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '1.5rem'
        }),
        
        # Mode Controls
        html.Div([
            html.Div([
                html.Label("Mixing Mode", style={
                    'color': COLORS['text'],
                    'fontSize': '1rem',
                    'fontWeight': '500',
                    'marginBottom': '0.75rem',
                    'display': 'block'
                }),
                dcc.RadioItems(
                    id='mixing-mode',
                    options=[
                        {'label': ' 📊 Magnitude + Phase', 'value': 'mag_phase'},
                        {'label': ' 🔢 Real + Imaginary', 'value': 'real_imag'}
                    ],
                    value='mag_phase',
                    inline=False,
                    style={'color': COLORS['text']},
                    labelStyle={'display': 'block', 'marginBottom': '0.5rem'}
                )
            ], style={'flex': '1', 'minWidth': '250px'}),
            
            html.Div([
                html.Label("Region Selection", style={
                    'color': COLORS['text'],
                    'fontSize': '1rem',
                    'fontWeight': '500',
                    'marginBottom': '0.75rem',
                    'display': 'block'
                }),
                dcc.RadioItems(
                    id='region-mode',
                    options=[
                        {'label': ' 🎯 Inner (Low Freq)', 'value': 'inner'},
                        {'label': ' 🌊 Outer (High Freq)', 'value': 'outer'}
                    ],
                    value='inner',
                    inline=False,
                    style={'color': COLORS['text']},
                    labelStyle={'display': 'block', 'marginBottom': '0.5rem'}
                )
            ], style={'flex': '1', 'minWidth': '250px'}),
            
            html.Div([
                html.Label("Region Size (%)", style={
                    'color': COLORS['text'],
                    'fontSize': '1rem',
                    'fontWeight': '500',
                    'marginBottom': '0.75rem',
                    'display': 'block'
                }),
                dcc.Slider(
                    id='region-size-slider',
                    min=10, max=100, step=5, value=30,
                    marks={10: {'label': '10%', 'style': {'color': COLORS['text_secondary']}},
                           50: {'label': '50%', 'style': {'color': COLORS['text_secondary']}},
                           100: {'label': '100%', 'style': {'color': COLORS['text_secondary']}}},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                # ✅ Region size display
                html.Div(id='region-size-display', style={
                    'textAlign': 'center',
                    'color': COLORS['text_secondary'],
                    'fontSize': '0.85rem',
                    'marginTop': '0.5rem'
                })
            ], style={'flex': '1', 'minWidth': '250px'})
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '2rem',
            'backgroundColor': COLORS['surface'],
            'padding': '1.5rem',
            'borderRadius': '1rem',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '1.5rem'
        }),
        
        # Output and Mix Controls
        html.Div([
            html.Div([
                html.Label("Output Viewer", style={
                    'color': COLORS['text'],
                    'fontSize': '1rem',
                    'fontWeight': '500',
                    'marginBottom': '0.75rem',
                    'display': 'block'
                }),
                dcc.RadioItems(
                    id='output-selector',
                    options=[
                        {'label': ' 📤 Output 1', 'value': '0'},
                        {'label': ' 📤 Output 2', 'value': '1'}
                    ],
                    value='0',
                    inline=True,
                    style={'color': COLORS['text']}
                )
            ], style={'flex': '1'}),
            
            html.Div([
                html.Button("✨ Mix Images", id='mix-button', n_clicks=0, style={
                    'width': '100%',
                    'padding': '1rem 2rem',
                    'fontSize': '1.1rem',
                    'fontWeight': '600',
                    'color': COLORS['text'],
                    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                    'border': 'none',
                    'borderRadius': '0.75rem',
                    'cursor': 'pointer',
                    'transition': 'transform 0.2s ease',
                    'boxShadow': f'0 4px 12px rgba(99, 102, 241, 0.3)'
                }),
                html.Button("🗑️ Clear All", id='clear-button', n_clicks=0, style={
                    'width': '100%',
                    'padding': '0.75rem 1.5rem',
                    'fontSize': '0.9rem',
                    'fontWeight': '500',
                    'color': COLORS['text'],
                    'backgroundColor': COLORS['surface_light'],
                    'border': f'1px solid {COLORS["border"]}',
                    'borderRadius': '0.5rem',
                    'cursor': 'pointer',
                    'marginTop': '0.5rem',
                    'transition': 'all 0.2s ease'
                })
            ], style={'flex': '1'})
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '2rem',
            'alignItems': 'center',
            'backgroundColor': COLORS['surface'],
            'padding': '1.5rem',
            'borderRadius': '1rem',
            'border': f'1px solid {COLORS["border"]}',
            'marginBottom': '1rem'
        }),
        
        # Status and Progress
        html.Div(id='mixing-status', style={
            'color': COLORS['text'],
            'fontSize': '1rem',
            'textAlign': 'center',
            'padding': '0.5rem',
            'marginBottom': '0.5rem'
        }),
        
        html.Div([
            html.Div(id='progress-bar-container', style={
                'width': '100%',
                'height': '8px',
                'backgroundColor': COLORS['surface_light'],
                'borderRadius': '4px',
                'overflow': 'hidden'
            }, children=[
                html.Div(id='progress-bar', style={
                    'width': '0%',
                    'height': '100%',
                    'background': f'linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                    'transition': 'width 0.3s ease'
                })
            ])
        ], style={'marginBottom': '2rem'})
    ]),
    
    html.Hr(style={'borderColor': COLORS['border'], 'margin': '2rem 0'}),
    
    # Output Images Section
    html.Div([
        html.H3("Output Images", style={
            'color': COLORS['text'],
            'fontSize': '1.5rem',
            'fontWeight': '600',
            'marginBottom': '1.5rem'
        }),
        html.Div([
            html.Div([
                create_image_viewer(f'output_{i}', f'Output {i+1}', False)
            ], style={'flex': '1', 'minWidth': '400px'})
            for i in range(2)
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '1rem'
        })
    ]),
    
    # Hidden stores for state management
    dcc.Store(id='unified-size-store', data=None),
    dcc.Store(id='region-rect-store', data={'x0': 0.35, 'y0': 0.35, 'x1': 0.65, 'y1': 0.65}),
    dcc.Interval(id='mixing-interval', interval=500, disabled=True),
    
], style={
    'maxWidth': '1600px',
    'margin': '0 auto',
    'padding': '2rem',
    'backgroundColor': COLORS['background'],
    'minHeight': '100vh',
    'color': COLORS['text']
})

# ═══════════════════════════════════════════════════════════════════════════
# CALLBACKS
# ═══════════════════════════════════════════════════════════════════════════

# ✅ Display image info
@callback(
    Output({'type': 'image-info', 'index': MATCH}, 'children'),
    [Input({'type': 'upload', 'index': MATCH}, 'contents')],
    [State({'type': 'upload', 'index': MATCH}, 'id')]
)
def display_image_info(contents, component_id):
    """Display image dimensions after upload."""
    if contents is None:
        return ""
    
    viewer_id = component_id['index']
    processor = image_processors.get(viewer_id)
    
    if processor and processor.shape:
        h, w = processor.shape
        return f"📐 {w} × {h} pixels"
    
    return ""

# ✅ FIXED: Handle upload and show image in the upload area itself
@callback(
    [Output({'type': 'upload-placeholder', 'index': MATCH}, 'style'),
     Output({'type': 'image-container', 'index': MATCH}, 'style'),
     Output({'type': 'image-container', 'index': MATCH}, 'children')],
    [Input({'type': 'upload', 'index': MATCH}, 'contents')],
    [State({'type': 'upload', 'index': MATCH}, 'id')]
)
def upload_and_display_in_area(contents, component_id):
    """✅ Display image directly in upload area after selection."""
    if contents is None:
        # Show placeholder
        return ({'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'height': '100%'},
                {'display': 'none', 'height': '100%'},
                None)
    
    viewer_id = component_id['index']
    processor = image_processors[viewer_id]
    processor.load_image(contents)
    
    # Create graph to show in the container
    fig = create_image_figure(processor.image, 
                             f"Original ({processor.shape[1]}×{processor.shape[0]})")
    
    graph = dcc.Graph(
        id={'type': 'graph-original', 'index': viewer_id},
        config={'displayModeBar': False},
        figure=fig,
        style={'height': '250px', 'width': '100%'}
    )
    
    # Hide placeholder, show image
    return ({'display': 'none'},
            {'display': 'block', 'height': '100%', 'lineHeight': 'normal'},
            graph)

@callback(
    Output('unified-size-store', 'data'),
    [Input({'type': 'upload', 'index': ALL}, 'contents')]
)
def unify_image_sizes(all_contents):
    """Unify sizes across all loaded images."""
    # Find minimum dimensions across all loaded images
    loaded_shapes = [p.shape for p in image_processors.values() 
                     if p.image is not None]
    
    if not loaded_shapes:
        raise PreventUpdate
    
    min_h = min(s[0] for s in loaded_shapes)
    min_w = min(s[1] for s in loaded_shapes)
    target_shape = (min_h, min_w)
    
    # Resize all images to minimum dimensions
    for p in image_processors.values():
        if p.image is not None:
            p.resize_to(target_shape)
    
    unified_size = {'height': min_h, 'width': min_w}
    
    return unified_size

# ✅ NEW: Mouse drag detection callback
@callback(
    Output({'type': 'mouse-drag-store', 'index': MATCH}, 'data'),
    [Input({'type': 'graph-component', 'index': MATCH}, 'relayoutData')],
    [State({'type': 'mouse-drag-store', 'index': MATCH}, 'data')],
    prevent_initial_call=True
)
def detect_mouse_drag(relayoutData, current_drag_data):
    """Detect mouse drag events on FT component graphs."""
    if relayoutData is None:
        raise PreventUpdate
    
    # Check if this is a drag event (pan/zoom)
    if 'xaxis.range[0]' in relayoutData or 'yaxis.range[0]' in relayoutData:
        # Extract drag information
        x_range = relayoutData.get('xaxis.range[0]', 0)
        y_range = relayoutData.get('yaxis.range[0]', 0)
        
        # Calculate delta from previous position
        prev_x = current_drag_data.get('x', 0)
        prev_y = current_drag_data.get('y', 0)
        
        delta_x = x_range - prev_x if prev_x != 0 else 0
        delta_y = y_range - prev_y if prev_y != 0 else 0
        
        return {
            'x': x_range,
            'y': y_range,
            'delta_x': delta_x,
            'delta_y': delta_y,
            'dragging': True
        }
    
    # Reset dragging state
    return {
        'x': current_drag_data.get('x', 0),
        'y': current_drag_data.get('y', 0),
        'delta_x': 0,
        'delta_y': 0,
        'dragging': False
    }

# ✅ NEW: Update bc-state from mouse drag OR sliders
@callback(
    [Output({'type': 'bc-state', 'index': MATCH}, 'data'),
     Output({'type': 'brightness-slider', 'index': MATCH}, 'value'),
     Output({'type': 'contrast-slider', 'index': MATCH}, 'value')],
    [Input({'type': 'mouse-drag-store', 'index': MATCH}, 'data'),
     Input({'type': 'brightness-slider', 'index': MATCH}, 'value'),
     Input({'type': 'contrast-slider', 'index': MATCH}, 'value')],
    [State({'type': 'bc-state', 'index': MATCH}, 'data')],
    prevent_initial_call=True
)
def update_bc_from_mouse_or_slider(drag_data, brightness_slider, contrast_slider, current_bc):
    """Update brightness/contrast from mouse drag OR sliders."""
    
    triggered = ctx.triggered_id
    
    if triggered is None:
        raise PreventUpdate
    
    # Check what triggered the callback
    if isinstance(triggered, dict):
        trigger_type = triggered.get('type', '')
        
        # If slider changed, update from slider
        if trigger_type in ['brightness-slider', 'contrast-slider']:
            adjusted_brightness = 128 + brightness_slider
            return (
                {'brightness': adjusted_brightness, 'contrast': contrast_slider},
                brightness_slider,
                contrast_slider
            )
        
        # If mouse drag detected
        if trigger_type == 'mouse-drag-store' and drag_data.get('dragging', False):
            delta_x = drag_data.get('delta_x', 0)
            delta_y = drag_data.get('delta_y', 0)
            
            # Skip if no movement
            if abs(delta_x) < 0.1 and abs(delta_y) < 0.1:
                raise PreventUpdate
            
            # Sensitivity factors
            BRIGHTNESS_SENSITIVITY = 0.2  # Sensitivity for brightness (up/down)
            CONTRAST_SENSITIVITY = 0.002   # Sensitivity for contrast (left/right)
            
            # Current values
            current_brightness = current_bc.get('brightness', 128)
            current_contrast = current_bc.get('contrast', 1.0)
            
            # Update based on mouse movement
            # Vertical drag (delta_y) → brightness (inverted: up = brighter)
            new_brightness = current_brightness - (delta_y * BRIGHTNESS_SENSITIVITY)
            new_brightness = np.clip(new_brightness, 0, 255)
            
            # Horizontal drag (delta_x) → contrast
            new_contrast = current_contrast + (delta_x * CONTRAST_SENSITIVITY)
            new_contrast = np.clip(new_contrast, 0.1, 3.0)
            
            # Update sliders to reflect mouse changes
            brightness_slider_val = new_brightness - 128
            
            return (
                {'brightness': new_brightness, 'contrast': new_contrast},
                brightness_slider_val,
                new_contrast
            )
    
    raise PreventUpdate

# ✅ Update component display with region rectangle overlay
@callback(
    Output({'type': 'graph-component', 'index': MATCH}, 'figure'),
    [Input({'type': 'component-selector', 'index': MATCH}, 'value'),
     Input({'type': 'upload', 'index': MATCH}, 'contents'),
     Input({'type': 'bc-state', 'index': MATCH}, 'data'),
     Input('region-rect-store', 'data'),
     Input('region-mode', 'value')],
    [State({'type': 'component-selector', 'index': MATCH}, 'id')]
)
def update_component_display(component_type, contents, bc_state, rect, region_mode, component_id):
    """Update FT component display with region rectangle overlay."""
    if contents is None:
        return create_empty_figure("Upload image first")
    
    viewer_id = component_id['index']
    processor = image_processors[viewer_id]
    
    if processor.image is None:
        return create_empty_figure("No image loaded")
    
    # Get selected component
    if component_type == 'magnitude':
        data = processor.get_magnitude()
        data = ImageProcessor.normalize_for_display(data, log_scale=True)
        title = "🔍 FT Magnitude"
    elif component_type == 'phase':
        data = processor.get_phase()
        data = ImageProcessor.normalize_for_display(data, log_scale=False)
        title = "🌀 FT Phase"
    elif component_type == 'real':
        data = processor.get_real()
        data = ImageProcessor.normalize_for_display(data, log_scale=False)
        title = "➕ FT Real"
    else:  # imaginary
        data = processor.get_imaginary()
        data = ImageProcessor.normalize_for_display(data, log_scale=False)
        title = "➖ FT Imaginary"
    
    # Apply brightness/contrast
    brightness = bc_state.get('brightness', 128)
    contrast = bc_state.get('contrast', 1.0)
    data = ImageProcessor.adjust_brightness_contrast(data, brightness, contrast)
    
    fig = create_image_figure(data, title, show_axes=True)
    
    # ✅ Add region rectangle overlay for input viewers only
    if viewer_id.startswith('input_') and rect is not None:
        h, w = processor.shape
        
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
        rect_color = COLORS['primary'] if use_inner else COLORS['error']
        
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
    
    return fig

# ✅ Update region rectangle in real-time when slider changes
@callback(
    [Output('region-rect-store', 'data'),
     Output('region-size-display', 'children')],
    [Input('region-size-slider', 'value')],
    [State('region-rect-store', 'data')]
)
def update_region_rect_from_slider(size_percent, current_rect):
    """Update region rectangle when size slider changes."""
    # Calculate new rect centered at current center
    center_x = (current_rect['x0'] + current_rect['x1']) / 2
    center_y = (current_rect['y0'] + current_rect['y1']) / 2
    
    size = size_percent / 100.0
    half_size = size / 2
    
    new_rect = {
        'x0': max(0, center_x - half_size),
        'y0': max(0, center_y - half_size),
        'x1': min(1, center_x + half_size),
        'y1': min(1, center_y + half_size)
    }
    
    display_text = f"📏 Region: {size_percent}% of frequency space"
    
    return new_rect, display_text

@callback(
    [Output('mixing-status', 'children'),
     Output('progress-bar', 'style'),
     Output('mixing-interval', 'disabled')],
    [Input('mix-button', 'n_clicks')],
    [State({'type': 'weight-slider', 'index': ALL}, 'value'),
     State('mixing-mode', 'value'),
     State('region-mode', 'value'),
     State('region-rect-store', 'data'),
     State('output-selector', 'value')],
    prevent_initial_call=True
)
def start_mixing(n_clicks, weights, mode, region_mode, rect, output_idx):
    """Start the mixing process in a background thread."""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    global mixing_thread
    
    # Cancel any existing mixing operation
    ft_mixer.cancel()
    if mixing_thread and mixing_thread.is_alive():
        mixing_thread.join(timeout=0.5)
    
    # Check if we have at least one input image
    input_processors = [image_processors[f'input_{i}'] for i in range(4)]
    if not any(p.image is not None for p in input_processors):
        return ("❌ Error: No input images loaded", 
                {'width': '0%', 'height': '100%', 
                 'background': f'linear-gradient(90deg, {COLORS["error"]} 0%, {COLORS["error"]} 100%)',
                 'transition': 'width 0.3s ease'},
                True)
    
    use_inner = (region_mode == 'inner')
    
    # Start mixing in background thread
    def mix_worker():
        ft_mixer.reset_cancel()
        result = ft_mixer.mix_components(
            input_processors, weights, mode, rect, use_inner
        )
        
        if result is not None and not ft_mixer.cancel_flag.is_set():
            # Store result in output processor
            output_key = f'output_{output_idx}'
            image_processors[output_key].image = result.astype(np.float64)
            image_processors[output_key].shape = result.shape
            image_processors[output_key].fft_result = None
    
    mixing_thread = threading.Thread(target=mix_worker, daemon=True)
    mixing_thread.start()
    
    return ("⚡ Mixing in progress...", 
            {'width': '50%', 'height': '100%',
             'background': f'linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
             'transition': 'width 0.3s ease'},
            False)

@callback(
    [Output('mixing-status', 'children', allow_duplicate=True),
     Output('progress-bar', 'style', allow_duplicate=True),
     Output('mixing-interval', 'disabled', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_0'}, 'figure', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_1'}, 'figure', allow_duplicate=True)],
    [Input('mixing-interval', 'n_intervals')],
    [State('output-selector', 'value')],
    prevent_initial_call=True
)
def check_mixing_progress(n_intervals, output_idx):
    """Check if mixing is complete and update output display."""
    global mixing_thread
    
    if mixing_thread is None or not mixing_thread.is_alive():
        # Mixing complete
        output_key = f'output_{output_idx}'
        processor = image_processors[output_key]
        
        if processor.image is not None:
            fig = create_image_figure(processor.image, f"✨ Mixed Result ({processor.shape[1]}×{processor.shape[0]})")
            
            # Return appropriate figure updates
            if output_idx == '0':
                return ("✅ Mixing complete!", 
                        {'width': '100%', 'height': '100%',
                         'background': f'linear-gradient(90deg, {COLORS["success"]} 0%, {COLORS["success"]} 100%)',
                         'transition': 'width 0.3s ease'},
                        True, fig, dash.no_update)
            else:
                return ("✅ Mixing complete!", 
                        {'width': '100%', 'height': '100%',
                         'background': f'linear-gradient(90deg, {COLORS["success"]} 0%, {COLORS["success"]} 100%)',
                         'transition': 'width 0.3s ease'},
                        True, dash.no_update, fig)
        else:
            return ("⚠️ Mixing cancelled", 
                    {'width': '0%', 'height': '100%',
                     'background': f'linear-gradient(90deg, {COLORS["error"]} 0%, {COLORS["error"]} 100%)',
                     'transition': 'width 0.3s ease'},
                    True, dash.no_update, dash.no_update)
    
    # Still mixing
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update


# ✅ Clear all images and reset
@callback(
    [Output({'type': 'upload', 'index': ALL}, 'contents'),
     Output({'type': 'graph-component', 'index': ALL}, 'figure', allow_duplicate=True),
     Output({'type': 'image-info', 'index': ALL}, 'children', allow_duplicate=True),
     Output('mixing-status', 'children', allow_duplicate=True),
     Output('progress-bar', 'style', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_0'}, 'figure', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_1'}, 'figure', allow_duplicate=True)],
    [Input('clear-button', 'n_clicks')],
    prevent_initial_call=True
)
def clear_all_images(n_clicks):
    """Clear all uploaded images and reset the application."""
    if n_clicks is None or n_clicks == 0:
        raise PreventUpdate
    
    # Reset all image processors
    global image_processors
    image_processors = {f'input_{i}': ImageProcessor() for i in range(4)}
    image_processors.update({f'output_{i}': ImageProcessor() for i in range(2)})
    
    # Create empty figures for all viewers
    component_figs = [create_empty_figure("Upload image first") for _ in range(4)]
    image_infos = [""] * 4
    
    # Reset upload contents to None - this will trigger upload_and_display_in_area 
    # callback which will automatically show placeholders and hide image containers
    return ([None] * 4, 
            component_figs, 
            image_infos, 
            "🗑️ All images cleared",
            {'width': '0%', 'height': '100%',
             'background': f'linear-gradient(90deg, {COLORS["warning"]} 0%, {COLORS["warning"]} 100%)',
             'transition': 'width 0.3s ease'},
            create_empty_figure("No result yet"),  # Output 0
            create_empty_figure("No result yet"))  # Output 1