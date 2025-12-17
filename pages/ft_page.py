"""
FT Magnitude/Phase Mixer - Refactored with Proper OOP
All logic encapsulated in classes. Callbacks are thin and just orchestrate.
✅ Output viewers now have FT component display (exactly similar to input viewers)
✅ All display logic in ImageViewer class
✅ All mixing logic in FTMixer class
✅ Callbacks just call class methods and update UI
"""

from dash import dcc, html, callback, Output, Input, State, ALL, MATCH, ctx
from dash.exceptions import PreventUpdate
import dash
import threading
from classes.ft_classes import ImageViewer, FTMixer

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
# GLOBAL STATE - OOP INSTANCES
# ═══════════════════════════════════════════════════════════════════════════

# Create ImageViewer instances for all viewers (4 inputs + 2 outputs)
image_viewers = {}
for i in range(4):
    image_viewers[f'input_{i}'] = ImageViewer(f'input_{i}', 'input', COLORS)
for i in range(2):
    image_viewers[f'output_{i}'] = ImageViewer(f'output_{i}', 'output', COLORS)

# FT Mixer instance
ft_mixer = FTMixer()

# Threading for async mixing
mixing_thread = None

# ═══════════════════════════════════════════════════════════════════════════
# UI COMPONENT FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def create_image_viewer_ui(viewer_id: str, title: str, is_input: bool = True):
    """Create UI components for an image viewer."""
    components = [
        html.H6(title, style={
            'textAlign': 'center',
            'marginBottom': '1rem',
            'color': COLORS['text'],
            'fontWeight': '600'
        })
    ]
    
    if is_input:
        # INPUT VIEWER: Upload area that becomes image display
        components.append(
            html.Div([
                dcc.Upload(
                    id={'type': 'upload', 'index': viewer_id},
                    children=html.Div([
                        html.Div(id={'type': 'upload-placeholder', 'index': viewer_id},
                                children=[
                                    html.Div('📤', style={'fontSize': '3rem', 'marginBottom': '0.5rem'}),
                                    html.Div('Click to Upload', style={'fontSize': '1rem', 'fontWeight': '500'}),
                                    html.Div('or drag and drop', style={'fontSize': '0.85rem', 'color': COLORS['text_secondary'], 'marginTop': '0.25rem'})
                                ],
                                style={
                                    'display': 'flex',
                                    'flexDirection': 'column',
                                    'alignItems': 'center',
                                    'justifyContent': 'center',
                                    'height': '100%'
                                }),
                        html.Div(id={'type': 'image-container', 'index': viewer_id},
                                style={'display': 'none', 'height': '100%'})
                    ]),
                    style={
                        'width': '100%',
                        'height': '250px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '0.75rem',
                        'textAlign': 'center',
                        'backgroundColor': COLORS['surface'],
                        'borderColor': COLORS['border'],
                        'cursor': 'pointer',
                        'transition': 'all 0.3s ease'
                    },
                    multiple=False
                ),
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
        # OUTPUT VIEWER: Traditional graph display
        components.append(
            dcc.Graph(
                id={'type': 'graph-original', 'index': viewer_id},
                config={'displayModeBar': False, 'scrollZoom': False},
                style={
                    'height': '250px',
                    'backgroundColor': COLORS['surface'],
                    'borderRadius': '0.75rem'
                }
            )
        )
    
    # Component selector and display (both input and output have this now!)
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
            config={'displayModeBar': False, 'scrollZoom': False},
            style={
                'height': '250px',
                'backgroundColor': COLORS['surface'],
                'borderRadius': '0.75rem',
                'cursor': 'crosshair'
            }
        )
    ])
    
    # Mouse drag instructions
    components.append(
        html.Div("🖱️ Drag: ↕️ Brightness | ↔️ Contrast", style={
            'textAlign': 'center',
            'color': COLORS['text_secondary'],
            'fontSize': '0.75rem',
            'marginTop': '0.5rem',
            'fontStyle': 'italic'
        })
    )
    
    # Brightness and Contrast Sliders
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
    
    # Stores
    components.extend([
        dcc.Store(id={'type': 'bc-state', 'index': viewer_id}, 
                 data={'brightness': 128, 'contrast': 1.0}),
        dcc.Store(id={'type': 'mouse-drag-store', 'index': viewer_id}, 
                 data={'x': 0, 'y': 0, 'dragging': False})
    ])
    
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
    # Back button
    html.Div([
        dcc.Link('← Back to Home', href='/', style={
            'color': COLORS['primary'],
            'textDecoration': 'none',
            'fontSize': '1rem',
            'fontWeight': '500'
        })
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
    
    html.P("✨ Output viewers now have FT components!", style={
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
                create_image_viewer_ui(f'input_{i}', f'Input {i+1}', is_input=True)
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
        
        # Weights
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
                        marks={0: '0', 0.5: '0.5', 1: '1'},
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
                    marks={10: '10%', 50: '50%', 100: '100%'},
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
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
                    'marginTop': '0.5rem'
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
        
        # Status
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
        html.H3("Output Images (Now with FT Component Display!)", style={
            'color': COLORS['text'],
            'fontSize': '1.5rem',
            'fontWeight': '600',
            'marginBottom': '1.5rem'
        }),
        html.Div([
            html.Div([
                create_image_viewer_ui(f'output_{i}', f'Output {i+1}', is_input=False)
            ], style={'flex': '1', 'minWidth': '400px'})
            for i in range(2)
        ], style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'gap': '1rem'
        })
    ]),
    
    # Hidden stores
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
# CALLBACKS - THIN AND CLEAN, JUST CALL CLASS METHODS
# ═══════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# IMAGE UPLOAD AND INFO
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output({'type': 'image-info', 'index': MATCH}, 'children'),
    [Input({'type': 'upload', 'index': MATCH}, 'contents')],
    [State({'type': 'upload', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def display_image_info(contents, component_id):
    """Display image info - THIN CALLBACK. Only for input viewers."""
    viewer_id = component_id['index']
    
    # Only process if this is an input viewer
    if not viewer_id.startswith('input_'):
        raise PreventUpdate
    
    if contents is None:
        return ""
    
    return image_viewers[viewer_id].get_image_info()


@callback(
    [Output({'type': 'upload-placeholder', 'index': MATCH}, 'style'),
     Output({'type': 'image-container', 'index': MATCH}, 'style'),
     Output({'type': 'image-container', 'index': MATCH}, 'children')],
    [Input({'type': 'upload', 'index': MATCH}, 'contents')],
    [State({'type': 'upload', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def upload_and_display(contents, component_id):
    """Handle upload - THIN CALLBACK. Only for input viewers."""
    viewer_id = component_id['index']
    
    # Only process if this is an input viewer
    if not viewer_id.startswith('input_'):
        raise PreventUpdate
    
    if contents is None:
        return ({'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 
                 'justifyContent': 'center', 'height': '100%'},
                {'display': 'none', 'height': '100%'},
                None)
    
    viewer = image_viewers[viewer_id]
    viewer.load_image(contents)
    
    fig = viewer.get_original_figure()
    graph = dcc.Graph(
        id={'type': 'graph-original', 'index': viewer_id},
        config={'displayModeBar': False},
        figure=fig,
        style={'height': '250px', 'width': '100%'}
    )
    
    return ({'display': 'none'},
            {'display': 'block', 'height': '100%', 'lineHeight': 'normal'},
            graph)


@callback(
    Output('unified-size-store', 'data'),
    [Input({'type': 'upload', 'index': ALL}, 'contents')]
)
def unify_sizes(all_contents):
    """Unify image sizes - THIN CALLBACK."""
    loaded_viewers = [v for v in image_viewers.values() if v.has_image()]
    if not loaded_viewers:
        raise PreventUpdate
    
    shapes = [v.processor.shape for v in loaded_viewers]
    min_h = min(s[0] for s in shapes)
    min_w = min(s[1] for s in shapes)
    target_shape = (min_h, min_w)
    
    for viewer in loaded_viewers:
        viewer.resize_to(target_shape)
    
    return {'height': min_h, 'width': min_w}

# ─────────────────────────────────────────────────────────────────────────────
# BRIGHTNESS/CONTRAST CONTROL
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output({'type': 'mouse-drag-store', 'index': MATCH}, 'data'),
    [Input({'type': 'graph-component', 'index': MATCH}, 'relayoutData')],
    [State({'type': 'mouse-drag-store', 'index': MATCH}, 'data')],
    prevent_initial_call=True
)
def detect_drag(relayoutData, current):
    """Detect mouse drag."""
    if not relayoutData or 'xaxis.range[0]' not in relayoutData:
        raise PreventUpdate
    
    x = relayoutData.get('xaxis.range[0]', 0)
    y = relayoutData.get('yaxis.range[0]', 0)
    prev_x = current.get('x', 0)
    prev_y = current.get('y', 0)
    
    return {
        'x': x, 'y': y,
        'delta_x': x - prev_x if prev_x != 0 else 0,
        'delta_y': y - prev_y if prev_y != 0 else 0,
        'dragging': True
    }


@callback(
    [Output({'type': 'bc-state', 'index': MATCH}, 'data'),
     Output({'type': 'brightness-slider', 'index': MATCH}, 'value'),
     Output({'type': 'contrast-slider', 'index': MATCH}, 'value')],
    [Input({'type': 'mouse-drag-store', 'index': MATCH}, 'data'),
     Input({'type': 'brightness-slider', 'index': MATCH}, 'value'),
     Input({'type': 'contrast-slider', 'index': MATCH}, 'value')],
    [State({'type': 'bc-state', 'index': MATCH}, 'data'),
     State({'type': 'brightness-slider', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_bc(drag_data, brightness_slider, contrast_slider, current_bc, component_id):
    """Update brightness/contrast - THIN CALLBACK."""
    triggered = ctx.triggered_id
    
    if triggered is None:
        raise PreventUpdate
    
    viewer_id = component_id['index']
    viewer = image_viewers[viewer_id]
    
    if isinstance(triggered, dict):
        trigger_type = triggered.get('type', '')
        
        # Slider changed
        if trigger_type in ['brightness-slider', 'contrast-slider']:
            adjusted_brightness = 128 + brightness_slider
            viewer.update_brightness_contrast(adjusted_brightness, contrast_slider)
            return (
                {'brightness': adjusted_brightness, 'contrast': contrast_slider},
                brightness_slider,
                contrast_slider
            )
        
        # Mouse drag
        if trigger_type == 'mouse-drag-store' and drag_data.get('dragging'):
            delta_x = drag_data.get('delta_x', 0)
            delta_y = drag_data.get('delta_y', 0)
            
            if abs(delta_x) < 0.1 and abs(delta_y) < 0.1:
                raise PreventUpdate
            
            current_brightness = current_bc.get('brightness', 128)
            current_contrast = current_bc.get('contrast', 1.0)
            
            import numpy as np
            new_brightness = np.clip(current_brightness - (delta_y * 0.2), 0, 255)
            new_contrast = np.clip(current_contrast + (delta_x * 0.002), 0.1, 3.0)
            
            viewer.update_brightness_contrast(new_brightness, new_contrast)
            
            return (
                {'brightness': new_brightness, 'contrast': new_contrast},
                new_brightness - 128,
                new_contrast
            )
    
    raise PreventUpdate

# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT DISPLAY
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    Output({'type': 'graph-component', 'index': MATCH}, 'figure'),
    [Input({'type': 'component-selector', 'index': MATCH}, 'value'),
     Input({'type': 'bc-state', 'index': MATCH}, 'data'),
     Input('region-rect-store', 'data'),
     Input('region-mode', 'value')],
    [State({'type': 'component-selector', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_component(component_type, bc_state, rect, region_mode, component_id):
    """Update component display - THIN CALLBACK, JUST CALLS CLASS METHOD."""
    viewer_id = component_id['index']
    viewer = image_viewers[viewer_id]
    
    viewer.update_component_selection(component_type)
    return viewer.get_component_figure(rect, region_mode)


@callback(
    Output({'type': 'graph-component', 'index': MATCH}, 'figure', allow_duplicate=True),
    [Input({'type': 'image-info', 'index': MATCH}, 'children')],
    [State({'type': 'component-selector', 'index': MATCH}, 'value'),
     State({'type': 'bc-state', 'index': MATCH}, 'data'),
     State('region-rect-store', 'data'),
     State('region-mode', 'value'),
     State({'type': 'component-selector', 'index': MATCH}, 'id')],
    prevent_initial_call=True
)
def update_component_after_upload(image_info, component_type, bc_state, rect, region_mode, component_id):
    """Trigger component update after image upload."""
    if not image_info:  # No image loaded
        raise PreventUpdate
    
    viewer_id = component_id['index']
    viewer = image_viewers[viewer_id]
    
    return viewer.get_component_figure(rect, region_mode)

# ─────────────────────────────────────────────────────────────────────────────
# REGION CONTROL
# ─────────────────────────────────────────────────────────────────────────────

@callback(
    [Output('region-rect-store', 'data'),
     Output('region-size-display', 'children')],
    [Input('region-size-slider', 'value')],
    [State('region-rect-store', 'data')]
)
def update_region(size_percent, current_rect):
    """Update region rectangle."""
    center_x = (current_rect['x0'] + current_rect['x1']) / 2
    center_y = (current_rect['y0'] + current_rect['y1']) / 2
    
    size = size_percent / 100.0
    half = size / 2
    
    import numpy as np
    new_rect = {
        'x0': max(0, center_x - half),
        'y0': max(0, center_y - half),
        'x1': min(1, center_x + half),
        'y1': min(1, center_y + half)
    }
    
    return new_rect, f"📏 Region: {size_percent}% of frequency space"

# ─────────────────────────────────────────────────────────────────────────────
# MIXING OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

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
    """Start mixing - THIN CALLBACK."""
    if not n_clicks:
        raise PreventUpdate
    
    global mixing_thread
    
    # Cancel existing
    ft_mixer.cancel()
    if mixing_thread and mixing_thread.is_alive():
        mixing_thread.join(timeout=0.5)
    
    # Check inputs
    input_processors = [image_viewers[f'input_{i}'].processor for i in range(4)]
    if not any(p.image is not None for p in input_processors):
        return ("❌ No input images", 
                {'width': '0%', 'height': '100%', 
                 'background': f'linear-gradient(90deg, {COLORS["error"]} 0%, {COLORS["error"]} 100%)'},
                True)
    
    use_inner = (region_mode == 'inner')
    
    # Mix in background
    def mix_worker():
        ft_mixer.reset_cancel()
        result = ft_mixer.mix_components(input_processors, weights, mode, rect, use_inner)
        
        if result is not None and not ft_mixer.cancel_flag.is_set():
            output_viewer = image_viewers[f'output_{output_idx}']
            output_viewer.load_from_array(result)
    
    mixing_thread = threading.Thread(target=mix_worker, daemon=True)
    mixing_thread.start()
    
    return ("⚡ Mixing...", 
            {'width': '50%', 'height': '100%',
             'background': f'linear-gradient(90deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)'},
            False)


@callback(
    [Output('mixing-status', 'children', allow_duplicate=True),
     Output('progress-bar', 'style', allow_duplicate=True),
     Output('mixing-interval', 'disabled', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_0'}, 'figure', allow_duplicate=True),
     Output({'type': 'graph-original', 'index': 'output_1'}, 'figure', allow_duplicate=True),
     Output({'type': 'graph-component', 'index': 'output_0'}, 'figure', allow_duplicate=True),
     Output({'type': 'graph-component', 'index': 'output_1'}, 'figure', allow_duplicate=True)],
    [Input('mixing-interval', 'n_intervals')],
    [State('output-selector', 'value'),
     State('region-rect-store', 'data'),
     State('region-mode', 'value')],
    prevent_initial_call=True
)
def check_progress(n_intervals, output_idx, rect, region_mode):
    """Check mixing progress - THIN CALLBACK."""
    global mixing_thread
    
    if mixing_thread is None or not mixing_thread.is_alive():
        output_viewer = image_viewers[f'output_{output_idx}']
        
        if output_viewer.has_image():
            fig = output_viewer.get_original_figure("✨ Mixed Result")
            component_fig = output_viewer.get_component_figure(rect, region_mode)
            
            if output_idx == '0':
                return ("✅ Complete!", 
                        {'width': '100%', 'height': '100%',
                         'background': f'linear-gradient(90deg, {COLORS["success"]} 0%, {COLORS["success"]} 100%)'},
                        True, fig, dash.no_update, component_fig, dash.no_update)
            else:
                return ("✅ Complete!", 
                        {'width': '100%', 'height': '100%',
                         'background': f'linear-gradient(90deg, {COLORS["success"]} 0%, {COLORS["success"]} 100%)'},
                        True, dash.no_update, fig, dash.no_update, component_fig)
        else:
            return ("⚠️ Cancelled", 
                    {'width': '0%', 'height': '100%',
                     'background': f'linear-gradient(90deg, {COLORS["error"]} 0%, {COLORS["error"]} 100%)'},
                    True, dash.no_update, dash.no_update, dash.no_update, dash.no_update)
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

# ─────────────────────────────────────────────────────────────────────────────
# CLEAR ALL
# ─────────────────────────────────────────────────────────────────────────────

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
def clear_all(n_clicks):
    """Clear all - THIN CALLBACK."""
    if not n_clicks:
        raise PreventUpdate
    
    # Reset all viewers
    global image_viewers
    for i in range(4):
        image_viewers[f'input_{i}'] = ImageViewer(f'input_{i}', 'input', COLORS)
    for i in range(2):
        image_viewers[f'output_{i}'] = ImageViewer(f'output_{i}', 'output', COLORS)
    
    empty_fig = image_viewers['input_0']._create_empty_figure("Upload image first")
    output_empty = image_viewers['output_0']._create_empty_figure("No result yet")
    
    return ([None] * 4,
            [empty_fig] * 6,  # 4 inputs + 2 outputs
            [""] * 4,
            "🗑️ Cleared",
            {'width': '0%', 'height': '100%',
             'background': f'linear-gradient(90deg, {COLORS["warning"]} 0%, {COLORS["warning"]} 100%)'},
            output_empty,
            output_empty)