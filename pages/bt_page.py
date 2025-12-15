#bt_page.py
import dash
from dash import dcc, html, callback, Input, Output, State, ctx, ALL
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np

# Import the modified beamforming classes
from classes.beamforming_classes import BeamformingSystem, Array, arrays_scenarios


beamforming_system = BeamformingSystem()


COLORS = {
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'secondary': '#8b5cf6',
    'background': '#1e293b',
    'surface': '#0f172a',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'info': '#3b82f6',
    'array_color': '#6366f1',
}

def create_empty_figure(title, message="Add arrays and configure parameters"):
    """Create an empty figure with a message"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=COLORS['text_secondary'])
    )
    fig.update_layout(
        title=title,
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
        font_color=COLORS['text'],
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)', zeroline=False),
        height=400
    )
    return fig

def create_wave_field_plot(wave_field, meshgrid_x, meshgrid_y):
    """Create heatmap for wave field visualization"""
    fig = go.Figure(data=go.Contour(
        z=wave_field,
        x=meshgrid_x[0, :],
        y=meshgrid_y[:, 0],
        colorscale='RdBu',
        ncontours=50,
        colorbar=dict(
            title=dict(
                text="Amplitude",
                font=dict(color=COLORS['text'])
            ),
            tickfont=dict(color=COLORS['text'])
        ),
        contours=dict(
            coloring='fill',
            showlabels=False
        ),
        hovertemplate='X: %{x:.1f}<br>Y: %{y:.1f}<br>Amplitude: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Wave Field Visualization",
        xaxis_title="X Position (wavelengths)",
        yaxis_title="Y Position (wavelengths)",
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
        font_color=COLORS['text'],
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig

def create_beam_pattern_plot(angles, pattern):
    """Create polar plot for beam pattern with improved beamwidth calculation"""
    # Convert angles from radians to degrees
    theta = np.degrees(angles)
    
    # Create polar plot
    fig = go.Figure()
    
    # Find main lobe direction (maximum power)
    main_lobe_idx = np.argmax(pattern)
    main_lobe_angle = theta[main_lobe_idx]
    max_power = pattern[main_lobe_idx]
    
    # Plot beam pattern with fill
    fig.add_trace(go.Scatterpolar(
        r=pattern,
        theta=theta,
        mode='lines',
        name='Beam Pattern',
        line=dict(color='#00FF00', width=2.5),
        fill='toself',
        fillcolor='rgba(0, 255, 0, 0.2)',
        hovertemplate='Azimuth: %{theta:.1f}°<br>Power: %{r:.1f} dB<extra></extra>'
    ))
    
    # Calculate -3 dB beamwidth
    half_power = max_power - 3
    
    # Find -3 dB points around main lobe
    beamwidth_angles = []
    
    # Search backwards from main lobe
    for i in range(main_lobe_idx, -1, -1):
        if pattern[i] <= half_power:
            if i < main_lobe_idx:
                # Interpolate between i and i+1
                if pattern[i+1] != pattern[i]:
                    t = (half_power - pattern[i]) / (pattern[i+1] - pattern[i])
                    angle = theta[i] + t * (theta[i+1] - theta[i])
                else:
                    angle = theta[i]
                beamwidth_angles.append(angle)
            break
    
    # Search forwards from main lobe
    for i in range(main_lobe_idx, len(pattern)):
        if pattern[i] <= half_power:
            if i > main_lobe_idx:
                # Interpolate between i-1 and i
                if pattern[i] != pattern[i-1]:
                    t = (half_power - pattern[i-1]) / (pattern[i] - pattern[i-1])
                    angle = theta[i-1] + t * (theta[i] - theta[i-1])
                else:
                    angle = theta[i]
                beamwidth_angles.append(angle)
            break
    
    # Add -3 dB markers and calculate beamwidth
    if len(beamwidth_angles) >= 2:
        for angle in beamwidth_angles:
            fig.add_trace(go.Scatterpolar(
                r=[half_power],
                theta=[angle],
                mode='markers',
                name='-3 dB',
                marker=dict(size=10, color='yellow', symbol='circle'),
                showlegend=False,
                hovertemplate=f'-3 dB: {angle:.1f}°<br>Power: {half_power:.1f} dB<extra></extra>'
            ))
        
        # Calculate beamwidth
        beamwidth = abs(beamwidth_angles[1] - beamwidth_angles[0])
        
        # Handle wraparound case
        if beamwidth > 180:
            beamwidth = 360 - beamwidth
        
        beamwidth_text = f'Half-Power Beamwidth: {beamwidth:.1f}°'
    else:
        # Fallback: estimate beamwidth from pattern
        above_half = pattern >= half_power
        if np.any(above_half):
            beamwidth_est = np.sum(above_half) * 360 / len(pattern)
            beamwidth_text = f'Beamwidth (est.): {beamwidth_est:.1f}°'
        else:
            beamwidth_text = 'Beamwidth: Unable to calculate'
    
    # Add main lobe marker
    fig.add_trace(go.Scatterpolar(
        r=[max_power],
        theta=[main_lobe_angle],
        mode='markers+text',
        name='Main Lobe',
        marker=dict(
            size=14,
            color='red',
            symbol='star',
            line=dict(color='white', width=1)
        ),
        text=[f'{main_lobe_angle:.0f}°'],
        textposition='top center',
        textfont=dict(color='white', size=10),
        hovertemplate=f'Main Lobe: {main_lobe_angle:.1f}°<br>Power: {max_power:.1f} dB<extra></extra>'
    ))
    
    # Find side lobes (local maxima)
    try:
        from scipy.signal import find_peaks
        # Find peaks at least 10 dB below main lobe
        peaks, properties = find_peaks(pattern, height=max_power-30, prominence=3, distance=10)
        
        # Filter out peaks too close to main lobe
        side_lobe_peaks = []
        for peak_idx in peaks:
            angle_diff = abs(theta[peak_idx] - main_lobe_angle)
            # Wrap around
            if angle_diff > 180:
                angle_diff = 360 - angle_diff
            # If more than 20° away from main lobe and not the main lobe
            if angle_diff > 20 and pattern[peak_idx] < max_power - 3:
                side_lobe_peaks.append(peak_idx)
        
        # Add side lobe markers (limit to 5)
        for peak_idx in side_lobe_peaks[:5]:
            side_lobe_angle = theta[peak_idx]
            side_lobe_power = pattern[peak_idx]
            fig.add_trace(go.Scatterpolar(
                r=[side_lobe_power],
                theta=[side_lobe_angle],
                mode='markers',
                name='Side Lobe',
                marker=dict(
                    size=8,
                    color='orange',
                    symbol='diamond'
                ),
                showlegend=False,
                hovertemplate=f'Side Lobe: {side_lobe_angle:.1f}°<br>Power: {side_lobe_power:.1f} dB<extra></extra>'
            ))
    except:
        # If scipy not available, skip side lobe detection
        pass
    
    # Update layout
    fig.update_layout(
        title=dict(
            text="Beam Pattern",
            font=dict(color=COLORS['text'], size=16)
        ),
        polar=dict(
            radialaxis=dict(
                visible=True,
                title=dict(
                    text="Power [dB]",
                    font=dict(color=COLORS['text'], size=12)
                ),
                range=[-40, 5],
                tickvals=[-40, -30, -20, -10, -3, 0],
                ticktext=['-40', '-30', '-20', '-10', '-3', '0'],
                tickfont=dict(color=COLORS['text'], size=10),
                gridcolor='rgba(255,255,255,0.2)',
                linecolor='rgba(255,255,255,0.5)',
                angle=90
            ),
            angularaxis=dict(
                direction="clockwise",
                rotation=90,  # 0° at top
                tickvals=[0, 45, 90, 135, 180, 225, 270, 315],
                ticktext=['0°', '45°', '90°', '135°', '180°', '225°', '270°', '315°'],
                tickfont=dict(color=COLORS['text'], size=10),
                gridcolor='rgba(255,255,255,0.2)',
                linecolor='rgba(255,255,255,0.5)'
            ),
            bgcolor=COLORS['surface'],
            hole=0.1
        ),
        annotations=[
            dict(
                text=beamwidth_text,
                x=0.5, y=0.02,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12, color='white'),
                bgcolor='rgba(0,0,0,0.5)',
                borderpad=4
            )
        ],
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.9)',
            font=dict(color=COLORS['text'], size=10),
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=500,
        margin=dict(l=20, r=20, t=50, b=50),
        paper_bgcolor=COLORS['surface'],
        plot_bgcolor=COLORS['surface'],
        font_color=COLORS['text']
    )
    
    return fig

def create_array_plot(array_positions):
    """Create scatter plot for array elements"""
    fig = go.Figure()
    
    # Plot array elements
    if array_positions:
        all_array_x = []
        all_array_y = []
        array_labels = []
        for array_name, positions in array_positions.items():
            if len(positions) > 0:
                positions_array = np.array(positions)
                all_array_x.extend(positions_array[:, 0])
                all_array_y.extend(positions_array[:, 1])
                array_labels.extend([f"{array_name} element {i+1}" for i in range(len(positions))])
        
        if all_array_x:
            fig.add_trace(go.Scatter(
                x=all_array_x,
                y=all_array_y,
                mode='markers',
                name='Array Elements',
                marker=dict(
                    size=8,
                    color=COLORS['array_color'],
                    symbol='circle',
                    line=dict(color='white', width=1)
                ),
                text=array_labels,
                hovertemplate='%{text}<br>X: %{x:.1f}<br>Y: %{y:.1f}<extra></extra>'
            ))
    
    fig.update_layout(
        title="Array Elements",
        xaxis_title="X Position",
        yaxis_title="Y Position",
        plot_bgcolor=COLORS['surface'],
        paper_bgcolor=COLORS['surface'],
        font_color=COLORS['text'],
        xaxis=dict(
            range=[-20, 20],
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)',
            scaleanchor="y",
            scaleratio=1
        ),
        yaxis=dict(
            range=[0, 40],
            gridcolor='rgba(255,255,255,0.1)',
            zeroline=True,
            zerolinecolor='rgba(255,255,255,0.3)'
        ),
        showlegend=True,
        legend=dict(
            bgcolor='rgba(30, 41, 59, 0.8)',
            font=dict(color=COLORS['text'])
        ),
        height=400,
        margin=dict(l=50, r=50, t=50, b=50)
    )
    
    return fig


layout = html.Div([
    # Main container
    html.Div([
        # Left Panel - Controls 
        html.Div([
            # Scenario Section
            html.Div([
                html.H4("Scenarios", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                dcc.Dropdown(
                    id='scenario-selector',
                    options=[
                        {'label': '5G', 'value': '5G'},
                        {'label': 'Ultrasound', 'value': 'Ultrasound'},
                        {'label': 'Tumor Ablation', 'value': 'Tumor Ablation'}
                    ],
                    value='5G',
                    clearable=False,
                    style={
                        'backgroundColor': COLORS['surface'],
                        'color': COLORS['text'],
                        'border': f'1px solid {COLORS["primary"]}',
                        'marginBottom': '10px'
                    }
                ),
                html.Button(
                    "Apply Scenario",
                    id='apply-scenario',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'backgroundColor': COLORS['primary'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'marginBottom': '20px'
                    }
                )
            ], style={
                'background': COLORS['surface'],
                'padding': '15px',
                'borderRadius': '8px',
                'marginBottom': '15px'
            }),
            
            # Array Controls Section
            html.Div([
                html.H4("Array Configuration", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                
                # Array Selection
                html.Div([
                    html.Label("Current Array:", style={'color': COLORS['text_secondary'], 'marginRight': '10px'}),
                    dcc.Dropdown(
                        id='current-array-selector',
                        options=[],
                        placeholder="Add Array...",
                        style={
                            'backgroundColor': COLORS['surface'],
                            'color': COLORS['text'],
                            'border': f'1px solid {COLORS["primary"]}',
                            'marginBottom': '10px'
                        }
                    )
                ]),
                
                # Array Type Radio Buttons
                html.Div([
                    dcc.RadioItems(
                        id='array-type',
                        options=[
                            {'label': ' Linear', 'value': 'Linear'},
                            {'label': ' Curved', 'value': 'Curved'}
                        ],
                        value='Linear',
                        labelStyle={'display': 'inline-block', 'marginRight': '20px', 'color': COLORS['text']},
                        style={'marginBottom': '15px'}
                    )
                ]),
                
                # Array Name
                html.Div([
                    html.Label("Array Name:", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    dcc.Input(
                        id='array-name',
                        type='text',
                        value='array',
                        style={
                            'width': '100%',
                            'backgroundColor': COLORS['surface'],
                            'color': COLORS['text'],
                            'border': f'1px solid {COLORS["primary"]}',
                            'borderRadius': '5px',
                            'padding': '8px',
                            'marginBottom': '10px'
                        }
                    )
                ]),
                
                # Number of Elements Slider
                html.Div([
                    html.Label("Number of Elements:", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    dcc.Slider(
                        id='num-elements',
                        min=1,
                        max=64,
                        step=1,
                        value=8,
                        marks={1: '1', 16: '16', 32: '32', 48: '48', 64: '64'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id='num-elements-value', style={'color': COLORS['text'], 'textAlign': 'center', 'marginBottom': '15px'})
                ], style={'marginBottom': '15px'}),
                
                # Element Spacing Slider (for linear arrays)
                html.Div(id='element-spacing-control', style={'marginBottom': '15px'}, children=[
                    html.Label("Element Spacing (λ):", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    dcc.Slider(
                        id='element-spacing',
                        min=0.5,
                        max=4,
                        step=0.5,
                        value=0.5,
                        marks={0.5: '0.5', 1: '1', 2: '2', 3: '3', 4: '4'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id='element-spacing-value', style={'color': COLORS['text'], 'textAlign': 'center'})
                ]),
                
                # Frequencies Input
                html.Div([
                    html.Label("Frequencies (Hz, comma separated):", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    dcc.Input(
                        id='frequencies',
                        type='text',
                        value='1',
                        style={
                            'width': '100%',
                            'backgroundColor': COLORS['surface'],
                            'color': COLORS['text'],
                            'border': f'1px solid {COLORS["primary"]}',
                            'borderRadius': '5px',
                            'padding': '8px',
                            'marginBottom': '15px'
                        }
                    )
                ]),
                
                # Position Inputs
                html.Div([
                    html.Label("Position:", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    html.Div([
                        dcc.Input(
                            id='array-pos-x',
                            type='number',
                            value=0,
                            style={
                                'width': '48%',
                                'backgroundColor': COLORS['surface'],
                                'color': COLORS['text'],
                                'border': f'1px solid {COLORS["primary"]}',
                                'borderRadius': '5px',
                                'padding': '8px',
                                'marginRight': '4%'
                            },
                            placeholder='X'
                        ),
                        dcc.Input(
                            id='array-pos-y',
                            type='number',
                            value=0,
                            style={
                                'width': '48%',
                                'backgroundColor': COLORS['surface'],
                                'color': COLORS['text'],
                                'border': f'1px solid {COLORS["primary"]}',
                                'borderRadius': '5px',
                                'padding': '8px'
                            },
                            placeholder='Y'
                        )
                    ], style={'display': 'flex', 'marginBottom': '15px'})
                ]),
                
                # Steering Angle Slider
                html.Div([
                    html.Label("Steering Angle:", style={
                        'color': COLORS['text_secondary'],
                        'marginBottom': '5px',
                        'display': 'block'
                    }),
                    dcc.Slider(
                        id='steering-angle',
                        min=-90,
                        max=90,
                        step=1,
                        value=0,
                        marks={-90: '-90°', -45: '-45°', 0: '0°', 45: '45°', 90: '90°'},
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    html.Div(id='steering-angle-value', style={'color': COLORS['text'], 'textAlign': 'center', 'marginBottom': '15px'})
                ], style={'marginBottom': '15px'}),
                
                # Curved Array Controls (hidden by default)
                html.Div(id='curved-array-controls', style={'display': 'none', 'marginBottom': '15px'}, children=[
                    html.Div([
                        html.Label("Radius:", style={
                            'color': COLORS['text_secondary'],
                            'marginBottom': '5px',
                            'display': 'block'
                        }),
                        dcc.Input(
                            id='radius',
                            type='number',
                            value=1,
                            style={
                                'width': '100%',
                                'backgroundColor': COLORS['surface'],
                                'color': COLORS['text'],
                                'border': f'1px solid {COLORS["primary"]}',
                                'borderRadius': '5px',
                                'padding': '8px',
                                'marginBottom': '15px'
                            }
                        )
                    ]),
                    html.Div([
                        html.Label("Arc Angle:", style={
                            'color': COLORS['text_secondary'],
                            'marginBottom': '5px',
                            'display': 'block'
                        }),
                        dcc.Slider(
                            id='arc-angle',
                            min=-120,
                            max=120,
                            step=1,
                            value=120,
                            marks={-120: '-120°', -60: '-60°', 0: '0°', 60: '60°', 120: '120°'},
                            tooltip={"placement": "bottom", "always_visible": False}
                        ),
                        html.Div(id='arc-angle-value', style={'color': COLORS['text'], 'textAlign': 'center'})
                    ])
                ]),
                
                # Array Action Buttons
                html.Div([
                    html.Button(
                        "Add New Array",
                        id='add-new-array',
                        n_clicks=0,
                        style={
                            'width': '48%',
                            'backgroundColor': COLORS['primary'],
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'cursor': 'pointer',
                            'marginRight': '4%'
                        }
                    ),
                    html.Button(
                        "Remove Array",
                        id='remove-array',
                        n_clicks=0,
                        style={
                            'width': '48%',
                            'backgroundColor': COLORS['danger'],
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px',
                            'borderRadius': '5px',
                            'cursor': 'pointer'
                        }
                    )
                ], style={'display': 'flex', 'marginBottom': '20px'}),
                
                # Save/Update Array Button
                html.Button(
                    "Save/Update Array",
                    id='save-array',
                    n_clicks=0,
                    style={
                        'width': '100%',
                        'backgroundColor': COLORS['success'],
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px',
                        'borderRadius': '5px',
                        'cursor': 'pointer',
                        'marginBottom': '20px'
                    }
                )
            ], style={
                'background': COLORS['surface'],
                'padding': '15px',
                'borderRadius': '8px'
            })
        ], style={
            'flex': '0 0 400px',
            'padding': '20px',
            'overflowY': 'auto',
            'height': 'calc(100vh - 40px)',
            'borderRight': f'2px solid {COLORS["surface"]}'
        }),
        
        # Right Panel - Visualizations 
        html.Div([
            # Top Row: Wave Field and Beam Pattern
            html.Div([
                # Wave Field Visualization
                html.Div([
                    html.H4("Wave Field Visualization", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    dcc.Graph(
                        id='wave-field-plot',
                        figure=create_empty_figure('Wave Field Visualization'),
                        style={'height': '400px'},
                        config={'displayModeBar': True, 'scrollZoom': True}
                    )
                ], style={
                    'flex': '1',
                    'marginRight': '10px',
                    'background': COLORS['surface'],
                    'padding': '15px',
                    'borderRadius': '8px'
                }),
                
                # Beam Pattern
                html.Div([
                    html.H4("Beam Pattern", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    dcc.Graph(
                        id='beam-pattern-plot',
                        figure=create_empty_figure('Beam Pattern'),
                        style={'height': '400px'},
                        config={'displayModeBar': True}
                    )
                ], style={
                    'flex': '1',
                    'background': COLORS['surface'],
                    'padding': '15px',
                    'borderRadius': '8px'
                })
            ], style={
                'display': 'flex',
                'marginBottom': '20px',
                'height': '450px'
            }),
            
            # Bottom Row: Array Plot and Array Info
            html.Div([
                # Array Plot
                html.Div([
                    html.H4("Array Elements", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    dcc.Graph(
                        id='array-plot',
                        figure=create_empty_figure('Array Elements'),
                        style={'height': '400px'},
                        config={'displayModeBar': True, 'scrollZoom': True}
                    )
                ], style={
                    'flex': '1',
                    'marginRight': '10px',
                    'background': COLORS['surface'],
                    'padding': '15px',
                    'borderRadius': '8px'
                }),
                
                # Array Information Panel
                html.Div([
                    html.H4("Array Information", style={'color': COLORS['text'], 'marginBottom': '10px'}),
                    
                    # Array Selection for Info
                    html.Div([
                        html.Label("Select Array:", style={
                            'color': COLORS['text_secondary'],
                            'marginBottom': '5px',
                            'display': 'block'
                        }),
                        dcc.Dropdown(
                            id='array-info-selector',
                            options=[],
                            style={
                                'backgroundColor': COLORS['surface'],
                                'color': COLORS['text'],
                                'border': f'1px solid {COLORS["primary"]}',
                                'marginBottom': '15px'
                            }
                        )
                    ]),
                    
                    # Array Info Display
                    html.Div(id='array-info-display', style={
                        'color': COLORS['text_secondary'],
                        'fontSize': '0.9rem',
                        'lineHeight': '1.6'
                    }, children=[
                        html.Div([
                            html.Strong("Type: ", style={'color': COLORS['text']}),
                            html.Span(id='info-type', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Transmitter Number: ", style={'color': COLORS['text']}),
                            html.Span(id='info-num-elements', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Elements Spacing: ", style={'color': COLORS['text']}),
                            html.Span(id='info-element-spacing', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Frequencies: ", style={'color': COLORS['text']}),
                            html.Span(id='info-frequencies', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Position: ", style={'color': COLORS['text']}),
                            html.Span(id='info-position', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div([
                            html.Strong("Steering Angle: ", style={'color': COLORS['text']}),
                            html.Span(id='info-steering-angle', children="………")
                        ], style={'marginBottom': '8px'}),
                        html.Div(id='curved-array-info', style={'marginBottom': '8px'}, children=[
                            html.Div([
                                html.Strong("Radius: ", style={'color': COLORS['text']}),
                                html.Span(id='info-radius', children="………")
                            ]),
                            html.Div([
                                html.Strong("Arc Angle: ", style={'color': COLORS['text']}),
                                html.Span(id='info-arc-angle', children="………")
                            ])
                        ])
                    ])
                ], style={
                    'flex': '0 0 300px',
                    'background': COLORS['surface'],
                    'padding': '15px',
                    'borderRadius': '8px',
                    'overflowY': 'auto'
                })
            ], style={
                'display': 'flex',
                'height': '450px'
            })
        ], style={
            'flex': '1',
            'padding': '20px',
            'overflowY': 'auto',
            'height': 'calc(100vh - 40px)'
        })
    ], style={
        'display': 'flex',
        'height': 'calc(100vh - 40px)',
        'backgroundColor': COLORS['background']
    }),
    
    # Hidden storage components
    dcc.Store(id='system-state', data={
        'arrays': {},
        'current_array': None
    }),
    dcc.Store(id='visualization-data'),
    dcc.Store(id='error-data', data={'message': '', 'show': False}),
    
    # Error modal
    dbc.Modal([
        dbc.ModalHeader("Error", style={'color': COLORS['text'], 'backgroundColor': COLORS['surface']}),
        dbc.ModalBody(id='error-message', style={'backgroundColor': COLORS['surface'], 'color': COLORS['text']}),
        dbc.ModalFooter(
            dbc.Button("Close", id="close-error", className="ml-auto", style={
                'backgroundColor': COLORS['primary'],
                'border': 'none'
            })
        ),
    ], id='error-modal', is_open=False, style={'color': COLORS['text']}),
])

# Callbacks

@callback(
    Output('curved-array-controls', 'style'),
    Output('element-spacing-control', 'style'),
    Input('array-type', 'value')
)
def toggle_array_controls(array_type):
    """Show/hide controls based on array type"""
    if array_type == 'Curved':
        return {'display': 'block', 'marginBottom': '15px'}, {'display': 'none'}
    else:
        return {'display': 'none', 'marginBottom': '15px'}, {'display': 'block', 'marginBottom': '15px'}

@callback(
    Output('num-elements-value', 'children'),
    Input('num-elements', 'value')
)
def update_num_elements_display(value):
    return f"{value} elements"

@callback(
    Output('element-spacing-value', 'children'),
    Input('element-spacing', 'value')
)
def update_element_spacing_display(value):
    return f"{value}λ"

@callback(
    Output('steering-angle-value', 'children'),
    Input('steering-angle', 'value')
)
def update_steering_angle_display(value):
    return f"{value}°"

@callback(
    Output('arc-angle-value', 'children'),
    Input('arc-angle', 'value')
)
def update_arc_angle_display(value):
    return f"{value}°"

@callback(
    Output('current-array-selector', 'options'),
    Output('array-info-selector', 'options'),
    Output('system-state', 'data', allow_duplicate=True),
    Output('visualization-data', 'data', allow_duplicate=True),
    Output('error-data', 'data', allow_duplicate=True),
    [Input('add-new-array', 'n_clicks'),
     Input('save-array', 'n_clicks'),
     Input('remove-array', 'n_clicks'),
     Input('apply-scenario', 'n_clicks')],
    [State('system-state', 'data'),
     State('current-array-selector', 'value'),
     State('array-name', 'value'),
     State('array-type', 'value'),
     State('num-elements', 'value'),
     State('frequencies', 'value'),
     State('array-pos-x', 'value'),
     State('array-pos-y', 'value'),
     State('steering-angle', 'value'),
     State('element-spacing', 'value'),
     State('radius', 'value'),
     State('arc-angle', 'value'),
     State('scenario-selector', 'value')],
    prevent_initial_call=True
)
def manage_arrays(add_clicks, save_clicks, remove_clicks, scenario_clicks,
                  state_data, current_array_name, array_name, array_type, num_elements,
                  frequencies, pos_x, pos_y, steering_angle, element_spacing, radius, 
                  arc_angle, scenario):
    """Manage array operations and update visualizations"""
    ctx_triggered = ctx.triggered_id
    
    arrays = state_data.get('arrays', {})
    vis_data = None
    error_data = {'message': '', 'show': False}
    
    try:
        # Apply scenario
        if ctx_triggered == 'apply-scenario' and scenario:
            success = beamforming_system.apply_scenario(scenario)
            if not success:
                raise ValueError(f"Unknown scenario: {scenario}")
            
            # Update arrays from system
            arrays = {}
            for name, array in beamforming_system.arrays.items():
                arrays[name] = {
                    'name': array.name,
                    'type': array.array_type,
                    'num_elements': array.number_of_elements,
                    'frequencies': array.frequencies,
                    'position': array.position,
                    'steering_angle': np.degrees(array.steering_angle),
                    'element_spacing': array.elements_spacing,
                    'radius': array.radius,
                    'arc_angle': array.arc_angle
                }
            
            state_data['arrays'] = arrays
            state_data['current_array'] = list(arrays.keys())[-1] if arrays else None
            
            # Calculate visualization data
            if arrays:
                result = beamforming_system.calculate_wave_field()
                vis_data = {
                    'wave_field': result['wave_field'].tolist(),
                    'beam_profile': result['beam_profile'].tolist(),
                    'polar_power': result['polar_power'].tolist(),
                    'frame': result['frame']
                }
        
        # Add new array
        elif ctx_triggered == 'add-new-array':
            if not array_name or array_name.strip() == '':
                array_name = f"Array_{len(arrays) + 1}"
            else:
                # Check if array name already exists
                if array_name in arrays:
                    array_name = f"{array_name}_{len(arrays) + 1}"
            
            # Parse frequencies
            try:
                freq_list = [float(f.strip()) for f in frequencies.split(',') if f.strip()]
            except:
                freq_list = [1.0]
            
            # Set defaults based on array type
            if array_type == 'Linear':
                element_spacing_val = float(element_spacing or 0.5)
                radius_val = None
                arc_angle_val = None
            else:  # Curved
                element_spacing_val = None
                radius_val = float(radius or 1)
                arc_angle_val = float(arc_angle or 120)
            
            arrays[array_name] = {
                'name': array_name,
                'type': array_type,
                'num_elements': num_elements,
                'frequencies': freq_list,
                'position': [float(pos_x or 0), float(pos_y or 0)],
                'steering_angle': float(steering_angle or 0),
                'element_spacing': element_spacing_val,
                'radius': radius_val,
                'arc_angle': arc_angle_val
            }
            
            state_data['arrays'] = arrays
            state_data['current_array'] = array_name
            
            # Add to beamforming system
            beamforming_system.add_array(
                array_name, array_type, num_elements, freq_list, 
                float(steering_angle or 0), [float(pos_x or 0), float(pos_y or 0)],
                element_spacing=element_spacing_val,
                radius=radius_val,
                arc_angle=arc_angle_val
            )
            
            # Calculate visualization data
            if beamforming_system.arrays:
                result = beamforming_system.calculate_wave_field()
                vis_data = {
                    'wave_field': result['wave_field'].tolist(),
                    'beam_profile': result['beam_profile'].tolist(),
                    'polar_power': result['polar_power'].tolist(),
                    'frame': result['frame']
                }
        
        # Save/update array
        elif ctx_triggered == 'save-array':
            # Check if we're updating an existing array or creating a new one
            is_updating = False
            target_array_name = None
            
            # Case 1: Array name exists in arrays (direct update)
            if array_name in arrays:
                is_updating = True
                target_array_name = array_name
            # Case 2: Current array selector has a value (updating selected array)
            elif current_array_name and current_array_name in arrays:
                is_updating = True
                target_array_name = current_array_name
            # Case 3: No existing array found - create new one
            else:
                is_updating = False
            
            # Parse frequencies
            try:
                freq_list = [float(f.strip()) for f in frequencies.split(',') if f.strip()]
            except:
                freq_list = [1.0]
            
            # Set defaults based on array type
            if array_type == 'Linear':
                element_spacing_val = float(element_spacing or 0.5)
                radius_val = None
                arc_angle_val = None
            else:  # Curved
                element_spacing_val = None
                radius_val = float(radius or 1)
                arc_angle_val = float(arc_angle or 120)
            
            if is_updating:
                # UPDATE EXISTING ARRAY
                old_name = target_array_name
                new_name = array_name
                
                # If name changed and new name already exists (and it's not the same array)
                if new_name != old_name and new_name in arrays and new_name != old_name:
                    raise ValueError(f"Array name '{new_name}' already exists!")
                
                # Update the array data
                arrays[old_name] = {
                    'name': new_name,
                    'type': array_type,
                    'num_elements': num_elements,
                    'frequencies': freq_list,
                    'position': [float(pos_x or 0), float(pos_y or 0)],
                    'steering_angle': float(steering_angle or 0),
                    'element_spacing': element_spacing_val,
                    'radius': radius_val,
                    'arc_angle': arc_angle_val
                }
                
                # If name changed, update the dictionary key
                if new_name != old_name:
                    arrays[new_name] = arrays.pop(old_name)
                    target_array_name = new_name
                
                state_data['current_array'] = target_array_name
                
                # Prepare update parameters for beamforming system
                update_params = {
                    'name': new_name,
                    'array_type': array_type,
                    'number_of_elements': num_elements,
                    'frequencies': freq_list,
                    'steering_angle': float(steering_angle or 0),
                    'position': [float(pos_x or 0), float(pos_y or 0)],
                    'element_spacing': element_spacing_val,
                    'radius': radius_val,
                    'arc_angle': arc_angle_val
                }
                
                # Update in beamforming system
                updated_name = beamforming_system.update_array(old_name, **update_params)
                
                # If name changed in beamforming system, update the key
                if updated_name and updated_name != old_name:
                    if old_name in beamforming_system.arrays:
                        beamforming_system.arrays[updated_name] = beamforming_system.arrays.pop(old_name)
                
            else:
                # CREATE NEW ARRAY (if doesn't exist)
                if not array_name or array_name.strip() == '':
                    array_name = f"Array_{len(arrays) + 1}"
                elif array_name in arrays:
                    array_name = f"{array_name}_{len(arrays) + 1}"
                
                arrays[array_name] = {
                    'name': array_name,
                    'type': array_type,
                    'num_elements': num_elements,
                    'frequencies': freq_list,
                    'position': [float(pos_x or 0), float(pos_y or 0)],
                    'steering_angle': float(steering_angle or 0),
                    'element_spacing': element_spacing_val,
                    'radius': radius_val,
                    'arc_angle': arc_angle_val
                }
                
                state_data['current_array'] = array_name
                
                # Add to beamforming system
                beamforming_system.add_array(
                    array_name, array_type, num_elements, freq_list, 
                    float(steering_angle or 0), [float(pos_x or 0), float(pos_y or 0)],
                    element_spacing=element_spacing_val,
                    radius=radius_val,
                    arc_angle=arc_angle_val
                )
            
            state_data['arrays'] = arrays
            
            # Calculate visualization data
            if beamforming_system.arrays:
                result = beamforming_system.calculate_wave_field()
                vis_data = {
                    'wave_field': result['wave_field'].tolist(),
                    'beam_profile': result['beam_profile'].tolist(),
                    'polar_power': result['polar_power'].tolist(),
                    'frame': result['frame']
                }
        
        # Remove array
        elif ctx_triggered == 'remove-array':
            if current_array_name and current_array_name in arrays:
                # Remove from arrays dict
                del arrays[current_array_name]
                
                # Remove from beamforming system
                beamforming_system.remove_array(current_array_name)
                
                # Update current array selection
                if arrays:
                    state_data['current_array'] = list(arrays.keys())[-1]
                else:
                    state_data['current_array'] = None
                
                state_data['arrays'] = arrays
                
                # Calculate visualization data if arrays still exist
                if beamforming_system.arrays:
                    result = beamforming_system.calculate_wave_field()
                    vis_data = {
                        'wave_field': result['wave_field'].tolist(),
                        'beam_profile': result['beam_profile'].tolist(),
                        'polar_power': result['polar_power'].tolist(),
                        'frame': result['frame']
                    }
            else:
                raise ValueError("Please select an array to remove.")
    
    except Exception as e:
        error_data = {'message': str(e), 'show': True}
        print(f"Error in manage_arrays: {e}")
        # Revert to previous state on error
        arrays = state_data.get('arrays', {})
    
    # Create options for dropdowns
    array_options = [{'label': name, 'value': name} for name in arrays.keys()]
    if not array_options:
        array_options = [{'label': 'Add Array...', 'value': 'add'}]
    
    return array_options, array_options, state_data, vis_data, error_data

@callback(
    [Output('array-name', 'value'),
     Output('array-type', 'value'),
     Output('num-elements', 'value'),
     Output('frequencies', 'value'),
     Output('array-pos-x', 'value'),
     Output('array-pos-y', 'value'),
     Output('steering-angle', 'value'),
     Output('element-spacing', 'value'),
     Output('radius', 'value'),
     Output('arc-angle', 'value')],
    Input('current-array-selector', 'value'),
    State('system-state', 'data')
)
def update_array_form(selected_array, state_data):
    """Update form fields when array is selected"""
    if not selected_array or selected_array == 'add':
        return 'array', 'Linear', 8, '1', 0, 0, 0, 0.5, 1, 120
    
    arrays = state_data.get('arrays', {})
    if selected_array in arrays:
        array = arrays[selected_array]
        
        # Format frequencies
        freq_str = ','.join(str(f) for f in array['frequencies'])
        
        return (
            array['name'],
            array['type'],
            array['num_elements'],
            freq_str,
            array['position'][0],
            array['position'][1],
            array['steering_angle'],
            array['element_spacing'] or 0.5,
            array['radius'] or 1,
            array['arc_angle'] or 120
        )
    
    return 'array', 'Linear', 8, '1', 0, 0, 0, 0.5, 1, 120

@callback(
    [Output('info-type', 'children'),
     Output('info-num-elements', 'children'),
     Output('info-element-spacing', 'children'),
     Output('info-frequencies', 'children'),
     Output('info-position', 'children'),
     Output('info-steering-angle', 'children'),
     Output('info-radius', 'children'),
     Output('info-arc-angle', 'children'),
     Output('curved-array-info', 'style')],
    Input('array-info-selector', 'value'),
    State('system-state', 'data')
)
def update_array_info(selected_array, state_data):
    """Update array information display"""
    if not selected_array:
        return ('………', '………', '………', '………', '………', '………', '………', '………', {'display': 'none'})
    
    arrays = state_data.get('arrays', {})
    if selected_array in arrays:
        array = arrays[selected_array]
        
        # Format frequencies
        freq_str = ', '.join(str(f) for f in array['frequencies']) + 'Hz'
        
        # Format position
        pos_str = f"{array['position'][0]}x {array['position'][1]}y"
        
        # Format steering angle
        steering_str = f"{int(array['steering_angle'])}˚"
        
        # Show/hide curved array info
        if array['type'] == 'Curved':
            curved_style = {'display': 'block'}
            radius_str = f"{array['radius']}m" if array['radius'] else "………"
            arc_str = f"{int(array['arc_angle'])}˚" if array['arc_angle'] else "………"
            element_spacing_str = "………"
        else:
            curved_style = {'display': 'none'}
            radius_str = "………"
            arc_str = "………"
            element_spacing_str = f"{array['element_spacing']}λ" if array['element_spacing'] else "………"
        
        return (
            array['type'],
            str(array['num_elements']),
            element_spacing_str,
            freq_str,
            pos_str,
            steering_str,
            radius_str,
            arc_str,
            curved_style
        )
    
    return ('………', '………', '………', '………', '………', '………', '………', '………', {'display': 'none'})

@callback(
    [Output('wave-field-plot', 'figure'),
     Output('beam-pattern-plot', 'figure'),
     Output('array-plot', 'figure')],
    [Input('visualization-data', 'data'),
     Input('system-state', 'modified_timestamp')],
    [State('system-state', 'data')]
)
def update_visualizations(vis_data, state_timestamp, state_data):
    """Update all visualizations"""
    # Get array positions
    array_positions = {}
    if state_data and 'arrays' in state_data:
        for name, array in state_data['arrays'].items():
            if name in beamforming_system.arrays:
                array_obj = beamforming_system.arrays[name]
                positions = array_obj.array_data["positions"]
                array_positions[name] = positions.tolist()
    
    # If no visualization data yet, create empty plots
    if not vis_data:
        return (
            create_empty_figure('Wave Field Visualization'),
            create_empty_figure('Beam Pattern'),
            create_array_plot(array_positions)
        )
    
    try:
        # Create wave field plot
        wave_field = np.array(vis_data['wave_field'])
        wave_fig = create_wave_field_plot(
            wave_field, 
            beamforming_system.meshgrid_x, 
            beamforming_system.meshgrid_y
        )
        
        # Create beam pattern plot
        angles = beamforming_system.angles
        pattern = np.array(vis_data['polar_power'])
        beam_fig = create_beam_pattern_plot(angles, pattern)
        
        # Create array plot
        array_fig = create_array_plot(array_positions)
        
        return wave_fig, beam_fig, array_fig
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        # Return empty plots on error
        return (
            create_empty_figure('Wave Field Visualization', 'Error creating plot'),
            create_empty_figure('Beam Pattern', 'Error creating plot'),
            create_array_plot(array_positions)
        )

@callback(
    [Output('error-modal', 'is_open'),
     Output('error-message', 'children')],
    [Input('error-data', 'data'),
     Input('close-error', 'n_clicks')],
    [State('error-modal', 'is_open')]
)
def handle_error_modal(error_data, close_clicks, is_open):
    """Handle error modal display"""
    ctx_triggered = ctx.triggered_id
    
    # Close modal when close button is clicked
    if ctx_triggered == 'close-error' and close_clicks:
        return False, ""
    
    # Open modal when there's an error
    if ctx_triggered == 'error-data' and error_data.get('show', False):
        return True, error_data.get('message', 'An error occurred')
    
    return is_open, "" 
@callback(
    Output('visualization-data', 'data', allow_duplicate=True),
    Output('system-state', 'data', allow_duplicate=True),
    [Input('steering-angle', 'value')],
    [State('current-array-selector', 'value'),
     State('system-state', 'data')],
    prevent_initial_call=True
)
def update_steering_angle_realtime(steering_angle, current_array_name, state_data):
    """Update steering angle in real-time and recalculate visualizations"""
    if not current_array_name or current_array_name == 'add':
        return dash.no_update, dash.no_update
    
    arrays = state_data.get('arrays', {})
    
    if current_array_name not in arrays:
        return dash.no_update, dash.no_update
    
    try:
        # Update the steering angle in the state
        arrays[current_array_name]['steering_angle'] = float(steering_angle)
        state_data['arrays'] = arrays
        
        # Update the beamforming system
        if current_array_name in beamforming_system.arrays:
            beamforming_system.arrays[current_array_name].steering_angle = np.radians(steering_angle)
            beamforming_system.arrays[current_array_name].update_steering_angle()
        
        # Recalculate visualizations
        result = beamforming_system.calculate_wave_field()
        vis_data = {
            'wave_field': result['wave_field'].tolist(),
            'beam_profile': result['beam_profile'].tolist(),
            'polar_power': result['polar_power'].tolist(),
            'frame': result['frame']
        }
        
        return vis_data, state_data
        
    except Exception as e:
        print(f"Error updating steering angle: {e}")
        return dash.no_update, dash.no_update