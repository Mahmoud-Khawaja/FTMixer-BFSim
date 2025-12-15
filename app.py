from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc

app = Dash(__name__, 
           external_stylesheets=[dbc.themes.BOOTSTRAP],
           suppress_callback_exceptions=True)

# Add custom CSS for modern styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .nav-card:hover {
                transform: translateY(-8px) !important;
                border-color: rgba(99, 102, 241, 0.5) !important;
                box-shadow: 0 20px 40px rgba(99, 102, 241, 0.2) !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Modern color scheme
COLORS = {
    'primary': '#6366f1',
    'primary_dark': '#4f46e5',
    'secondary': '#8b5cf6',
    'background': '#0f172a',
    'surface': '#1e293b',
    'text': '#f1f5f9',
    'text_secondary': '#94a3b8'
}

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
], style={
    'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    'margin': 0,
    'padding': 0,
    'minHeight': '100vh',
    'background': f'linear-gradient(135deg, {COLORS["background"]} 0%, #1a2332 100%)'
})

# Import your pages
from pages import ft_page, bt_page

@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/ft':
        return ft_page.layout
    elif pathname == '/bt':
        return bt_page.layout
    else:
        return html.Div([
            # Header
            html.Div([
                html.H1("Welcome", style={
                    'fontSize': '3.5rem',
                    'fontWeight': '700',
                    'background': f'linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["secondary"]} 100%)',
                    'WebkitBackgroundClip': 'text',
                    'WebkitTextFillColor': 'transparent',
                    'marginBottom': '1rem',
                    'letterSpacing': '-0.02em'
                }),
                html.P("Choose your destination", style={
                    'fontSize': '1.25rem',
                    'color': COLORS['text_secondary'],
                    'fontWeight': '400'
                })
            ], style={
                'textAlign': 'center',
                'paddingTop': '8rem',
                'marginBottom': '4rem'
            }),
            
            # Navigation Cards for FT and BT Pages
            html.Div([
                # FT Page Card
                dcc.Link([
                    html.Div([
                        html.Div([
                            html.Div("FT", style={
                                'fontSize': '2rem',
                                'fontWeight': '700',
                                'marginBottom': '0.5rem'
                            }),
                            html.Div("Fourier Transform Mixer", style={
                                'fontSize': '0.95rem',
                                'color': COLORS['text_secondary'],
                                'marginBottom': '1.5rem'
                            }),
                            html.Div("→", style={
                                'fontSize': '1.5rem',
                                'transition': 'transform 0.3s ease'
                            })
                        ])
                    ], style={
                        'background': COLORS['surface'],
                        'padding': '2.5rem',
                        'borderRadius': '1rem',
                        'border': f'1px solid rgba(139, 92, 246, 0.2)',
                        'transition': 'all 0.3s ease',
                        'cursor': 'pointer',
                        'minHeight': '200px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    }, className='nav-card')
                ], href='/ft', style={'textDecoration': 'none', 'color': COLORS['text']}),
                
                # BT Page Card
                dcc.Link([
                    html.Div([
                        html.Div([
                            html.Div("BT", style={
                                'fontSize': '2rem',
                                'fontWeight': '700',
                                'marginBottom': '0.5rem'
                            }),
                            html.Div("Your BT Application", style={
                                'fontSize': '0.95rem',
                                'color': COLORS['text_secondary'],
                                'marginBottom': '1.5rem'
                            }),
                            html.Div("→", style={
                                'fontSize': '1.5rem',
                                'transition': 'transform 0.3s ease'
                            })
                        ])
                    ], style={
                        'background': COLORS['surface'],
                        'padding': '2.5rem',
                        'borderRadius': '1rem',
                        'border': f'1px solid rgba(139, 92, 246, 0.2)',
                        'transition': 'all 0.3s ease',
                        'cursor': 'pointer',
                        'minHeight': '200px',
                        'display': 'flex',
                        'flexDirection': 'column',
                        'justifyContent': 'center'
                    }, className='nav-card')
                ], href='/bt', style={'textDecoration': 'none', 'color': COLORS['text']})
            ], style={
                'display': 'grid',
                'gridTemplateColumns': 'repeat(auto-fit, minmax(280px, 1fr))',
                'gap': '2rem',
                'maxWidth': '800px',
                'margin': '0 auto',
                'padding': '0 2rem'
            }),
            
            # Footer
            html.Div([
                html.P("Built with Dash & Bootstrap", style={
                    'color': COLORS['text_secondary'],
                    'fontSize': '0.875rem'
                })
            ], style={
                'textAlign': 'center',
                'marginTop': '6rem',
                'paddingBottom': '2rem'
            })
        ], style={
            'minHeight': '100vh',
            'color': COLORS['text']
        })

if __name__ == '__main__':
    app.run(debug=True)