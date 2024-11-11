import json
import dash
from dash import dcc, html
import pandas as pd
from dash.dependencies import Input, Output

# Load the spell data from JSONL
def load_spell_data(filepath):
    spells = []
    with open(filepath, 'r') as f:
        for line in f:
            spells.append(json.loads(line))
    return pd.DataFrame(spells)

# Load data
spell_data = load_spell_data('C:/Users/Andrew/Documents/AI Series/data/spell.jsonl')

# Initialize Dash app
app = dash.Dash(__name__)

# Layout of the Dash app
app.layout = html.Div([
    html.H1("Interactive Spell Database", style={'text-align': 'center'}),
    
    html.Div([
        html.Label("Select Spell Type"),
        dcc.Dropdown(
            id='spell-type-dropdown',
            options=[{'label': spell_type, 'value': spell_type} for spell_type in spell_data['spell_type'].unique()] + [{'label': 'All', 'value': 'All'}],
            value='All',
            style={'width': '50%'}
        ),
        
        html.Label("Search Spells by Name"),
        dcc.Input(
            id='spell-name-input',
            type='text',
            placeholder='Enter spell name...',
            style={'width': '50%'}
        )
    ], style={'margin-bottom': '20px'}),

    html.Div([
        html.H2("Spell Details"),
        dcc.Graph(id='spell-table')
    ])
])

# Callback for updating the table based on filters
@app.callback(
    Output('spell-table', 'figure'),
    [Input('spell-type-dropdown', 'value'),
     Input('spell-name-input', 'value')]
)
def update_table(spell_type, search_term):
    filtered_data = spell_data

    # Filter by spell type
    if spell_type and spell_type != "All":
        filtered_data = filtered_data[filtered_data['spell_type'] == spell_type]

    # Filter by spell name
    if search_term:
        filtered_data = filtered_data[filtered_data['spell_name'].str.contains(search_term, case=False, na=False)]

    # Create table
    fig = {
        'data': [{
            'type': 'table',
            'header': {
                'values': [['Spell Name'], ['Spell Type'], ['Description']],
                'align': 'left'
            },
            'cells': {
                'values': [
                    filtered_data['spell_name'],
                    filtered_data['spell_type'],
                    filtered_data['spell_description']
                ],
                'align': 'left'
            }
        }]
    }

    return fig

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
