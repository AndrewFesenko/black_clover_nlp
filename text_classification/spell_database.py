import json
import pandas as pd

# Function to load spells from the JSONL file
def load_spell_data(filepath):
    spells = []
    with open(filepath, 'r') as f:
        for line in f:
            spells.append(json.loads(line))
    return pd.DataFrame(spells)

# Function to filter spells based on name or type
def filter_spells(spell_data, search_term=None, spell_type=None):
    filtered_data = spell_data
    
    # Apply search filter if a search term is provided
    if search_term:
        filtered_data = filtered_data[filtered_data['spell_name'].str.contains(search_term, case=False, na=False)]
    
    # Apply type filter if provided and not "All"
    if spell_type and spell_type != "All":
        filtered_data = filtered_data[filtered_data['spell_type'] == spell_type]

    return filtered_data

if __name__ == "__main__":
    # Load spell data and print the first 5 rows for verification
    spell_data = load_spell_data('C:/Users/Andrew/Documents/AI Series/data/spell.jsonl')
    print(spell_data.head())