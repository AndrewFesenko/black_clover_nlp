import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd
from ast import literal_eval
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_loader import load_subtitles_dataset

# Define allowed_characters directly
allowed_characters = {
    # Main characters
    "Asta": ["Asta"],
    "Yuno": ["Yuno"],
    "Noelle Silva": ["Noelle", "Noelle Silva"],

    # Black Bulls
    "Yami Sukehiro": ["Yami", "Yami Sukehiro"],
    "Charmy Pappitson": ["Charmy", "Charmy Pappitson"],
    "Luck Voltia": ["Luck", "Luck Voltia"],
    "Gauche Adlai": ["Gauche", "Gauche Adlai"],
    "Vanessa Enoteca": ["Vanessa", "Vanessa Enoteca"],
    "Finral Roulacase": ["Finral", "Finral Roulacase"],
    "Magna Swing": ["Magna", "Magna Swing"],
    "Henry Legolant": ["Henry", "Henry Legolant"],
    "Grey": ["Grey"],
    "Zora Ideale": ["Zora", "Zora Ideale"],
    "Secre Swallowtail": ["Secre", "Secre Swallowtail"],

    # Golden Dawn
    "William Vangeance": ["William", "William Vangeance", "Vangeance"],
    "Klaus Lunettes": ["Klaus", "Klaus Lunettes"],
    "Mimosa Vermillion": ["Mimosa", "Mimosa Vermillion"],
    "Langris Vaude": ["Langris", "Langris Vaude"],
    "Hamon Caseus": ["Hamon", "Hamon Caseus"],
    "Shiren Tium": ["Shiren", "Shiren Tium"],

    # Silver Eagles
    "Nozel Silva": ["Nozel", "Nozel Silva"],
    "Solid Silva": ["Solid", "Solid Silva"],
    "Nebra Silva": ["Nebra", "Nebra Silva"],

    # Crimson Lion Kings
    "Fuegoleon Vermillion": ["Fuegoleon", "Fuegoleon Vermillion"],
    "Leopold Vermillion": ["Leopold", "Leopold Vermillion"],
    "Mereoleona Vermillion": ["Mereoleona", "Mereoleona Vermillion"],
    "Randall Luftair": ["Randall", "Randall Luftair"],

    # Blue Rose Knights
    "Charlotte Roselei": ["Charlotte", "Charlotte Roselei"],
    "Sol Marron": ["Sol", "Sol Marron"],
    "Puli Angel": ["Puli", "Puli Angel"],

    # Green Praying Mantises
    "Jack the Ripper": ["Jack", "Jack the Ripper"],
    "Sekke Bronzazza": ["Sekke", "Sekke Bronzazza"],

    # Coral Peacocks
    "Dorothy Unsworth": ["Dorothy", "Dorothy Unsworth"],
    "Kirsch Vermillion": ["Kirsch", "Kirsch Vermillion"],
    "En Ringard": ["En", "En Ringard"],

    # Purple Orcas
    "Gueldre Poizot": ["Gueldre", "Gueldre Poizot"],

    # Aqua Deer
    "Rill Boismortier": ["Rill", "Rill Boismortier"],
    "Nils Ragus": ["Nils", "Nils Ragus"],
    "Rubens": ["Rubens"],

    # Eye of the Midnight Sun
    "Patolli": ["Patolli"],
    "Rhya": ["Rhya"],
    "Fana": ["Fana"],
    "Vetto": ["Vetto"],
    "Rades Spirito": ["Rades", "Rades Spirito"],
    "Sally": ["Sally"],
    "Valtos": ["Valtos"],

    # Other Elves
    "Licht": ["Licht"],
    "Tetia": ["Tetia"],
    "Charla": ["Charla"],
    "Drowa": ["Drowa"],

    # Diamond Kingdom
    "Mars": ["Mars"],
    "Ladros": ["Ladros"],
    "Fanzell Kruger": ["Fanzell", "Fanzell Kruger"],
    "Dominante Code": ["Dominante", "Dominante Code"],
    "Lotus Whomalt": ["Lotus", "Lotus Whomalt"],

    # Spade Kingdom and Dark Triad
    "Zenon Zogratis": ["Zenon", "Zenon Zogratis"],
    "Dante Zogratis": ["Dante", "Dante Zogratis"],
    "Vanica Zogratis": ["Vanica", "Vanica Zogratis"],
    "Liebe": ["Liebe"],
    "Nacht Faust": ["Nacht", "Nacht Faust"],
    "Morris": ["Morris"],

    # Spirit Guardians (Heart Kingdom)
    "Gaja": ["Gaja"],
    "Undine": ["Undine"],
    "Lolopechka": ["Lolopechka"],

    # Magic Knights Captains
    "Kaiser Granvorka": ["Kaiser", "Kaiser Granvorka"],

    # Other recurring characters
    "Owen": ["Owen"],
    "Acier Silva": ["Acier", "Acier Silva"],
    "Morgen Faust": ["Morgen", "Morgen Faust"],
    "Damnatio Kira": ["Damnatio", "Damnatio Kira"],

    # Dwarves and mysterious characters
    "Megicula": ["Megicula"],
    "Lucifero": ["Lucifero"],
    "Beelzebub": ["Beelzebub"],
    "Astaroth": ["Astaroth"],

    # Royal Family
    "Ciel Grinberryall": ["Ciel", "Ciel Grinberryall"],
    "Lumiere Silvamillion Clover": ["Lumiere", "Lumiere Silvamillion Clover"],
    "Revchi": ["Revchi"],
    "Heath Grice": ["Heath", "Heath Grice"],
    "Mariella": ["Mariella"],
    "Orsi Orfai": ["Orsi", "Father Orsi", "Orsi Orfai"],
    "Nash": ["Nash"],
    "Rebecca Scarlet": ["Rebecca", "Rebecca Scarlet"],
    "Marie Adlai": ["Marie", "Marie Adlai"],
    "Sister Lily": ["Lily", "Sister Lily"],
    "Baro": ["Baro"],
    "Gordon Agrippa": ["Gordon", "Gordon Agrippa"],
    "Mark": ["Mark"],
    "Kira Clover": ["Kira", "Kira Clover"],
    "Zara Ideale": ["Zara", "Zara Ideale"],
}

# Create a set of character names for easy lookup
character_names = {variant for variants in allowed_characters.values() for variant in variants}

class NamedEntityRecognizer:
    def __init__(self):
        self.nlp_model = self.load_model()

    def load_model(self):
        # Load a pre-trained SpaCy model for NER
        nlp = spacy.load("en_core_web_trf")
        return nlp

    def get_ners_inference(self, script):
        # Tokenize the script into sentences
        script_sentences = sent_tokenize(script)
        ner_output = []

        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text
                    first_name = full_name.split(" ")[0].strip()

                    # Only include characters in the predefined list
                    if first_name in character_names:
                        ners.add(first_name)
            ner_output.append(ners)

        return ner_output

    def get_ners(self, dataset_path, save_path=None):
        # Check if a processed file already exists
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df['ners'] = df['ners'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df

        # Load dataset
        df = load_subtitles_dataset(dataset_path)

        # Process each script to extract entities
        df['ners'] = df['script'].apply(self.get_ners_inference)

        # Save processed data if path provided
        if save_path is not None:
            df.to_csv(save_path, index=False)

        return df
