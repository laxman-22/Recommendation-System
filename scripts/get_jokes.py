import os
import re

def read_clean_joke_text(joke_id, joke_folder='joke'):
    joke_texts = {}

        
    joke_num = joke_id[1:]
    filename = f"init{joke_num}.html"
    filepath = os.path.join(joke_folder, filename)

    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            html = file.read()

            # Extract joke section
            match = re.search(r'<!--begin of joke -->(.*?)<!--end of joke -->', html, re.DOTALL | re.IGNORECASE)
            if match:
                joke_text = match.group(1)

                # Remove <P>, <BR>, and any basic HTML tags
                joke_text = re.sub(r'<.*?>', '', joke_text)

                # Strip extra whitespace
                joke_texts[joke_id] = joke_text.strip()
            else:
                joke_texts[joke_id] = '[No joke text found]'
    except FileNotFoundError:
        print(f"File not found for {joke_id}: {filepath}")
        joke_texts[joke_id] = '[Missing file]'

    return joke_texts

if __name__ == "__main__":
    # Example usage
    joke_ids = 'J7'
    results = read_clean_joke_text(joke_ids)

    for joke_id, text in results.items():
        print(f"\n--- {joke_id} ---\n{text}")
