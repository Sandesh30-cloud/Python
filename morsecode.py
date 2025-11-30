class MorseCode:
    # Morse code dictionary
    morse_dict = {
        'A': '.-', 'B': '-...', 'C': '-.-.', 'D': '-..', 'E': '.', 
        'F': '..-.', 'G': '--.', 'H': '....', 'I': '..', 'J': '.---', 
        'K': '-.-', 'L': '.-..', 'M': '--', 'N': '-.', 'O': '---', 
        'P': '.--.', 'Q': '--.-', 'R': '.-.', 'S': '...', 'T': '-', 
        'U': '..-', 'V': '...-', 'W': '.--', 'X': '-..-', 'Y': '-.--', 
        'Z': '--..', '1': '.----', '2': '..---', '3': '...--', 
        '4': '....-', '5': '.....', '6': '-....', '7': '--...', 
        '8': '---..', '9': '----.', '0': '-----', ' ': '/'
    }

    @classmethod
    def encode(cls, text=""):
        if not text:
            text = "SOS"  # Default
        try:
            return ' '.join(cls.morse_dict[char.upper()] for char in text)
        except KeyError as e:
            print(f"Error: Character '{e.args[0]}' cannot be encoded in Morse code.")
            return None

    @classmethod
    def decode(cls, morse_code=""):
        if not morse_code:
            morse_code = "... --- ..."  # Default
        try:
            # Creates a new dictionary where key and value are swapped.
            reverse_dict = {v: k for k, v in cls.morse_dict.items()}
            return ''.join(reverse_dict[code] for code in morse_code.split())
        except KeyError as e:
            print(f"Error: Morse code '{e.args[0]}' cannot be decoded.")
            return None


if __name__ == "__main__":
    # Example usage
    text = input("Enter text to encode (leave blank for default 'SOS'): ")
    morse_code = MorseCode.encode(text)
    print(f"Morse Code: {morse_code}")

    morse_input = input("Enter Morse code to decode (leave blank for default '... --- ...'): ")
    decoded_text = MorseCode.decode(morse_input)
    print(f"Decoded Text: {decoded_text}")