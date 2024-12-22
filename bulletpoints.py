import pyperclip

def add_bullets():
    # Ask user what symbol they want to use as bullet point
    bullet_symbol = input("Enter the bullet point symbol (default is *): ") or "*"
    
    # Ask if they want to add spaces after the bullet
    spaces = input("How many spaces after the bullet? (default is 1): ") or "1"
    spaces = " " * int(spaces)
    
    try:
        # Get text from user instead of clipboard
        print("\nEnter your text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
        
        # Add bullets to each line
        formatted_text = "\n".join(f"{bullet_symbol}{spaces}{line}" for line in lines)
        
        # Ask user if they want to copy to clipboard
        copy_choice = input("\nDo you want to copy the result to clipboard? (yes/no): ").lower()
        
        # Show the result
        print("\nFormatted text:")
        print(formatted_text)
        
        # Copy to clipboard if user wants
        if copy_choice.startswith('y'):
            pyperclip.copy(formatted_text)
            print("\nText has been copied to clipboard!")
            
    except Exception as e:
        print(f"An error occurred: {e}")

# Run the program
if __name__ == "__main__":
    add_bullets()