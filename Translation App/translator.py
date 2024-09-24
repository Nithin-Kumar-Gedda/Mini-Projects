import tkinter as tk
from tkinter import ttk
from googletrans import Translator

def translate_text():
    # Get the input text and target language
    text_to_translate = text_input.get("1.0", tk.END).strip()  # Get text from the text area
    target_language = language_combo.get()  # Get selected language
    
    if text_to_translate and target_language:
        translator = Translator()
        # Translate the text
        translation = translator.translate(text_to_translate, dest=target_language)
        translated_output.delete("1.0", tk.END)  # Clear previous output
        translated_output.insert(tk.END, translation.text)  # Insert new translation

# Create the main application window
app = tk.Tk()
app.title("Translation App")
app.geometry("600x600")  # Width x Height
app.config(bg="#f0f8ff")  # Light blue background

language_dict = {
    'te': 'Telugu',
    'hi': 'Hindi',
    'en': 'English',
    'fr': 'French',
    'es': 'Spanish',
    'de': 'German',
    'ja': 'Japanese',
    'zh-CN': 'Chinese'
}

language_names = list(language_dict.values())

# Create a label and text input for the text to translate
tk.Label(app, text="Enter text to translate:", bg="#f0f8ff", font=("Arial", 12), fg="#333").pack(pady=10)  # Label in English
text_input = tk.Text(app, height=5, width=40, bg="#ffffff", fg="#000000", font=("Arial", 10))  # Text area for input
text_input.pack(pady=10)

# Create a dropdown for language selection
tk.Label(app, text="Select target language (e.g.,'telugu','hindi','english')", bg="#f0f8ff", font=("Arial", 12), fg="#333").pack(pady=10)  # Label in English
language_combo = ttk.Combobox(app, values=language_names) 
language_combo.pack(pady=10)
language_combo.config(font=("Arial", 10))

# Create a button to trigger translation
translate_button = tk.Button(app, text="Translate", command=translate_text, bg="#4caf50", fg="white", font=("Arial", 12))  # Button in English
translate_button.pack(pady=20)

# Create a label and text area for displaying the translated output
tk.Label(app, text="Translated text:", bg="#f0f8ff", font=("Arial", 12), fg="#333").pack(pady=10)  # Label in English
translated_output = tk.Text(app, height=5, width=40, bg="#ffffff", fg="#000000", font=("Arial", 10))  # Text area for output
translated_output.pack(pady=10)

# Start the Tkinter event loop
app.mainloop()
