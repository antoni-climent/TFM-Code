
import discord
from discord.ext import commands
import pickle
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Here goes the bot token configured in discord developer portal
TOKEN = ""

# Define the intents
intents = discord.Intents.default()
intents.messages = True  # Ensure the bot can read messages
intents.message_content = True

# Initialize the bot with the specified intents
bot = commands.Bot(command_prefix='!', intents=intents)

# Load the model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
tokenizer = AutoTokenizer.from_pretrained("roberta-base")


# Test the modeltraining_roberta_v4
labs = ['Non-Toxic', 'Toxic']


def classify_text(text):
    text = text.content.lower()
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = torch.argmax(logits).item()
    print(f'Text: {text}')
    print(f'Predicted class: {labs[predicted_class_idx]}')

    # If any value in pred is greater than 0.5, return True
    return predicted_class_idx

@bot.event
async def on_ready():
    
    print(f'Bot is ready. Logged in as {bot.user}')

@bot.event
async def on_message(message):
    print("MESSAGE RECEIVED")
    # Prevent the bot from processing its own messages
    if message.author == bot.user:
        return

    # Apply the toxicity filter
    if classify_text(message):
        # Delete the toxic message
        await message.delete()
        # Send a warning to the user
        await message.channel.send(f'{message.author.mention}, your message was deleted because it was deemed toxic.')
    print("MESSAGE PROCESSED")
    # Process other commands
    await bot.process_commands(message)

# Run the bot
bot.run(TOKEN)
