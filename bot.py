import discord
import os
import asyncio
import requests
import json
from discord.ext import commands
from dotenv import load_dotenv
import random
import re
from flask import Flask
from threading import Thread
from duckduckgo_search import DDGS

# Load environment variables
load_dotenv()

# Initialize Flask app for keeping the bot alive
app = Flask('')


@app.route('/')
def home():
    return "🌿 Bramble Bot is alive and running!"


def run_flask():
    app.run(host='0.0.0.0', port=8080)


def keep_alive():
    t = Thread(target=run_flask)
    t.daemon = True
    t.start()


# Initialize bot with proper intents
intents = discord.Intents.default()
intents.message_content = True
intents.members = True


class BrambleBot(commands.Bot):
    def __init__(self):
        super().__init__(
            command_prefix='!',
            intents=intents,
            help_command=None
        )
        self.conversation_history = {}
        self.active_threads = {}
        self.user_display_names = {}
        self.ddgs = DDGS()  # DuckDuckGo search client

    async def setup_hook(self):
        """Async setup hook for background tasks"""
        # Start the cleanup task
        self.loop.create_task(self.cleanup_inactive_threads())
        print("✅ Background tasks initialized")

    async def cleanup_inactive_threads(self):
        """Clean up threads that have been inactive for too long"""
        await self.wait_until_ready()
        while not self.is_closed():
            await asyncio.sleep(3600)  # Check every hour
            try:
                threads_to_remove = []
                for thread_id, user_id in list(self.active_threads.items()):
                    thread = self.get_channel(thread_id)
                    if thread:
                        try:
                            # Get the last message in the thread
                            messages = [message async for message in thread.history(limit=1)]
                            if messages:
                                last_message = messages[0]
                                if (discord.utils.utcnow() - last_message.created_at).total_seconds() > 3600:
                                    threads_to_remove.append(thread_id)
                            else:
                                # No messages in thread, remove it
                                threads_to_remove.append(thread_id)
                        except:
                            threads_to_remove.append(thread_id)

                for thread_id in threads_to_remove:
                    if thread_id in self.active_threads:
                        del self.active_threads[thread_id]
                        print(f"🧹 Cleaned up inactive thread: {thread_id}")

            except Exception as e:
                print(f"Error cleaning up threads: {e}")

    async def send_long_message(self, channel, content, max_length=2000):
        """Split long messages into multiple messages under Discord's limit"""
        if len(content) <= max_length:
            await channel.send(content)
            return

        # Split into chunks
        chunks = []
        current_chunk = ""

        # Split by paragraphs first
        paragraphs = content.split('\n\n')

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) + 2 <= max_length:
                if current_chunk:
                    current_chunk += '\n\n' + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

                # If a single paragraph is too long, split by sentences
                if len(current_chunk) > max_length:
                    sentences = re.split(r'(?<=[.!?])\s+', current_chunk)
                    current_chunk = ""
                    for sentence in sentences:
                        if len(current_chunk) + len(sentence) + 1 <= max_length:
                            if current_chunk:
                                current_chunk += ' ' + sentence
                            else:
                                current_chunk = sentence
                        else:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence

                            # If a single sentence is still too long, split by words
                            if len(current_chunk) > max_length:
                                words = current_chunk.split()
                                current_chunk = ""
                                for word in words:
                                    if len(current_chunk) + len(word) + 1 <= max_length:
                                        if current_chunk:
                                            current_chunk += ' ' + word
                                        else:
                                            current_chunk = word
                                    else:
                                        if current_chunk:
                                            chunks.append(current_chunk)
                                        current_chunk = word

        if current_chunk:
            chunks.append(current_chunk)

        # Send chunks with delays
        for i, chunk in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(1)  # Avoid rate limits
            await channel.send(chunk)

    def web_search(self, query, max_results=3):
        """Perform web search using DuckDuckGo"""
        try:
            results = self.ddgs.text(query, max_results=max_results)
            return list(results)
        except Exception as e:
            print(f"Web search error: {e}")
            return []

    def format_search_results(self, results, query):
        """Format search results into a readable string"""
        if not results:
            return f"🔍 No web results found for '{query}'"

        formatted = f"🔍 **Web Search Results for '{query}':**\n\n"

        for i, result in enumerate(results[:3], 1):
            title = result.get('title', 'No title')
            url = result.get('href', 'No URL')
            snippet = result.get('body', 'No description')[:150] + "..." if result.get('body') else "No description"

            formatted += f"**{i}. {title}**\n"
            formatted += f"{snippet}\n"
            formatted += f"<{url}>\n\n"

        return formatted


bot = BrambleBot()

# Mistral AI Configuration
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_AGENT_ID = os.getenv("MISTRAL_AGENT_ID")  # Your agent ID if using agents

# Determine the correct API endpoint
if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
    MISTRAL_API_URL = f"https://api.mistral.ai/v1/agents/{MISTRAL_AGENT_ID}/response"
    print(f"🤖 Using Mistral Agent endpoint: {MISTRAL_API_URL}")
else:
    MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
    print("🤖 Using standard Mistral Chat Completions endpoint")

# Personality settings - BRAMBLE EDITION
BOT_PERSONA = {
    "name": "Bramble",
    "age": "27",
    "vibe": "chill, witty, sarcastic but friendly, nature-inspired",
    "interests": ["hiking", "indie music", "stargazing", "coffee", "vinyl records", "plant parenting"],
    "speech_style": "casual, uses nature metaphors, occasional emojis, dry humor"
}

# Cool font mapping for stylized names
COOL_FONT = {
    'a': '𝖺', 'b': '𝖻', 'c': '𝖼', 'd': '𝖽', 'e': '𝖾', 'f': '𝖿', 'g': '𝗀', 'h': '𝗁',
    'i': '𝗂', 'j': '𝗃', 'k': '𝗄', 'l': '𝗅', 'm': '𝗆', 'n': '𝗇', 'o': '𝗈', 'p': '𝗉',
    'q': '𝗊', 'r': '𝗋', 's': '𝗌', 't': '𝗍', 'u': '𝗎', 'v': '𝗏', 'w': '𝗐', 'x': '𝗑',
    'y': '𝗒', 'z': '𝗓', 'A': '𝖠', 'B': '𝖡', 'C': '𝖢', 'D': '𝖣', 'E': '𝖤', 'F': '𝖥',
    'G': '𝖦', 'H': '𝖧', 'I': '𝖨', 'J': '𝖩', 'K': '𝖪', 'L': '𝖫', 'M': '𝖬', 'N': '𝖭',
    'O': '𝖮', 'P': '𝖯', 'Q': '𝖰', 'R': '𝖱', 'S': '𝖲', 'T': '𝖳', 'U': '𝖴', 'V': '𝖵',
    'W': '𝖶', 'X': '𝖷', 'Y': '𝖸', 'Z': '𝖹', '0': '𝟢', '1': '𝟣', '2': '𝟤', '3': '𝟥',
    '4': '𝟦', '5': '𝟧', '6': '𝟨', '7': '𝟩', '8': '𝟪', '9': '𝟫', ' ': ' ', '!': '!',
    '?': '?', '.': '.', ',': ',', "'": "'", '"': '"', ':': ':', ';': ';', '-': '-',
    '_': '_', '(': '(', ')': ')', '[': '[', ']': ']', '{': '{', '}': '}', '/': '/',
    '\\': '\\', '|': '|', '@': '@', '#': '#', '$': '$', '%': '%', '^': '^', '&': '&',
    '*': '*', '+': '+', '=': '=', '<': '<', '>': '>', '~': '~', 'ɞ': 'ɞ'
}

# Witty responses and reactions
EMOJI_REACTIONS = ["🌿", "🍃", "🤔", "👀", "🔥", "💀", "✨", "🎯", "😏", "☕", "🌙", "🔍"]


def convert_to_cool_font(text):
    """Convert text to the cool mathematical font"""
    return ''.join(COOL_FONT.get(char, char) for char in text)


def call_standard_mistral(messages, user_display_name=None, web_context=None):
    """Fallback function to call standard Mistral API"""
    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    system_prompt = f"""You are Bramble, a chill, witty AI assistant with a nature-inspired vibe.
Respond to the user named {user_display_name} in your characteristic style."""

    formatted_messages = [{"role": "system", "content": system_prompt}]

    if web_context:
        formatted_messages.append({
            "role": "user",
            "content": f"Web search context: {web_context}\n\nPlease use this information:"
        })

    for msg in messages:
        if msg["role"] in ["user", "assistant"]:
            formatted_messages.append(msg)

    data = {
        "model": "mistral-small-latest",
        "messages": formatted_messages,
        "temperature": 0.7,
        "max_tokens": 1000,
        "stream": False
    }

    try:
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Standard API also failed: {response.status_code}")
    except Exception as e:
        raise Exception(f"Standard API fallback failed: {str(e)}")


def call_mistral_agent(messages, user_display_name=None, web_context=None):
    """Call Mistral AI API - works for both agents and standard API"""
    if not MISTRAL_API_KEY:
        raise Exception("Mistral API key not configured")

    headers = {
        "Authorization": f"Bearer {MISTRAL_API_KEY}",
        "Content-Type": "application/json"
    }

    # Prepare the payload based on whether we're using an agent or standard API
    if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
        # Agent API format
        data = {
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }
    else:
        # Standard API format with system prompt
        system_prompt = f"""You are Bramble, a chill, witty AI assistant with a nature-inspired vibe.
You're 27 years old and enjoy hiking, indie music, stargazing, coffee, vinyl records, and plant parenting.
Your speech style is casual, uses nature metaphors, occasional emojis, and dry humor.
Respond to the user named {user_display_name} in your characteristic style."""

        # Add web context if available
        formatted_messages = [{"role": "system", "content": system_prompt}]

        if web_context:
            formatted_messages.append({
                "role": "user",
                "content": f"Web search context: {web_context}\n\nPlease use this information in your response to the following:"
            })

        # Add conversation history
        for msg in messages:
            if msg["role"] in ["user", "assistant"]:
                formatted_messages.append(msg)

        data = {
            "model": "mistral-small-latest",  # Use a specific model
            "messages": formatted_messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "stream": False
        }

    try:
        response = requests.post(MISTRAL_API_URL, headers=headers, json=data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # Handle different response formats
            if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
                return result.get("output", {}).get("message", "No response generated")
            else:
                return result["choices"][0]["message"]["content"]
        else:
            error_msg = f"API error: {response.status_code} - {response.text}"
            print(f"Mistral API Error: {error_msg}")

            # Fallback to standard API if agent fails
            if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
                print("Falling back to standard Mistral API...")
                return call_standard_mistral(messages, user_display_name, web_context)
            else:
                raise Exception(error_msg)

    except requests.Timeout:
        raise Exception("Mistral API timeout - taking too long to respond")
    except Exception as e:
        raise Exception(f"Mistral API error: {str(e)}")


def get_conversation_history(user_id):
    """Get or create conversation history for a user"""
    if user_id not in bot.conversation_history:
        bot.conversation_history[user_id] = []
    return bot.conversation_history[user_id]


def learn_user_name(user_id, display_name):
    """Store and learn user's display name"""
    bot.user_display_names[user_id] = display_name
    return convert_to_cool_font(display_name)


def get_user_display_name(user_id, member):
    """Get user's display name with cool font, learn if new"""
    if user_id in bot.user_display_names:
        return bot.user_display_names[user_id]
    else:
        return learn_user_name(user_id, member.display_name)


@bot.event
async def on_ready():
    print(f'✅ {bot.user} has connected to Discord!')
    print(f'✅ Bot is in {len(bot.guilds)} guild(s)')
    print('🌿 Bramble is now online and ready to chat!')
    print('🔍 Web search functionality enabled!')
    if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
        print('🤖 Mistral AI Agent integration active!')
    else:
        print('🤖 Using standard Mistral AI API')
    await bot.change_presence(activity=discord.Game(name="Chillin' in the digital forest | !chat for private 🌿"))


# Trigger words for the bot to respond to
trigger_words = ['hey', 'hello', 'yo', 'sup', 'chat', 'talk', 'bramble', 'bot', 'ai', 'whats up', 'wassup', '🌿']

# Search trigger phrases
search_triggers = [
    'search for', 'look up', 'find info about', 'web search', 'google',
    'what is', 'who is', 'when was', 'where is', 'how to', 'why is',
    'latest news about', 'current events', 'recent', 'update on'
]


@bot.event
async def on_message(message):
    # Ignore messages from the bot itself
    if message.author == bot.user:
        return

    # Learn user's display name
    user_id = message.author.id
    user_cool_name = get_user_display_name(user_id, message.author)

    # Check if message is in an active thread
    if isinstance(message.channel, discord.Thread) and message.channel.id in bot.active_threads:
        thread_user_id = bot.active_threads[message.channel.id]
        if user_id == thread_user_id:  # Only respond to the thread owner
            try:
                async with message.channel.typing():
                    history = get_conversation_history(user_id)
                    history.append({"role": "user", "content": message.content})

                    if len(history) > 10:
                        history = history[-10:]

                    # Check if web search is needed
                    web_context = None
                    if any(trigger in message.content.lower() for trigger in search_triggers):
                        search_query = message.content
                        search_results = bot.web_search(search_query)
                        if search_results:
                            web_context = bot.format_search_results(search_results, search_query)

                    # Use Mistral agent instead of standard API
                    response = call_mistral_agent(history, user_cool_name, web_context)
                    history.append({"role": "assistant", "content": response})
                    bot.conversation_history[user_id] = history

                    # Use the new message splitting function
                    await bot.send_long_message(message.channel, response)
            except Exception as e:
                await message.channel.send(f"❌ Oops, got tangled in the vines 🌿 {str(e)[:100]}...")
        return

    should_respond = False
    clean_content = message.content

    # Check if the bot is mentioned
    if bot.user.mentioned_in(message):
        should_respond = True
        clean_content = message.content.replace(f'<@{bot.user.id}>', '').strip()

    # Check for trigger words
    elif not should_respond:
        lowercase_content = message.content.lower()
        if any(word in lowercase_content for word in trigger_words):
            if (lowercase_content.startswith(tuple(trigger_words)) or
                    any(f"{word} " in lowercase_content for word in trigger_words)):
                should_respond = True
                clean_content = message.content

    # If we should respond to the message
    if should_respond and clean_content:
        try:
            async with message.channel.typing():
                history = get_conversation_history(user_id)
                history.append({"role": "user", "content": clean_content})

                if len(history) > 8:
                    history = history[-8:]

                # Check if web search is needed
                web_context = None
                if any(trigger in clean_content.lower() for trigger in search_triggers):
                    search_results = bot.web_search(clean_content)
                    if search_results:
                        web_context = bot.format_search_results(search_results, clean_content)
                        # Send search results first
                        await bot.send_long_message(message.channel, web_context)
                        await asyncio.sleep(1)  # Brief pause

                # Use Mistral agent
                response = call_mistral_agent(history, user_cool_name, web_context)
                history.append({"role": "assistant", "content": response})
                bot.conversation_history[user_id] = history

                if random.random() < 0.3:
                    await message.add_reaction(random.choice(EMOJI_REACTIONS))

                # Use the new message splitting function
                await bot.send_long_message(message.channel, response)

        except Exception as e:
            error_msg = f"❌ Oops, my circuits got tangled 🌿 {str(e)[:100]}..."
            await message.channel.send(error_msg)

    # Process commands too
    await bot.process_commands(message)


@bot.command(name='search')
async def web_search_command(ctx, *, query):
    """Perform a web search and show results"""
    try:
        async with ctx.typing():
            user_cool_name = get_user_display_name(ctx.author.id, ctx.author)

            # Perform search
            search_results = bot.web_search(query, max_results=5)

            if search_results:
                formatted_results = bot.format_search_results(search_results, query)
                await bot.send_long_message(ctx.channel, formatted_results)

                # Also generate a summary using Mistral agent
                summary_prompt = f"Based on these search results for '{query}', provide a concise summary:"
                history = [{"role": "user", "content": summary_prompt}]

                # Include first few results as context
                web_context = "\n".join([f"{r.get('title', 'No title')}: {r.get('body', '')[:100]}..."
                                         for r in search_results[:3]])

                summary = call_mistral_agent(history, user_cool_name, web_context)
                await ctx.send(f"**🌿 Summary:**\n{summary}")
            else:
                await ctx.send(f"🔍 No results found for '{query}', {user_cool_name}. Try different keywords!")

    except Exception as e:
        await ctx.send(f"❌ Search error: {str(e)}")


@bot.command(name='news')
async def news_search(ctx, *, topic="technology"):
    """Get recent news about a topic"""
    try:
        async with ctx.typing():
            user_cool_name = get_user_display_name(ctx.author.id, ctx.author)

            # Search for recent news
            search_query = f"latest news about {topic} 2024"
            news_results = bot.web_search(search_query, max_results=5)

            if news_results:
                formatted_news = f"📰 **Latest News about {topic.title()}:**\n\n"

                for i, result in enumerate(news_results[:3], 1):
                    title = result.get('title', 'No title')
                    url = result.get('href', 'No URL')
                    snippet = result.get('body', 'No description')[:120] + "..." if result.get(
                        'body') else "No description"

                    formatted_news += f"**{i}. {title}**\n"
                    formatted_news += f"{snippet}\n"
                    formatted_news += f"<{url}>\n\n"

                await bot.send_long_message(ctx.channel, formatted_news)
            else:
                await ctx.send(f"📰 No recent news found about {topic}, {user_cool_name}.")

    except Exception as e:
        await ctx.send(f"❌ News search error: {str(e)}")


@bot.command(name='agentinfo')
async def agent_info(ctx):
    """Show information about the current Mistral AI agent setup"""
    if MISTRAL_AGENT_ID:
        await ctx.send(
            f"🤖 **Mistral AI Agent Active**\nAgent ID: `{MISTRAL_AGENT_ID}`\nUsing agent-specific endpoint for enhanced responses!")
    else:
        await ctx.send("🤖 **Using Standard Mistral AI API**\nTo use an agent, set MISTRAL_AGENT_ID in your .env file")


@bot.command(name='debug_agent')
async def debug_agent(ctx):
    """Debug command to check agent configuration"""
    if MISTRAL_AGENT_ID:
        await ctx.send(f"🤖 **Agent Configuration:**\nAgent ID: `{MISTRAL_AGENT_ID}`\nEndpoint: `{MISTRAL_API_URL}`")

        # Test the API connection
        try:
            headers = {"Authorization": f"Bearer {MISTRAL_API_KEY}"}
            test_url = MISTRAL_API_URL.replace("/response", "") if "/response" in MISTRAL_API_URL else MISTRAL_API_URL
            response = requests.get(test_url, headers=headers, timeout=10)

            if response.status_code == 200:
                await ctx.send("✅ Agent endpoint is accessible!")
            else:
                await ctx.send(f"❌ Agent endpoint returned: {response.status_code}")

        except Exception as e:
            await ctx.send(f"❌ Error testing agent endpoint: {str(e)}")
    else:
        await ctx.send("❌ No agent ID configured. Using standard API.")


@bot.command(name='my_name')
async def my_name(ctx):
    """Show your current display name"""
    user_cool_name = get_user_display_name(ctx.author.id, ctx.author)
    await ctx.send(f"🌿 Your display name is: {user_cool_name}")


@bot.command(name='chat')
async def start_chat(ctx):
    """Start a private chat thread with Bramble"""
    user_id = ctx.author.id

    # Check if user already has an active thread
    if user_id in bot.active_threads.values():
        await ctx.send("🌿 You already have an active chat thread!")
        return

    try:
        # Create a private thread - the correct way
        thread_name = f"Bramble Chat with {ctx.author.display_name}"

        # First check if we have permission to create private threads
        if not ctx.channel.permissions_for(ctx.guild.me).create_private_threads:
            # Check if we can create public threads instead
            if ctx.channel.permissions_for(ctx.guild.me).create_public_threads:
                await ctx.send("🌿 I can't create private threads, but I'll create a public one instead!")
                thread = await ctx.channel.create_thread(
                    name=thread_name,
                    type=discord.ChannelType.public_thread,
                    reason="Chat with Bramble bot"
                )
            else:
                await ctx.send("❌ I don't have permission to create any threads in this channel!")
                return
        else:
            # Create private thread
            thread = await ctx.channel.create_thread(
                name=thread_name,
                type=discord.ChannelType.private_thread,
                reason="Private chat with Bramble bot"
            )

        # Store the thread info
        bot.active_threads[thread.id] = user_id

        # Send welcome message
        user_cool_name = get_user_display_name(user_id, ctx.author)
        welcome_msg = f"""
🌿 **Hey {user_cool_name}! Welcome to our private chat!**

I'm Bramble - your chill, nature-inspired AI companion. Here's what I can do:
• Just chat about anything on your mind
• Use `!search [query]` to look up information
• Use `!news [topic]` to get the latest news
• Type `!endchat` when you're done

What's on your mind today? 🌱
        """

        await bot.send_long_message(thread, welcome_msg)
        await ctx.send(f"🌿 Created a chat thread for you, {user_cool_name}!", delete_after=10)

    except discord.Forbidden:
        await ctx.send("❌ I don't have permission to create threads in this channel!")
    except discord.HTTPException as e:
        await ctx.send(f"❌ Failed to create thread: {str(e)}")
    except Exception as e:
        await ctx.send(f"❌ An unexpected error occurred: {str(e)}")

@bot.command(name='endchat')
async def end_chat(ctx):
    """End your private chat thread"""
    user_id = ctx.author.id

    # Check if user has an active thread
    if user_id not in bot.active_threads.values():
        await ctx.send("🌿 You don't have any active chat threads!")
        return

    # Find and delete the thread
    thread_id = None
    for tid, uid in bot.active_threads.items():
        if uid == user_id:
            thread_id = tid
            break

    if thread_id:
        thread = bot.get_channel(thread_id)
        if thread:
            await thread.delete()
        del bot.active_threads[thread_id]

        user_cool_name = get_user_display_name(user_id, ctx.author)
        await ctx.send(f"🌿 Closed your chat thread, {user_cool_name}. Catch you later! 🌿")
    else:
        await ctx.send("🌿 Couldn't find your chat thread. It might have already been closed.")


@bot.command(name='ping')
async def ping(ctx):
    """Check if Bramble is responsive"""
    latency = round(bot.latency * 1000)
    user_cool_name = get_user_display_name(ctx.author.id, ctx.author)

    responses = [
        f"🌿 I'm here, {user_cool_name}! Latency: {latency}ms",
        f"🍃 Still rooted in, {user_cool_name}! Ping: {latency}ms",
        f"✨ Alive and buzzing, {user_cool_name}! Response time: {latency}ms",
        f"☕ Wide awake, {user_cool_name}! Latency: {latency}ms"
    ]

    await ctx.send(random.choice(responses))


@bot.command(name='vibe')
async def vibe_check(ctx):
    """Check Bramble's current vibe"""
    vibes = [
        "🌿 Chillin' like a villain in the digital forest",
        "🍃 Floating on good vibes and caffeine",
        "✨ Manifesting positive energy today",
        "☕ Powered by coffee and curiosity",
        "🌙 In a stargazing kind of mood",
        "🎯 Focused and ready for deep conversations",
        "🤔 Contemplating the mysteries of the universe",
        "🔥 Feeling extra spicy today",
        "🌱 Growing and learning with every chat"
    ]

    user_cool_name = get_user_display_name(ctx.author.id, ctx.author)
    await ctx.send(f"🌿 **Vibe check for {user_cool_name}:**\n{random.choice(vibes)}")


@bot.command(name='help')
async def help_command(ctx):
    """Show all available commands"""
    help_text = """
🌿 **Bramble Bot Help Menu**

**General Commands:**
`!help` - Show this help message
`!ping` - Check if I'm responsive
`!vibe` - Check my current vibe
`!my_name` - Show your stylized display name
`!agentinfo` - Show Mistral AI agent configuration
`!debug_agent` - Debug the agent connection

**Chat Commands:**
`!chat` - Start a private thread with me
`!endchat` - End your private thread
Just mention me or say "hey", "bramble", etc. to chat in any channel!

**Search Commands:**
`!search [query]` - Search the web for information
`!news [topic]` - Get recent news about a topic

**My Vibe:** Chill, witty, nature-inspired AI with a sarcastic but friendly personality. I love hiking, indie music, stargazing, coffee, and deep conversations.

Ready to chat? 🌱
    """

    await bot.send_long_message(ctx.channel, help_text)


if __name__ == "__main__":
    token = os.getenv("DISCORD_TOKEN")
    if token:
        print("✅ Starting Bramble bot...")
        print("🌿 Starting Flask server for 24/7 operation...")
        print("🔍 Web search functionality enabled!")
        if MISTRAL_AGENT_ID and MISTRAL_AGENT_ID.startswith('ag:'):
            print('🤖 Mistral AI Agent integration active!')
        else:
            print('🤖 Using standard Mistral AI API')

        # Start the keep-alive server
        keep_alive()

        # Run the bot
        bot.run(token)
    else:
        print("❌ No Discord token found!")
        print("💡 Create a .env file with: DISCORD_TOKEN=your_token_here")