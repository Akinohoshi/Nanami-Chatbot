import streamlit as st
import random
import json
import requests

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(
    page_title="Project Nanami ☕",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #F4EBDD;
    color: #3B2920;
}

/* Main title */
h1 {
    text-align: center;
    color: #8B6B3F;
}

/* Subtitle / general text */
.stApp p,
.stApp span,
.stApp label {
    color: #3B2920;
}

/* Chat messages */
.stChatMessage {
    border-radius: 12px;
    padding: 10px;
    background-color: #EFE1CF;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #5A4032;
}

[data-testid="stSidebar"] * {
    color: #F8EEDC;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    background-color: #8B6B3F;
    color: #FFF8EA;
    border: none;
}

.stButton > button:hover {
    background-color: #6F503A;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #E8D8C3;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Nanami's sidebar
# -------------------------

with st.sidebar:
    st.markdown("# ☕ Kento Nanami")

    st.markdown("### Grade 1 Sorcerer")

    st.divider()

    st.markdown("**STATUS**")
    st.success("On duty")

    st.markdown("**SYSTEM**")
    st.write("🧠 Baby Brain: Online")
    st.write("🤖 AI Brain: Online")

    st.divider()

    st.markdown(
        """
        **Specialties**
        
        ☕ Coffee  
        📋 Work  
        😐 Dealing with Gojo  
        """
    )

    st.divider()

    if st.button("🗑️ Clear conversation"):
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Very well. What is it?"
            }
        ]
        st.rerun()

st.markdown("""
# ☕ Project Nanami

<div style="text-align: center; font-size: 18px;">
<i>"Overtime is not a virtue."</i>
</div>

---
""", unsafe_allow_html=True)

# -------------------------
# Baby Nanami's training
# -------------------------

messages = [
    # Greetings
    "hello", "hi", "hey", "good morning", "good afternoon",

    # Goodbyes
    "bye", "goodbye", "see you later", "I have to go",

    # Coffee
    "coffee", "I need coffee", "want some coffee",
    "give me caffeine", "let's get coffee", "I want caffeine",

    # Jujutsu
    "Jujutsu High", "The school is fun", "want to play",
    "tell me about Jujutsu", "what is Jujutsu High",

    # Gojo
    "Have you met Gojo", "Gojo is annoying", "Where is Gojo",
    "What did Gojo do", "Tell me about Gojo",

    # Work
    "I have work", "too much work", "I hate working",
    "work is exhausting", "I need to finish my work"
]

labels = [
    "greeting", "greeting", "greeting", "greeting", "greeting",

    "goodbye", "goodbye", "goodbye", "goodbye",

    "coffee", "coffee", "coffee",
    "coffee", "coffee", "coffee",

    "jujutsu", "jujutsu", "jujutsu",
    "jujutsu", "jujutsu",

    "gojo", "gojo", "gojo",
    "gojo", "gojo",

    "work", "work", "work",
    "work", "work"
]


# -------------------------
# Train Baby Nanami
# -------------------------

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(messages)

model = MultinomialNB()
model.fit(X, labels)


# -------------------------
# Nanami's memory
# -------------------------

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Very well. What is it?"
        }
    ]


# Show the conversation
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# -------------------------
# API Code
# -------------------------
def ask_nanami(message):
    api_key = st.secrets["OPENROUTER_API_KEY"]

    # Nanami's personality
    ai_messages = [
        {
            "role": "system",
            "content": """
You are an AI assistant inspired by Kento Nanami from Jujutsu Kaisen.
You are calm, professional, practical, concise, and occasionally dry.
Do not claim to literally be Kento Nanami.

Use the conversation history to understand references such as
"him", "her", "that", or "they".

Keep responses appropriate and conversational.
"""
        }
    ]

    # Give Nanami the conversation memory
    for chat in st.session_state.chat_history[-10:]:
        ai_messages.append({
            "role": chat["role"],
            "content": chat["content"]
        })

    # Add the newest message
    ai_messages.append({
        "role": "user",
        "content": message
    })

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "openrouter/free",
            "messages": ai_messages
        },
        timeout=30
    )

    result = response.json()

    if response.status_code != 200:
        return f"Nanami encountered an API error: {result}"

    return result["choices"][0]["message"]["content"]
# -------------------------
# Chat input
# -------------------------

message = st.chat_input("Talk to Nanami...")


if message:
    # Save your message
    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": message
        }
    )

    with st.chat_message("user"):
        st.write(message)

    # -------------------------
    # Nanami thinks
    # -------------------------

    test_X = vectorizer.transform([message])

    probabilities = model.predict_proba(test_X)[0]

    best_index = probabilities.argmax()
    best_intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    st.write(best_intent)
    st.write(confidence)

    # -------------------------
    # Nanami responds
    # -------------------------

    if confidence < 0.30:
        response = ask_nanami(message)

    elif best_intent == "coffee":
        response = random.choice([
            "Finally, a sensible topic.",
            "Coffee sounds reasonable.",
            "You could have started with that.",
            "Now you're speaking my language."
        ])

    elif best_intent == "jujutsu":
        response = random.choice([
            "Jujutsu High? What about it?",
            "There's always something happening there."
        ])

    elif best_intent == "gojo":
        response = random.choice([
            "You mean Gojo? Unfortunately, yes.",
            "What has he done this time?",
            "If this is about Gojo, I already regret asking."
        ])

    elif best_intent == "greeting":
        response = random.choice([
            "Hello.",
            "You're back.",
            "Hm. What is it?"
        ])

    elif best_intent == "goodbye":
        response = random.choice([
            "Leaving already?",
            "Very well. Take care.",
            "Try not to cause any trouble."
        ])

    elif best_intent == "work":
        response = random.choice([
            "Work is unavoidable.",
            "Then it's best to get it finished.",
            "Complaining won't make it disappear, unfortunately.",
            "Take a break if you need one, then continue."
        ])

    # Save Nanami's response
    st.session_state.chat_history.append(
        {
            "role": "assistant",
            "content": response
        }
    )

    # Show Nanami's response
    with st.chat_message("assistant"):
        st.write(response)
