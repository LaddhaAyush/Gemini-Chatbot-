import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    return tokenizer, model

tokenizer, model = load_model()

st.title("🤖 Simple Hugging Face Chatbot")

if "chat_history_ids" not in st.session_state:
    st.session_state.chat_history_ids = None
if "past" not in st.session_state:
    st.session_state.past = []
if "generated" not in st.session_state:
    st.session_state.generated = []

user_input = st.text_input("You:", key="input")

if st.button("Send") and user_input:
    new_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    bot_input_ids = (
        torch.cat([st.session_state.chat_history_ids, new_input_ids], dim=-1)
        if st.session_state.chat_history_ids is not None
        else new_input_ids
    )

    st.session_state.chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(
        st.session_state.chat_history_ids[:, bot_input_ids.shape[-1]:][0],
        skip_special_tokens=True
    )

    st.session_state.past.append(user_input)
    st.session_state.generated.append(response)

if st.session_state.generated:
    for i in range(len(st.session_state.generated)-1, -1, -1):
        st.markdown(f"**You:** {st.session_state.past[i]}")
        st.markdown(f"**Bot:** {st.session_state.generated[i]}")

st.write("Type your message and click 'Send'. Type 'quit' to reset the conversation.")

if st.button("Reset Conversation"):
    st.session_state.chat_history_ids = None
    st.session_state.past = []
    st.session_state.generated = []