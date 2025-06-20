import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# App-titel
st.set_page_config(page_title="Streamlit Replicate Chatbot", page_icon="💬")

# Replicate-gegevens in de zijbalk
with st.sidebar:
    st.title('💬 Streamlit Replicate Chatbot')
    st.write('Maak chatbots met verschillende LLM-modellen die worden gehost op [Replicate](https://replicate.com/).')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Voer je Replicate API-token in:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api) == 40):
            st.warning('Vul je geldige Replicate API-token in.', icon='⚠️')
            st.markdown("**Nog geen API-token?** Ga naar [Replicate](https://replicate.com) om er een aan te maken.")
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader("Modellen en parameters")
    model = st.selectbox("Selecteer een model", (
        "meta/meta-llama-3-70b-instruct",
        "mistralai/mistral-7b-instruct-v0.2",
        "google-deepmind/gemma-2b-it"
    ), key="model")
    if model == "google-deepmind/gemma-2b-it":
        model = "google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"

    temperature = st.sidebar.slider('temperatuur', min_value=0.01, max_value=5.0, value=0.7, step=0.01,
                                    help="Willekeurigheid van de gegenereerde output")
    if temperature >= 1:
        st.warning('Waarden hoger dan 1 geven creatievere en meer willekeurige antwoorden, maar vergroten de kans op hallucinaties.')
    if temperature < 0.1:
        st.warning('Waarden dichter bij 0 geven meer voorspelbare antwoorden. Aanbevolen beginwaarde is 0.7')

    top_p = st.sidebar.slider('top_p', min_value=0.01, max_value=1.0, value=0.9, step=0.01,
                             help="Top p-percentage van de meest waarschijnlijke tokens voor outputgeneratie")

# Sla LLM-antwoorden op
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Stel mij een vraag."}]

# Toon of wis chatgeschiedenis
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def wis_chatgeschiedenis():
    st.session_state.messages = [{"role": "assistant", "content": "Stel mij een vraag."}]

st.sidebar.button('Wis chatgeschiedenis', on_click=wis_chatgeschiedenis)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Haalt een tokenizer op om te voorkomen dat we te veel tekst naar het model sturen."""
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Geeft het aantal tokens in een prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Functie voor het genereren van een model-antwoord
def genereer_antwoord():
    prompt = []
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")

    prompt.append("<|im_start|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)

    if get_num_tokens(prompt_str) >= 3072:
        st.error("Gesprekslengte te lang. Houd het onder 3072 tokens.")
        st.button('Wis chatgeschiedenis', on_click=wis_chatgeschiedenis, key="wis_chatgeschiedenis")
        st.stop()

    for event in replicate.stream(
        model,
        input={
            "prompt": prompt_str,
            "prompt_template": r"{prompt}",
            "temperature": temperature,
            "top_p": top_p,
        }
    ):
        yield str(event)

# Gebruikersprompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Genereer een nieuw antwoord als het laatste bericht niet van de assistent is
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = genereer_antwoord()
        volledig_antwoord = st.write_stream(response)
    message = {"role": "assistant", "content": volledig_antwoord}
    st.session_state.messages.append(message)
