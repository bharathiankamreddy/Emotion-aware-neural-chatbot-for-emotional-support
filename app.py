try:
    import streamlit as st
    import sklearn as sk
    import pandas as pd
    import numpy as np
    import altair as alt
    import joblib
except ModuleNotFoundError as e:
    print(f"Error: {e}")
    print("Please make sure to install the required modules.")
    exit()

# Rest of your code
# ...
pipe_lr = joblib.load("text_emotion.pkl")

emotions_emoji_dict = {
    "anger": "😠", "disgust": "🤮", "fear": "😨😱", "happy": "🤗", "joy": "😂",
    "neutral": "😐", "sad": "😔", "sadness": "😔", "shame": "😳", "surprise": "😮"
}

def predict_emotions(docx):
    results = pipe_lr.predict([docx])
    return results[0]

def get_prediction_proba(docx):
    results = pipe_lr.predict_proba([docx])
    return results

def main():
    st.title("Text Emotion Detection")
    st.subheader("Detect Emotions In Text")

    with st.form(key='my_form'):
        raw_text = st.text_area("Type Here")
        submit_text = st.form_submit_button(label='Submit')

    if submit_text:
        if raw_text.strip():  # Check if input is not empty
            col1, col2 = st.columns(2)

            prediction = predict_emotions(raw_text)
            probability = get_prediction_proba(raw_text)

            with col1:
                st.success("Original Text")
                st.write(raw_text)

                st.success("Prediction")
                emoji_icon = emotions_emoji_dict[prediction]
                st.write(f"{prediction}: {emoji_icon}")
                st.write(f"Confidence: {max(probability[0]):.2%}")

            with col2:
                st.success("Prediction Probability")
                proba_df = pd.DataFrame({'emotions': pipe_lr.classes_, 'probability': probability[0]})
                fig = alt.Chart(proba_df).mark_bar().encode(x='emotions', y='probability', color='emotions')
                st.altair_chart(fig, use_container_width=True)
        else:
            st.warning("Please provide some text for analysis.")


if __name__ == '__main__':
   main()
