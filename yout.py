import streamlit as st
import pandas as pd
from processing import  run_full_pipeline,process_pending_videos



import joblib


model = joblib.load('sentiment_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
le = joblib.load('label_encoder.pkl')

def get_final_sentiment(video_url, video_text):
    
    match = df[df['url'].str.strip() == video_url.strip()]
    
    if not match.empty:
       
        sentiment = match.iloc[0]['sentiment_analysis']
        return sentiment, "Database Lookup (100% Match)"
    
    
    else:
       
        text_vector = tfidf.transform([video_text])
        prediction = model.predict(text_vector)
      
        return prediction[0], "AI Prediction (Experimental)"
    

def predict_sentiment(text):
    
    text_vector = tfidf.transform([text])
  
    prediction = model.predict(text_vector)
    
    return le.inverse_transform(prediction)[0]

st.set_page_config(page_title="YouTube AI Search", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #0d0221; color: #00f7ff; }
    .stTextInput>div>div>input { background-color: #1a1a2e; color: #00f7ff; border: 1px solid #ff00c1; }
    .stButton>button { background-color: #ff00c1; color: white; width: 100%; border-radius: 5px; font-weight: bold; }
    .result-box { padding: 20px; border: 1px dashed #00f7ff; border-radius: 10px; background: #16213e; }
    </style>
    """, unsafe_allow_html=True)

with st.sidebar:
    st.title("Admin Tools")
    st.write("Batch process all pending videos in the dataset.")
    if st.button("Run Batch Processing"):
        with st.spinner("Processing all pending videos..."):
            process_pending_videos() 
            st.success("Batch processing completed!")

st.title("Youtube-AI-Analyzer-ECA")
st.write('Enter a YouTube video URL to retrieve key points and sentiment analysis.')


video_url = st.text_input("YouTube Video URL:")


if st.button("Search"):
    if video_url:
        try:
            
            df = pd.read_csv("youtube.csv", encoding='utf-8-sig')
            
           
            result = df[df['url'].str.strip() == video_url.strip()]
            
            if not result.empty:
                st.success("Data retrieved successfully!")
                
                
                points = result.iloc[0]['key_points']
                sentiment = result.iloc[0]['sentiment_analysis']
                title = result.iloc[0]['title']
                st.markdown(f"### Title: {title}")
                
         
                st.markdown('<div>', unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(label="Target Dialect", value="Egyptian (ECA)")
                with col2:
                    st.metric(label="Calculated Sentiment", value=str(sentiment).upper())

                st.markdown('</div>', unsafe_allow_html=True)
                
              
                st.subheader(f"Title: {title}")
                st.subheader("Key Points:")
                st.info(points)
                st.markdown('</div>', unsafe_allow_html=True)
           
            else:
                st.info("Sorry, this URL is not found in the database. Processing now...")
                    
                with st.spinner("Running AI Pipeline (Whisper + Analysis)..."):
                       
                        video_id = video_url.split('=')[-1].split('&')[0]
                        
                       
                        transcription, new_points, new_sentiment = run_full_pipeline(video_url, video_id)
                        
                        
                        new_data = {
                            'video_id': video_id,
                            'url': video_url.strip(),
                            'title': "Processed Video", 
                            'gather_status': 'completed',
                            'transcription': transcription,
                            'key_points': new_points,
                            'sentiment_analysis': new_sentiment
                        }
                        
                       
                        new_df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)
                        new_df.to_csv("youtube.csv", index=False, encoding='utf-8-sig')
                        
                        st.success("Analysis Complete and Saved!")
                        
                        
                        st.markdown(f"### 🎬 Title: Processed Video")
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.subheader("📌 Key Points:")
                        st.write(new_points)
                        st.subheader("🎭 Sentiment Analysis:")
                        st.info(new_sentiment)
                        st.markdown('</div>', unsafe_allow_html=True)
            
                
        except FileNotFoundError:
            st.error("Data file not found.")
    else:
        st.info("Please enter a valid YouTube video URL.")
