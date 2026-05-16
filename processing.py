import pandas as pd
import yt_dlp
import whisper
import os
import re



df = pd.read_csv('youtube.csv', encoding='utf-8-sig')
model = whisper.load_model("base") 



def get_sentiment_arabic(text):
    """تحليل مشاعر مبني على الكلمات المفتاحية لدعم العربي بدون تعقيد"""
    pos_words = ['جميل', 'رائع', 'ممتاز', 'ذكاء', 'مفيد', 'شكرا', 'عبقري', 'تحفة', 'ممتع', 'صح']
    neg_words = ['سيء', 'ضعيف', 'ممل', 'خطأ', 'فشل', 'خسارة', 'صعب', 'ناقص', 'غلط', 'وحش']
    
    text = str(text).lower()
    score = 0
    
    for word in pos_words:
        if word in text: score += 1
    for word in neg_words:
        if word in text: score -= 1
        
    if score > 0: return "Positive "
    elif score < 0: return "Negative "
    else: return "Neutral "

def get_key_points_smart(text):
    """استخراج أهم الجمل بناءً على علامات الترقيم والطول"""
    if not text or pd.isna(text):
        return "• لا يوجد نص كافٍ لاستخراج النقاط"
    
  
    sentences = re.split(r'[.\n،]', str(text))
    
    points = [s.strip() for s in sentences if len(s.strip()) > 25]
    
    if not points:
        
        words = str(text).split()
        points = [" ".join(words[i:i+12]) for i in range(0, len(words), 12)]

 
    return "\n".join([f"• {p}" for p in points[:5]])



def run_full_pipeline(url, video_id):
   
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{video_id}.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192'
        }],
    }
    
    print(f"🎬 Downloading audio for: {video_id}...")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    actual_filename = f"{video_id}.wav"

    # 2. Transcription (Whisper)
    print(f"Transcribing with Whisper...")
    result = model.transcribe(actual_filename, language='ar')
    transcription = result['text']

    # 3. NLP Analysis (Lightweight Logic)
    print(f"Analyzing sentiment and key points...")
    sentiment = get_sentiment_arabic(transcription)
    key_points = get_key_points_smart(transcription)

    # 4. Cleanup
    if os.path.exists(actual_filename):
        os.remove(actual_filename)

    return transcription, key_points, sentiment

def process_pending_videos():
    
    pending_videos = df[df['gather_status'] == 'metadata_only']
    
    if pending_videos.empty:
        print("No pending videos to process.")
        return

    for index, row in pending_videos.iterrows():
        print(f"\n--- Processing: {row['title']} ---")
        try:
            txt, points, sent = run_full_pipeline(row['url'], row['video_id'])
            
           
            df.at[index, 'transcription'] = txt
            df.at[index, 'key_points'] = points
            df.at[index, 'sentiment_analysis'] = sent
            df.at[index, 'gather_status'] = 'completed'
            
            
            df.to_csv('youtube.csv', index=False, encoding='utf-8-sig')
            print(f"✅ Finished and Saved: {row['video_id']}")
            
        except Exception as e:
            print(f" Error in Pipeline: {str(e)}")


if __name__ == "__main__":
    process_pending_videos()