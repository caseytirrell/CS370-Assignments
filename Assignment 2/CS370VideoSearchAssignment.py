import traceback
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from pytube import YouTube
import psycopg2
import os
import uuid
from urllib.error import URLError



model = ResNet50(weights='imagenet', include_top=False)

def genEmbeddings(frames):
    embeddings = []
    for frame in frames:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)

        embedding = model.predict(img)
        embeddings.append(embedding.flatten())
    return embeddings
''''
def storeEmbeddings(videoID, embeddings, outputPath):
    vDirectory = os.path.join(outputPath, videoID)
    if not os.path.exists(vDirectory):
        os.makedirs(vDirectory)
    embeddingsFile = os.path.join(vDirectory, 'frames.npy')
    np.save(embeddingsFile, np.array(embeddings))
    print(f"Embeddings successfully saved to {embeddingsFile}")
'''
def storeEmbeddings(videoID, embeddings, dbParams):
    conn = psycopg2.connect(**dbParams)
    cursor = conn.cursor()
    for i, j in enumerate(embeddings):
        embeddingList = j.tolist()
        cursor.execute(
            "INSERT INTO videoembeddings (videoID, frameNumber, embedding) VALUES (%s, %s, %s)",
            (videoID, i, embeddingList)
        )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Embeddings for video {videoID} successfully saved to the database.")

def extractFrames(videoPath, interval=1):
    count = 0
    frames = []    
    videoCapture = cv2.VideoCapture(videoPath)
    success, image = videoCapture.read()
    while success:
        if count % interval == 0:
            frames.append(image)
        success, image = videoCapture.read()
        count += 1
    videoCapture.release()
    return frames

def process_video(videoPath, videoID, dbParams, interval=1):
    frames = extractFrames(videoPath, interval)
    embeddings = genEmbeddings(frames)
    storeEmbeddings(videoID, embeddings, dbParams)

def download(url, outputPath):   
    try:
        youtube = YouTube(url)
        video = youtube.streams.get_highest_resolution()
        safeTitle = "".join(x for x in video.title if x.isalnum() or x in " -_").rstrip()
        fName = safeTitle + ".mp4"

        try:
            video.download(outputPath, filename=fName)
            print(f"Video Downloaded: {video.title}")
        except URLError as e:
            print(f"Failed to download video: {e}")
            return None

        try:
            if 'en' in youtube.captions:
                englishCaps = youtube.captions['en']
                englishCapsSRT = englishCaps.generate_srt_captions()

                captionsFile = os.path.join(outputPath, f"{safeTitle}.srt")
                with open(captionsFile, "w") as f:
                    f.write(englishCapsSRT)
                print(f"Captions Successfully Downloaded: {captionsFile}")
            else:
                print("No english captions available...")
        except KeyError as e:
            print(f"Failed to process captions due to a KeyError: {e}")
        
        return safeTitle
        
    except Exception as e:
        print(f"An error has occured: {e}")
        traceback.print_exc()
        return None

def downloadVideosAndCaptions(urls, outputPath):
    titles = []
    for url in urls:
        title = download(url, outputPath)
        if title:
            titles.append(title)
    return titles

if __name__ == "__main__":
    outputPath = "/Users/caseytirrell/Documents/CS370-Assignments/Assignment 2"
    dbParams = {
        'dbname': 'cs370videodb',
        'user': 'caseytirrell',
        'password': 'Ct1234567!',
        'host': 'localhost'
    }
    youtubeURLs = [
        "https://www.youtube.com/watch?v=wbWRWeVe1XE",
        "https://www.youtube.com/watch?v=FlJoBhLnqko",
        "https://www.youtube.com/watch?v=Y-bVwPRy_no"
    ]

    downloadTitles = downloadVideosAndCaptions(youtubeURLs, outputPath)

    for title in downloadTitles:
        if title is not None:
            videoPath = os.path.join(outputPath, title + ".mp4")
            videoID = str(uuid.uuid4())
            process_video(videoPath, videoID, dbParams, interval=10)