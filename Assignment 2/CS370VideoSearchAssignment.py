import traceback
import numpy as np
import cv2
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from pytube import YouTube
import psycopg2
import os
import uuid
from urllib.error import URLError

model = ResNet50(weights='imagenet', include_top=False)

#CNN Model
modelURL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = tfhub.load(modelURL)

IDtoClassName = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}


def detectObjects(frame, detector):

    convertedResult = {}
    detections = []
    
    #Convert to the right format
    convertedImg = tf.image.convert_image_dtype(frame, tf.float32)[tf.newaxis, ...]
    result = detector(convertedImg)

    numDetections = int(result["num_detections"].numpy())
    boundingBox = result["detection_boxes"].numpy()[0]
    ID = result["detection_classes"].numpy()[0]
    confidenceScore = result["detection_scores"].numpy()[0]

    #detection_scores is returned by the R-CNN model and represents a confidence level for the detection
    for i in range(numDetections):
        #Threshold set to 50%
        if confidenceScore[i] >= 0.5:
                score = confidenceScore[i]
                bbox = boundingBox[i]
                classID = int(ID[i])
                className = IDtoClassName.get(classID, "Unknown")

                detections.append({
                    'classID': classID,
                    'className': className,
                    'bbox': bbox,
                    'score': score
                })            
    return detections    

def storeDetections(videoID, frameData, dbParams):
    conn = psycopg2.connect(**dbParams)
    cursor = conn.cursor()

    #loops through the index ad the item themselves
    for frame, (timestamp, detections) in enumerate(frameData):
        for detection in detections:
            score = detection['score']
            bbox = detection['bbox']
            classID = detection['classID']
            className = IDtoClassName.get(classID, "Unknown")

            cursor.execute(
                "INSERT INTO detectedObjects (vidId, frameNum, timestamp, detectedObjId, detectedObjClass, confidence, bbox) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (videoID, frame, timestamp, classID, className, score, bbox)
            )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Object Detection for for video {videoID} successfully saved to the database.")

def genEmbeddings(frames):
    embeddings = []
    #preprocessing
    for frame in frames:
        #correcting color mode
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #resizing
        img = cv2.resize(img, (224, 224))
        #convert to numpy array
        img = image.img_to_array(img)
        #Reshaping
        img = np.expand_dims(img, axis=0)
        #normalization and scaling
        img = preprocess_input(img)

        embedding = model.predict(img)
        embeddings.append(embedding.flatten())
    return embeddings

def storeEmbeddings(videoID, embeddings, dbParams):
    conn = psycopg2.connect(**dbParams)
    cursor = conn.cursor()
    for i, j in enumerate(embeddings):
        embeddingList = j.tolist()
        cursor.execute(
            "INSERT INTO vidembeddings (videoID, frameNumber, embedding) VALUES (%s, %s, %s)",
            (videoID, i, embeddingList)
        )
    conn.commit()
    cursor.close()
    conn.close()
    print(f"Embeddings for video {videoID} successfully saved to the database.")

def extractFrames(videoPath, interval=1):

    count = 0
    frames = []    
    timestamps = []
    videoCapture = cv2.VideoCapture(videoPath)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    success, image = videoCapture.read()
    while success:
        frames.append(image)
        timestamps.append(count / fps)
        success, image = videoCapture.read()
        count += 1
    videoCapture.release()
    return frames, timestamps

def process_video(videoPath, videoID, dbParams, interval=1):
    frameData = []

    frames, timestamps = extractFrames(videoPath, interval)
    embeddings = genEmbeddings(frames)
    storeEmbeddings(videoID, embeddings, dbParams)

    for i, frame in enumerate(frames):
        detections = detectObjects(frame, detector)
        timestamp = timestamps[i]
        frameData.append((timestamp, detections))
    storeDetections(videoID, frameData, dbParams)


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