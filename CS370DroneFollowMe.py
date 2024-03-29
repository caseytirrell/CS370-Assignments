import traceback
import numpy as np
import cv2
import os
import tensorflow as tf
import tensorflow_hub as tfhub
from filterpy.kalman import KalmanFilter
from pytube import YouTube
from urllib.error import URLError


TARGET_SIZE = (320, 320)
modelURL = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1"
detector = tfhub.load(modelURL)

IDtoClassName = {
    2: "bicycle",
    3: "car"
}

def downloadVideo(url, outputPath):
    print(f"Starting {url} Download...")
    try:
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if not video:
            print("No suitable video found...")
            return None
        print(f"Downloading {video.title} to {outputPath}...")
        video.download(outputPath)
        filename = os.path.join(outputPath, video.default_filename)
        print(f"Downloaded successfully to {filename}...")
        return filename
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        traceback.print_exc()
        return None

def detectObjects(frame, detector):
    print("Detecting objects in frame...")
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis,...]

    detections = detector(input_tensor)
    bbox = detections['detection_boxes'][0].numpy()
    classIDs = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()

    result = []
    for i in range(scores.shape[0]):
        if scores[i] >= 0.5:  # confidence threshold
            result.append({'bbox': bbox[i], 'class_id': classIDs[i], 'score': scores[i]})
    return result

def kalmanFilter():
    print("Starting Kalman Filter...")
    kf = KalmanFilter(dim_x=7, dim_z=4)
    kf.F = np.eye(7)
    kf.H = np.eye(4, 7)
    kf.R = np.eye(4) * 0.1
    kf.P *= 1000 
    kf.Q = np.eye(7) * 0.1
    return kf

def extractAndProcessFrames(videoPath, kalmanFilters, trajectories):
    print(f"Processing video {videoPath}")
    cap = cv2.VideoCapture(videoPath)
    processedFrames = []
    frameCount = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frameCount += 1
        print(f"Processsing frame {frameCount}")
        detections = detectObjects(frame, detector)

        if not kalmanFilters:
            for i in detections:
                kf = kalmanFilter()
                kalmanFilters.append(kf)
                trajectories.append([])
        
        for i in detections:
            classID = i['class_id']
            if classID in [2, 3]:
                x, y, height, width = i['bbox']
                kfIndex = 0
                kf = kalmanFilters[kfIndex]
                kf.predict()
                kf.update(np.array([x + (width / 2), y + (height / 2), width, height]))
                trajectories[kfIndex].append(kf.x[:2].tolist())
        
        for i in trajectories:
            for j in range(1, len(i)):
                if i[j] is not None and i[j - 1] is not None:
                    start = (int(i[j - 1][0]), int(i[j - 1][1]))
                    end = (int(i[j][0]), int(i[j][1]))
                    cv2.line(frame, start, end, (0, 255, 0), 2)
        
        processedFrames.append(frame)
    
    cap.release()
    return processedFrames

def processVideo(videoPath):
    print(f"Starting to process video {videoPath}")
    kalmanFilters = []
    trajectories = []
    processedFrames = extractAndProcessFrames(videoPath, kalmanFilters, trajectories)

    width= processedFrames[0].shape[1]
    height = processedFrames[0].shape[0]

    output_file_name = os.path.splitext(os.path.basename(videoPath))[0] + '_tracked.mp4'
    output_path = os.path.join(os.path.dirname(videoPath), output_file_name)

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 25, (width, height))

    for i in processedFrames:
        out.write(i)

    print("Video processing complete")
    out.release()


def downloadVideosAndCaptions(urls, outputPath):
    titles = []
    for url in urls:
        title = downloadVideo(url, outputPath)
        if title:
            titles.append(title)
    return titles

if __name__ == "__main__":
    outputPath = "/Users/caseytirrell/Documents/CS370-Assignments/Assignment 3"
    youtubeURLs = [
        "https://www.youtube.com/watch?v=2hQx48U1L-Y",
        "https://www.youtube.com/watch?v=2NFwY15tRtA",
        "https://www.youtube.com/watch?v=5dRramZVu2Q"
    ]
    downloadVideos = downloadVideosAndCaptions(youtubeURLs, outputPath)
    
    for video in downloadVideos:
        if video is not None:
            videoPath = os.path.join(outputPath, video)
            processVideo(videoPath)
