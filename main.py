import os
import cv2
import dlib
import os
import numpy as np

def landmark(frame, stride=10):
    detector = dlib.get_frontal_face_detector()
    file = 'shape_predictor_68_face_landmarks.dat'
    predictor = dlib.shape_predictor(file)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

        landmarks = predictor(gray, face)
        xmin = landmarks.part(48).x
        xmax = landmarks.part(48).x
        ymin = landmarks.part(48).y
        ymax = landmarks.part(48).y
        for n in range(48, 68):

            x = landmarks.part(n).x
            y = landmarks.part(n).y
            if xmin > x:
                xmin = x
            if xmax < x:
                xmax = x
            if ymin > y:
                ymin = y
            if ymax < y:
                ymax = y
            # cv2.circle(frame, (x, y), 4, (255, 0, 0), -1)
    # cv2.imshow(name, frame)
    return frame[ymin - stride:ymax + stride, xmin - stride:xmax + stride]




def extract_number(string):
    number_str = ""
    for char in string:
        if char.isdigit():
            number_str += char
        elif number_str:
            break
    if number_str:
        return number_str
    else:
        return None


def extract_folders_and_files(path):
    components = path.split(os.sep)
    folders = components[:-1]
    file_name = components[-1]
    return folders, file_name
def loop_through_videos(data_folder_path):

    counterFolder=13
    if not os.path.isdir(data_folder_path):
        raise ValueError(f"Invalid data folder path: {data_folder_path}")

    # Use os.walk for efficient directory traversal
    for root, _, files in os.walk(data_folder_path):
        for filename in files:
            # Ensure all extensions are considered, regardless of case
            if filename.lower().endswith(('.mp4', '.avi', '.mkv', '.wmv')):
                video_path = os.path.join(root, filename)

                print(f"Processing video: {video_path}")
                print(f"Number video:{extract_number(video_path)}")

                extract_frames(video_path,extract_folders_and_files(video_path)[0][1],counterFolder,extract_number(video_path),"new_data_frames_28")
        counterFolder+=1
def extract_frames(video_path,FileName,FileNumber,number, output_folder):

    videoCap = cv2.VideoCapture(video_path)

    if not videoCap.isOpened():
        print("Error opening video file")
        return

    os.makedirs(output_folder+"/" + str(FileNumber)+ "/" +number, exist_ok=True)
    os.makedirs(output_folder +"/"+FileName, exist_ok=True)
    fps = videoCap.get(cv2.CAP_PROP_FPS)
    frame_count = int(videoCap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video FPS: {fps}")
    print(f"Total frames: {frame_count}")
    if frame_count >= 30:
        extract_less_frames(videoCap, frame_count, output_folder+"/" + str(FileNumber)+ "/" +number)
    else:
        extract_more_frames(videoCap, frame_count, output_folder+"/" + str(FileNumber)+ "/" +number)
    # for frame_number in range(frame_count):
    #     ret, frame = videoCap.read()
    #
    #     if not ret:
    #         print(f"Error reading frame {frame_number}")
    #         break
    #     frame_filename = os.path.join(output_folder ,number, f"{frame_number:0d}.jpg")
    #
    #     #frame_filename = f"{output_folder}"+"/"+f"{FileName}"+"/"+f"{number}"+"/"+f"{frame_number:0d}.jpg"
    #
    #     lframe = landmark(frame)
    #     lframe = cv2.resize(lframe, (400, 400))
    #
    #     cv2.imwrite(frame_filename, lframe)
    # videoCap.release()

def convert_images_to_video_opencv(image_folder, output_video, frame_rate=30):


  video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (500,300))  # Adjust resolution if needed

  image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
  image_files.sort(key=lambda x: int(x.split('.')[0]))  # Example sorting
  for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)
    video_writer.write(image)

  video_writer.release()
  print(f"Video created successfully: {output_video}")
def extract_less_frames(videoCap,frame_count, output_folder):
    lc=frame_count-30
    b=False
    f=0
    for frame_number in range(frame_count):
        ret, frame = videoCap.read()

        if not ret:
            print(f"Error reading frame {frame_number}")
            break
        if b == True and lc>0:
            b=False
            lc=lc-1
            continue

        frame_filename = f"{output_folder}/{f:0d}.jpg"
        f=f+1
        frame=landmark(frame)
        frame=cv2.resize(frame,(500,300))
        cv2.imwrite(frame_filename,frame)
        b=True

    videoCap.release()
    convert_images_to_video_opencv(output_folder,output_folder+'/'+"f.mp4")
def extract_more_frames(videoCap,frame_count, output_folder):
    mc = 30-frame_count
    f = 0

    for frame_number in range(frame_count):
        ret, frame = videoCap.read()

        if not ret:
            print(f"Error reading frame {frame_number}")
            break
        frame=landmark(frame)
        frame=cv2.resize(frame,(500,300))
        if mc>0:
            frame_filename = f"{output_folder}/{f:0d}.jpg"
            f=f+1
            cv2.imwrite(frame_filename, frame)
            mc=mc-1
        frame_filename = f"{output_folder}/{f:0d}.jpg"
        f=f+1
        cv2.imwrite(frame_filename, frame)
    blackframe=np.zeros((300,500))
    for i in range(mc):
        frame_filename = f"{output_folder}/{f:0d}.jpg"
        f = f + 1
        cv2.imwrite(frame_filename, blackframe)
    convert_images_to_video_opencv(output_folder,output_folder+'/'+"f.mp4")
    videoCap.release()
loop_through_videos("FinalData")
# model = VGG19(weights='imagenet',include_top=False)
#
# img_path = 'test.png'
# img = image.load_img(img_path,target_size=(224,224))
# x = image.img_to_array(img)
# x = np.expand_dims(x,axis=0)
# x = preprocess_input(x)
#
# features = model.predict(x)
# print(features.shape)
#
#
# extract_frames("test.mp4", 'C:/Users/kerolos/Desktop/kk')

data0=np.load("Old_Data_With_Feature_Extraction_using_vggg/0_data.npy")
data1=np.load("Old_Data_With_Feature_Extraction_using_vggg/1_data.npy")
data2=np.load("Old_Data_With_Feature_Extraction_using_vggg/2_data.npy")
data3=np.load("Old_Data_With_Feature_Extraction_using_vggg/3_data.npy")
data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/5_data.npy")
data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/6_data.npy")
data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/7_data.npy")
data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/8_data.npy")
data4=np.load("Old_Data_With_Feature_Extraction_using_vggg/9_data.npy")

#data3=np.load("Old_Data_With_Feature_Extraction_using_vggg/3_data.npy")
data=np.concatenate((data0, data1, data2), axis=0)
del data0,data1,data2
label0=np.load("Old_Data_With_Feature_Extraction_using_vggg/0_labels.npy")
label1=np.load("Old_Data_With_Feature_Extraction_using_vggg/1_labels.npy")
label2=np.load("Old_Data_With_Feature_Extraction_using_vggg/2_labels.npy")
label3=np.load("Old_Data_With_Feature_Extraction_using_vggg/3_labels.npy")
label4=np.load("Old_Data_With_Feature_Extraction_using_vggg/5_labels.npy")
label5=np.load("Old_Data_With_Feature_Extraction_using_vggg/6_labels.npy")
label6=np.load("Old_Data_With_Feature_Extraction_using_vggg/7_labels.npy")
label7=np.load("Old_Data_With_Feature_Extraction_using_vggg/8_labels.npy")
label8=np.load("Old_Data_With_Feature_Extraction_using_vggg/9_labels.npy")
labels=np.concatenate((label0[:11], label1, label2), axis=0)
del label0,label1,label2