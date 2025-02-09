import cv2
import mediapipe as mp
import pandas as pd

def get_palm_cut(image):
    coordinate = []
    # Inizializza MediaPipe Hands
    mp_hands = mp.solutions.hands
    # Converti l'immagine in RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Rileva la mano
    #cv2.imshow("Hand Detection", image_rgb)
    #cv2.waitKey(0)
    #print(image_rgb)
    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5) as hands:
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    h, w, _   = image.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    #print(f"ID Punto {idx}: x={x}, y={y}")
                    coordinate.append((x, y))
                                        # Disegna i punti sulla mano
        #punto 0 1 5 17

    h, w, _ = image.shape
    image_vertex = []
    index_point_to_use = [0, 2, 5, 17]

    for x in index_point_to_use:
        try:
            if coordinate[x][0] > w:
                image_vertex[index_point_to_use.index(x)] = (w, coordinate[x][1])
            elif coordinate[x][0] < 0:
                image_vertex[index_point_to_use.index(x)] = (0, coordinate[x][1])
            if coordinate[x][1] > h:
                image_vertex[index_point_to_use.index(x)] = (coordinate[x][0], h)
            elif coordinate[x][1] < 0:
                image_vertex[index_point_to_use.index(x)] = (coordinate[x][0], 0)
            else:
                image_vertex.append(coordinate[x])
        except:
                # To handle critical errors on the mediapipe palmar point detection
                match x:
                        case 0:
                            image_vertex.insert(0, (w,0))
                        case 2:
                            image_vertex.insert(1, (0,0))
                        case 5:
                            image_vertex.insert(2, (0,h))
                        case 17:
                            image_vertex.insert(3, (w,h))

    # x, y, w, h 

    x = min(image_vertex[0][0], image_vertex[1][0], image_vertex[2][0], image_vertex[3][0])
    y = min(image_vertex[0][1], image_vertex[1][1], image_vertex[2][1], image_vertex[3][1])
    w = max(image_vertex[0][0], image_vertex[1][0], image_vertex[2][0], image_vertex[3][0]) - x
    h = max(image_vertex[0][1], image_vertex[1][1], image_vertex[2][1], image_vertex[3][1]) - y

    height, width, _ = image.shape

    
    if x < 0 or y < 0 or x + w > width or y + h > height:
        print("Le coordinate di ritaglio sono fuori dai limiti dell'immagine")
    else:
        # Esegui il ritaglio dell'immagine
        cropped_image = image[y:y+h, x:x+w]

    return cropped_image


def create_palm_cut_dataset(image_path:str, palm_cut_image_path:str):
    df = pd.read_csv('HandInfo.csv')

    for image_name in df['imageName']:
        image = cv2.imread(image_path + '/' + image_name)
        palmar_dorsal = str(df.loc[df['imageName'] == image_name, 'aspectOfHand'].values[0])

        if palmar_dorsal.find('palmar') != -1:
            image = get_palm_cut(image)
        cv2.imwrite(palm_cut_image_path + '/' + image_name, image)

    print("Palm Cut Completed\n")
          
#create_palm_cut_dataset(image_path="/home/mattpower/Downloads/Hands", palm_cut_image_path="/home/mattpower/Downloads/PalmCutHands")