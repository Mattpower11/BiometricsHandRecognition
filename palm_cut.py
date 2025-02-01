import cv2
import mediapipe as mp

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


    #print(len(coordinate))
    #print(coordinate)
    h, w, _ = image.shape
    for x in [0,1,5,17]:
        #print(coordinate[x])

        if coordinate[x][0] > w:
            coordinate[x] = (w, coordinate[x][1])
        elif coordinate[x][0] < 0:
            coordinate[x] = (0, coordinate[x][1])

        if coordinate[x][1] > h:
            coordinate[x] = (coordinate[x][0], h)
        elif coordinate[x][1] < 0:
            coordinate[x] = (coordinate[x][0], 0)
        
    # x, y, w, h 

    x = min(coordinate[0][0], coordinate[1][0], coordinate[5][0], coordinate[17][0])
    y = min(coordinate[0][1], coordinate[1][1], coordinate[5][1], coordinate[17][1])
    w = max(coordinate[0][0], coordinate[1][0], coordinate[5][0], coordinate[17][0]) - x
    h = max(coordinate[0][1], coordinate[1][1], coordinate[5][1], coordinate[17][1]) - y

    height, width, _ = image.shape

    
    if x < 0 or y < 0 or x + w > width or y + h > height:
        print("Le coordinate di ritaglio sono fuori dai limiti dell'immagine")
    else:
        # Esegui il ritaglio dell'immagine
        cropped_image = image[y:y+h, x:x+w]

        # Mostra l'immagine ritagliata
        
        #cv2.destroyAllWindows()
    resized_image = cv2.resize(cropped_image, (150, 150))
    #cv2.imshow('Immagine Ritagliata', resized_image)
    #cv2.waitKey(0)
    #cv2.imwrite(r'D:\Users\Patrizio\Desktop\cropped_image.jpg', resized_image)            
    return resized_image


#image_path = r'D:\Users\Patrizio\Desktop\samp\Hand_0000553.jpg'  # Sostituisci con il tuo file
#get_palm_cut(cv2.imread(r'D:\Users\Patrizio\Desktop\samp\Hand_0000553.jpg'))