import cv2  
import mediapipe as mp  
from mediapipe.python.solutions.face_mesh import FACEMESH_TESSELATION
import time

class FaceTracking:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # Utilisation de la caméra par défaut (indice 0)
        self.pTime = 0
        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(max_num_faces=2)
        self.drawspec = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=2)

        # Configuration de la taille de la vidéo
        self.frameWidth = 1920
        self.frameHeight = 1080
        self.cap.set(3, self.frameWidth)
        self.cap.set(4, self.frameHeight)
        self.cap.set(10, 150)  # Optionnel : ajuster la luminosité

    def run(self):
        while True:
            success, img = self.cap.read()
            if not success:
                print("Erreur lors de la lecture de la vidéo")
                break

            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.faceMesh.process(imgRGB)

            if results.multi_face_landmarks:
                for faceLms in results.multi_face_landmarks:
                    self.mpDraw.draw_landmarks(img, faceLms, FACEMESH_TESSELATION, self.drawspec, self.drawspec)

            cTime = time.time()  # seconde actuelle
            fps = 1 / (cTime - self.pTime) if self.pTime != 0 else 0  # Pour éviter la division par zéro
            self.pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

            cv2.imshow("Image", img)

            if cv2.waitKey(1) == 27:  # Touche 'ESC' pour quitter
                break

        self.cap.release()

# Main
if __name__ == "__main__":
    # Création de l'objet de suivi de visage
    tracker = FaceTracking()
    
    # Exécution du suivi
    tracker.run()
