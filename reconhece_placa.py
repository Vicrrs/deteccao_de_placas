# Bibliotecas
import numpy as np
import cv2
import imutils
import pytesseract

verbose = True

# Imagem original
image = cv2.imread('1.jpg')

# Pré-Processamento
# Resize a imagem - mudar width para 500
image = imutils.resize(image, width=500)

# Show Imagem Original
if verbose == True:
    cv2.imshow("Imagem Original ", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# RGB -> Tons de cinza
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

if verbose == True:
    cv2.imshow("1 - Conversao Tons de cinza", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Processamento de Imagem
# Remoção de ruído com filtro bilateral iterativo (remove o ruído enquanto preserva as bordas)
gray = cv2.bilateralFilter(gray, 11, 17, 17)

if verbose == True:
    cv2.imshow("2 - Filtro Bilateral ", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Encontrar bordas da Imagem em tons de cinza
edged = cv2.Canny(gray, 170, 200)

if verbose == True:
    cv2.imshow("3 - Canny Edges", edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Encontrar contornos baseado nas bordas
cnts, _ = cv2.findContours(
    edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Crie uma cópia da imagem original para desenhar todos os contornos
img1 = image.copy()
cv2.drawContours(img1, cnts, -1, (0, 255, 0), 3)

if verbose == True:
    cv2.imshow("4 - All Contours", img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# classifique os contornos com base em sua área, mantendo a área mínima exigida como '30' (qualquer coisa menor que isso não será considerada)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumPlacaCnt = None  # Sem contorno da placa

# Top 30 Contornos
img2 = image.copy()
cv2.drawContours(img2, cnts, -1, (0, 255, 0), 3)

if verbose == True:
    cv2.imshow("5 - Top 30 Contours", img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# loop sobre os contornos para encontrar a melhor aproximacao do contorno do numero da placa
count = 0
idx = 1
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    # print ("approx = ",approx)
    if len(approx) == 4:  # Selecionar contorno com 4 bordas
        NumPlacaCnt = approx  # Aproximacao da placa

        # Cortar esses contornos e armazená-lo na pasta Imagens recortadas
        x, y, w, h = cv2.boundingRect(c)  # Encontrar as coorde da placa
        new_img = gray[y:y + h, x:x + w]  # Criar uma nova imagem
        cv2.imwrite('Placa' + str(idx) + '.png', new_img)  # Armazenar a imagem
        break


# Desenhando o contorno selecionado na imagem original
# print(NumberPlateCnt)
cv2.drawContours(image, [NumPlacaCnt], -1, (0, 255, 0), 3)

if verbose == True:
    cv2.imshow("Imagem final com a placa detectada", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


Cropped_img_loc = 'Placa' + str(idx) + '.png'

if verbose == True:
    cv2.imshow("Cropped Image ", cv2.imread(Cropped_img_loc))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
