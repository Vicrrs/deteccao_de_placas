import cv2 as cv
import numpy as np
import imutils as imu
import pytesseract
from matplotlib import pyplot as plt


verbose = True


# Carrega a imagem
imgOriginal = cv.imread("2.jpg")

# Redimensiona a imagem
imgOriginal = imu.resize(imgOriginal, width=500)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem Original", imgOriginal)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Converte a imagem original para tons de conza

imgGray = cv.cvtColor(imgOriginal, cv.COLOR_BGR2GRAY)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem em tons de cinza", imgGray)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Aplica um filtro bilateral - é mais lento que os demais filtros, mas destaca mais as bordas
# Função bilateralFilter(img, galssian1, galssian2, galssian3)

imgFiltrada = cv.bilateralFilter(imgGray, 15, 17, 17)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem com filtro bilateral", imgFiltrada)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Aplicando detecção de bordas com Canny(img,limear minimo, limiar maximo)

imgBordas = cv.Canny(imgFiltrada, 170, 200)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem com bordas por Canny", imgBordas)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Localiza os contronos da imagemde canny com função finde(img, modo_da_aproximação_do_conorno, media_de_aproximação)
# A saida dessa função é a imagem, os contornos e hierarquia dos contornos

contornos, _ = cv.findContours(
    imgBordas.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)


# Depois de localizados os contornos, agora é hora de dezenhalos com função drawContours
# Faz uma copia da imagem original, para dezenhar os contor nela

imgContornos = imgOriginal.copy()
cv.drawContours(imgContornos, contornos, -1, (0, 255, 0), 3)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem com os contornos", imgContornos)
    cv.waitKey(0)
    cv.destroyAllWindows()


contornos = sorted(contornos, key=cv.contourArea, reverse=True)[:30]

# Um variavel para a placa
NumPlacaContornos = None


# Pega os top 30 contornos e desenha as linhas, em verde

imgContornos = imgOriginal.copy()
cv.drawContours(imgContornos, contornos, -1, (0, 255, 0), 3)


# Plota a imagem

if verbose == True:
    cv.imshow("Imagem TOP contornos 30", imgContornos)
    cv.waitKey(0)
    cv.destroyAllWindows()


# Depois de encontrar os contornos principais, TOP 30, é preciso dar loop em todos os contornos,

# até localizar o retangulo da placa

contador = 0
indice = 1


for contorno in contornos:
    perimetro = cv.arcLength(contorno, True)  # Calcula os perimetros polignos
    # Localiza os polignos mais proximos
    proximo = cv.approxPolyDP(contorno, 0.02*perimetro, True)
    # Dentro do polignos encontraados, seleciona o contorno com 4 bordas, a placa

    if len(proximo) == 4:

        NumPlacaContornos = proximo
        # Recorta e armazena os contronos da placa localizada
        print(NumPlacaContornos)
        x, y, w, h = cv.boundingRect(contorno)
        # Cria uma nova imagem com a placa localizada
        imgNova = imgGray[y:y+h, x:x+w]
        # Salva a imagem criada
        cv.imwrite("placa" + str(indice) + ".png", imgNova)
        indice = + 1
        break


imgContornos = imgOriginal.copy()
# print("Imagem ", imgContornos)
# print("Contornos da placa", NumPlacaContornos)

cv.drawContours(imgContornos, [NumPlacaContornos], -1, (0, 255, 0), 3)

if verbose == True:
    cv.imshow("Imagem original com a placa detectada.", imgContornos)
    cv.waitKey(0)
    cv.destroyAllWindows()
