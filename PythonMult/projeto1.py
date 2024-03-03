import matplotlib.pyplot as plt #3.1
import matplotlib.colors as clr #3.2
import numpy as np #decoder
import cv2 
import scipy.fftpack as fft

#-----Ex.2-----

#Encoder

def splitRGB(img): #separar matrizes
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    return R, G, B

#Decoder

def joinRGB(R, G, B):
    nl, nc= R.shape #n linhas e colunas
    imgRec= np.zeros((nl, nc, 3), dtype=np.uint8) #array inicializado a 0

    imgRec[:, :, 0] = R #img reconstruida no canal 0 para vermelho
    imgRec[:, :, 1] = G
    imgRec[:, :, 2] = B

    return imgRec



def showImg(img, cmap= None, caption= ""):
    plt.figure()
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(caption)

#5.1
def RGB_to_yCbCr(R, G, B):
    #converter para YCbCr
    #depois alterar pro do slide 46 M1.1 (com divisão)
    Y= 0.299*R + 0.587*G + 0.114*B
    #Cb= -0.168736*R - 0.331264*G + 0.5*B + 128
    Cb = ((B - Y) / 1.772) + 128
    #Cr= 0.5*R - 0.418688*G - 0.081312*B + 128
    Cr = ((R - Y) / 1.402) + 128
    return Y, Cb, Cr

def yCbCr_to_RGB(Y, Cb, Cr):
    #converter para RGB
    R= np.clip((Cr-128)*1.402 + Y, 0, 255) #clip para valores entre 0 e 255
    B= np.clip((Cb-128)*1.772 + Y, 0, 255)
    G= np.clip((Y - (0.299*R) - (0.114*B))/0.587, 0, 255)
    return R, G, B

# ---Ex 6---
def downsampling(Y,Cb, Cr, n):
    if n == 444:
        return Y, Cb, Cr
    if n == 422:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=1, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_d, Cr_d
    if n == 420:
        Cb_d = cv2.resize(Cb, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        Cr_d = cv2.resize(Cr, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_d, Cr_d
    return print("valor de n errado!")

def upsampling(Y,Cb, Cr, n):
    if n == 422:
        Cb_u = cv2.resize(Cb,None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr,None, fx=2, fy=1, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_u, Cr_u
    if n == 420:
        Cb_u = cv2.resize(Cb,None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        Cr_u = cv2.resize(Cr,None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        return Y,Cb_u, Cr_u
    return print("valor de n errado!")

# ---- Ex 7 ----

#7.1.1
def dct(Y, Cb, Cr):
    Y_dct = fft.dct(fft.dct(Y, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cb_dct = fft.dct(fft.dct(Cb, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cr_dct = fft.dct(fft.dct(Cr, axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct


#7.1.2
def idct(Y_dct, Cb_dct, Cr_dct):
    Y = fft.idct(fft.idct(Y_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cb = fft.idct(fft.idct(Cb_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    Cr = fft.idct(fft.idct(Cr_dct, axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y, Cb, Cr




#ortho = normalização
#axis=0 -> aplicar a dct/idct a cada linha
#axis=1 -> aplicar a dct/idct a cada coluna
#chama a função dct duas vezes para aplicar a dct a cada linha e coluna

#7.2

#7.2.1
#calcular a DCT em blocos BSxBS
def dctBlocks(Y, Cb, Cr, BS):
    Y_dct = np.zeros(Y.shape)
    Cb_dct = np.zeros(Cb.shape)
    Cr_dct = np.zeros(Cr.shape)
    for i in range(0, Y.shape[0], BS):
        for j in range(0, Y.shape[1], BS):
            Y_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Y[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    #outro ciclo, pois após o downsampling, dimensão de Cb e Cr podem ser diferentes de Y
    for i in range(0, Cb.shape[0], BS):
        for j in range(0, Cb.shape[1], BS):
            Cb_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Cb[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
            Cr_dct[i:i+BS, j:j+BS] = fft.dct(fft.dct(Cr[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct

#7.2.2
#calcular a DCT inversa em blocos BSxBS
def idctBlocks(Y, Cb, Cr, BS):
    Y_dct = np.zeros(Y.shape)
    Cb_dct = np.zeros(Cb.shape)
    Cr_dct = np.zeros(Cr.shape)
    for i in range(0, Y.shape[0], BS):
        for j in range(0, Y.shape[1], BS):
            Y_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Y[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    for i in range(0, Cb.shape[0], BS):
        for j in range(0, Cb.shape[1], BS):
            Cb_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Cb[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
            Cr_dct[i:i+BS, j:j+BS] = fft.idct(fft.idct(Cr[i:i+BS, j:j+BS], axis=0, norm='ortho'), axis=1, norm='ortho')
    return Y_dct, Cb_dct, Cr_dct 


def encoder(img):
    #3.2 e 3.3
    if (input("Mostrar imagem original com colormap especifico? (s/n): ") in "sS"):
        cm_N= int(input("Introduza o valor de N para o colormap: "))
        cm_red = input("Introduza o valor de cor para o vermelho (0-1): ")
        cm_green = input("Introduza o valor de cor para o verde (0-1): ")
        cm_blue = input("Introduza o valor de cor para o azul (0-1): ")
        cm_cor = (float(cm_red), float(cm_green), float(cm_blue))
        cm_user = clr.LinearSegmentedColormap.from_list("user", [(0,0,0), cm_cor], N= cm_N)
        showImg(img, cmap= cm_user, caption= "Imagem original com colormap do user")
        plt.show(block= False)

    R, G, B= splitRGB(img)
    #Padding caso dimensão não seja multiplo de 32x32 (copia a ultima linha/coluna para preencher o espaço em falta)
    #4.1
    nl, nc= R.shape #n linhas e colunas
    nl_pad= 32-(nl%32)
    nc_pad= 32-(nc%32)
    R= np.pad(R, ((0, nl_pad), (0, nc_pad)), mode= 'edge') #edge= copia a ultima linha/coluna
    G= np.pad(G, ((0, nl_pad), (0, nc_pad)), mode= 'edge')
    B= np.pad(B, ((0, nl_pad), (0, nc_pad)), mode= 'edge')

    cm_red= clr.LinearSegmentedColormap.from_list("red", [(0,0,0), (1,0,0)], N= 256) #color, (range de cores rgb do vermelho)
    cm_green = clr.LinearSegmentedColormap.from_list("green", [(0, 0, 0), (0, 1, 0)], N=256) #256 = 2^8 (8 bits de cores)
    cm_blue = clr.LinearSegmentedColormap.from_list("blue", [(0, 0, 0), (0, 0, 1)], N=256)
    if (input("Mostrar imagens RGB? (s/n): ") in "sS"):
        showImg(R, cmap= cm_red, caption= "Red")
        showImg(G, cmap= cm_green, caption= "Green")
        showImg(B, cmap= cm_blue, caption= "Blue")
        plt.show(block= False)

    #5.3
    Y , Cb, Cr= RGB_to_yCbCr(R, G, B)
    cm_grey= clr.LinearSegmentedColormap.from_list("grey", [(0,0,0), (1,1,1)], N= 256)
    if(input("Mostrar imagens YCbCr? (s/n): ") in "sS"):
        showImg(Y, cmap= cm_grey, caption= "Y")
        showImg(Cb, cmap= cm_grey, caption= "Cb")
        showImg(Cr, cmap= cm_grey, caption= "Cr")
        plt.show(block= False)

    #6.3
    n= int(input("Introduza o valor de n para o downsampling (420, 422, 444): "))
    Y_d, Cb_d, Cr_d = downsampling(Y, Cb, Cr, n)
    if(input("Mostrar imagens YCbCr downsampling? (s/n): ") in "sS"):
        showImg(Y_d, cmap= cm_grey, caption= "Y downsampling")
        showImg(Cb_d, cmap= cm_grey, caption= "Cb downsampling")
        showImg(Cr_d, cmap= cm_grey, caption= "Cr downsampling")
        plt.show(block= False)

    #7.1.3
    Y_dct, Cb_dct, Cr_dct = dct(Y_d, Cb_d, Cr_d)
    if(input("Mostrar imagens YCbCr DCT? (s/n): ") in "sS"):
        showImg(np.log(abs(Y_dct) + 0.0001), cmap= cm_grey, caption= "Y DCT")
        showImg(np.log(abs(Cb_dct) + 0.0001), cmap= cm_grey, caption= "Cb DCT")
        showImg(np.log(abs(Cr_dct) + 0.0001), cmap= cm_grey, caption= "Cr DCT")
        plt.show(block= False)
    #7.2.3
    BS= int(input("Introduza o valor de BS para os blocos (8, 64): "))
    Y_dct8, Cb_dct8, Cr_dct8 = dctBlocks(Y_d, Cb_d, Cr_d, BS)
    if(input("Mostrar imagens YCbCr DCT {}x{}? (s/n): ".format(BS, BS)) in "sS"):
        showImg(np.log(abs(Y_dct8) + 0.0001), cmap= cm_grey, caption= "Y DCT 8x8")
        showImg(np.log(abs(Cb_dct8) + 0.0001), cmap= cm_grey, caption= "Cb DCT 8x8")
        showImg(np.log(abs(Cr_dct8) + 0.0001), cmap= cm_grey, caption= "Cr DCT 8x8")
        plt.show(block= False)
    

def quantizationY_matrix_gen(qf):
    s = get_scale(qf)
    Q_Y = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                    [12, 12, 14, 19, 26, 58, 60, 55],
                    [14, 13, 16, 24, 40, 57, 69, 56],
                    [14, 17, 22, 29, 51, 87, 80, 62],
                    [18, 22, 37, 56, 68, 109, 103, 77],
                    [24, 35, 55, 64, 81, 104, 113, 92],
                    [49, 64, 78, 87, 103, 121, 120, 101],
                    [72, 92, 95, 98, 112, 100, 103, 99]])
    
    Qs = np.floor((s * Q_Y + 50 ) / 100)
    Qs[Qs == 0] = 1

    return Qs

def quantizationCbCr_matrix_gen(qf):
    s= get_scale(qf)
    Q_CbCr = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                       [18,21,26,66,99,99,99,99],
                        [24,26,66,99,99,99,99,99],
                        [47,66,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99],
                        [99,99,99,99,99,99,99,99]])
    Qs = np.floor((s * Q_CbCr + 50 ) / 100)
    Qs[Qs == 0] = 1               
    return Qs    

def get_scale(qf):
    if qf < 1:
        qf = 1
    elif qf > 100:
        qf = 100
    
    if qf < 50:
        return (100 - qf)/50
    if qf >= 50:
        return 50/qf

#é retornado Y_q, Cb_q, Cr_q
def quantization(Y_dct_block, Cb_dct_block, Cr_dct_block, qf):
    return np.round(Y_dct_block/quantizationY_matrix_gen(qf)), np.round(Cb_dct_block/quantizationCbCr_matrix_gen(qf)), np.round(Cr_dct_block/quantizationCbCr_matrix_gen(qf))
#acho q a desquantização está errada

#é retornado Y_dct_block, Cb_dct_block, Cr_dct_block
def dequantization(Y_dct_block, Cb_dct_block, Cr_dct_block, qf):
    return Y_dct_block*quantizationY_matrix_gen(qf), Cb_dct_block*quantizationCbCr_matrix_gen(qf), Cr_dct_block*quantizationCbCr_matrix_gen(qf)



def decoder(R,G,B, tamOriginal):
    imgRec= joinRGB(R, G, B)
    #4.2
    #remover padding
    imgRec= imgRec[:tamOriginal[0], :tamOriginal[1], :]
    return imgRec


#Main
def main():
    # -----Ex.3-----
    #3.1
    #Ler imagens
    fname= "airport.bmp"
    img= plt.imread("imagens/" + fname)
    showImg(img, caption="Imagem original: " + fname)
    encoder(img)
    

if __name__ == "__main__":
    main()